import http.client
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timezone

import numpy as np

from wmal.monitor.catalog import RunCatalog
from wmal.monitor.contracts import SimFrame
from wmal.monitor.frame_buffer import LatestFrameBuffer
from wmal.monitor.server import MonitorServer


class MonitorServerTests(unittest.TestCase):
    def test_stopped_source_does_not_leave_an_unfinished_task_running(self):
        from wmal.monitor.publisher import FramePublisher
        publisher = FramePublisher(self.buffer, self.run_id, 'overview')
        self.server.register_source(self.run_id, publisher)
        publisher.close()
        data = json.loads(self.request('/api/runs/' + self.run_id + '/events')[2])
        self.assertEqual(data['summary']['tasks'][0]['status'], 'incomplete')

    def test_each_reader_receives_rotation_reset(self):
        base = '/api/runs/' + self.run_id + '/events?after=1'
        self.request('/api/runs/' + self.run_id + '/events')
        replacement = self.run / 'replacement.jsonl'
        replacement.write_text('{"event":"new_source"}\n')
        replacement.replace(self.log)
        first, second = [json.loads(self.request(base)[2]) for _ in range(2)]
        self.assertTrue(first['reset'])
        self.assertTrue(second['reset'])
        self.assertEqual(second['events'][0]['event'], 'new_source')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / 'experiment'
        self.run.mkdir()
        self.log = self.run / 'events.jsonl'
        self.log.write_text('{"event":"task_started","payload":{"task_id":"t1","max_cycles":10}}\n', encoding='utf-8')
        self.catalog = RunCatalog(self.root)
        self.run_id = self.catalog.list_runs()[0]['run_id']
        self.buffer = LatestFrameBuffer()
        self.server = MonitorServer(self.catalog, self.buffer, port=0).start()

    def tearDown(self):
        self.server.shutdown()
        self.temp.cleanup()

    def request(self, path, method='GET', headers=None):
        connection = http.client.HTTPConnection(*self.server.address, timeout=3)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_catalog_and_incremental_event_cursor(self):
        status, _, data = self.request('/api/runs')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)['runs'][0]['run_id'], self.run_id)
        base = '/api/runs/' + self.run_id
        first = json.loads(self.request(base + '/events?after=0')[2])
        self.assertEqual(len(first['events']), 1)
        self.assertEqual(first['summary']['tasks'][0]['status'], 'running')
        with self.log.open('a') as stream:
            stream.write('{"event":"task_result","payload":{"task_id":"t1","status":"succeeded","cycles":3}}\n')
        second = json.loads(self.request(base + '/events?after=1')[2])
        self.assertEqual([row['event'] for row in second['events']], ['task_result'])
        self.assertEqual(second['summary']['tasks'][0]['status'], 'succeeded')

    def test_latest_frame_and_metadata_are_from_the_selected_run(self):
        self.buffer.publish(SimFrame(self.run_id, 'episode1', 7, 1.4,
                                    datetime.now(timezone.utc).isoformat(), 'overview',
                                    np.zeros((8, 8, 3), dtype=np.uint8)))
        status, headers, data = self.request('/api/live/' + self.run_id + '/frame.png')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'image/png')
        self.assertEqual(headers['X-Episode-ID'], 'episode1')
        self.assertEqual(float(headers.get('X-Sim-Time-S', -1)), 1.4)
        self.assertIn('X-Captured-At-UTC', headers)
        self.assertTrue(data.startswith(b'\x89PNG'))
        live = json.loads(self.request('/api/live/' + self.run_id + '/status')[2])
        self.assertEqual(live['step_id'], 7)
        self.assertEqual(live['state'], 'live')
        self.assertEqual(self.request('/api/live/unknown/frame.png')[0], 404)

    def test_sse_emits_a_real_event_snapshot(self):
        connection = http.client.HTTPConnection(*self.server.address, timeout=3)
        connection.request('GET', '/api/runs/' + self.run_id + '/stream')
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers['Content-Type'], 'text/event-stream')
        lines = []
        while True:
            line = response.readline()
            lines.append(line)
            if line == b'\n':
                break
        self.assertIn(b'task_started', b''.join(lines))
        connection.close()

    def test_root_boundary_origin_and_read_only_routes(self):
        base = '/api/runs/' + self.run_id
        self.assertEqual(self.request(base + '/artifacts/%2e%2e/outside.json')[0], 400)
        self.assertEqual(self.request('/api/runs', headers={'Origin': 'https://external.example'})[0], 403)
        self.assertEqual(self.request('/api/runs', headers={'Host': 'external.example'})[0], 403)
        self.assertEqual(self.request('/api/execute', method='POST')[0], 405)
        self.assertEqual(self.request(base + '/events?after=nan')[0], 400)

    def test_secret_fields_are_redacted_in_json_artifacts(self):
        (self.run / 'task_001.json').write_text(json.dumps({
            'api_key': 'private-value', 'state': {'status': 'succeeded'},
            'authorization': 'Bearer private-value'}))
        status, _, data = self.request('/api/runs/' + self.run_id + '/artifacts/task_001.json')
        self.assertEqual(status, 200)
        self.assertNotIn(b'private-value', data)
        self.assertEqual(json.loads(data)['state']['status'], 'succeeded')

    def test_non_loopback_binding_is_rejected(self):
        with self.assertRaises(ValueError):
            MonitorServer(self.catalog, self.buffer, host='0.0.0.0', port=0)


class MonitorTaskRouteTests(unittest.TestCase):
    def setUp(self):
        from wmal.monitor.live_tasks import LiveTaskService
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / 'experiment'
        self.run.mkdir()
        (self.run / 'events.jsonl').write_text('', encoding='utf-8')
        (self.root / 'runs').mkdir()
        self.catalog = RunCatalog(self.root / 'runs')
        # Register a real catalog run while keeping task data outside its artifact tree.
        run = self.catalog.root / 'experiment'
        run.mkdir()
        (run / 'events.jsonl').write_text('', encoding='utf-8')
        self.run_id = self.catalog.list_runs()[0]['run_id']
        historic = self.catalog.root / 'historic-run'
        historic.mkdir()
        (historic / 'events.jsonl').write_text('', encoding='utf-8')
        self.historic_id = next(row['run_id'] for row in self.catalog.list_runs()
                                if row['run_path'] == 'historic-run')
        self.task_root = self.root / 'tasks'
        self.service = LiveTaskService(self.task_root, {
            'g1_navigation': lambda spec, context: {'status': 'succeeded', 'video': {'status': 'unavailable'}},
            'arm_joint_targets': lambda spec, context: {'status': 'succeeded', 'video': {'status': 'unavailable'}},
        })
        self.capabilities = {'available_modes': {'g1_navigation', 'arm_joint_targets'}, 'arm_max_horizon': 12}
        self.server = MonitorServer(self.catalog, port=0, task_service=self.service,
                                    capabilities=self.capabilities, task_run_id=self.run_id).start()

    def tearDown(self):
        self.server.shutdown()
        self.service.close()
        self.temp.cleanup()

    def request(self, path, method='GET', headers=None, body=None):
        connection = http.client.HTTPConnection(*self.server.address, timeout=3)
        request_headers = {'Host': f'127.0.0.1:{self.server.address[1]}'}
        request_headers.update(headers or {})
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def post_command(self, payload, key='cmd_1', **headers):
        all_headers = {'Origin': f'http://127.0.0.1:{self.server.address[1]}',
                       'Content-Type': 'application/json', 'Idempotency-Key': key}
        all_headers.update({name.replace('_', '-'): value for name, value in headers.items()})
        return self.request(f'/api/live/{self.run_id}/commands', 'POST', all_headers,
                            json.dumps(payload).encode('utf-8'))

    def test_health_and_task_routes_report_capabilities_and_lifecycle(self):
        health = json.loads(self.request('/api/health')[2])
        self.assertFalse(health['read_only'])
        listing = json.loads(self.request(f'/api/live/{self.run_id}/tasks')[2])
        self.assertEqual(set(listing['capabilities']['available_modes']),
                         {'g1_navigation', 'arm_joint_targets'})
        status, _, body = self.post_command({'mode': 'g1_navigation',
                                             'goal': {'x_m': .2, 'y_m': 0}, 'waypoints': []})
        self.assertEqual(status, 202)
        submitted = json.loads(body)
        self.assertEqual(submitted['command_id'], 'cmd_1')
        self.assertEqual(submitted['state'], 'queued')
        self.assertEqual(self.request(submitted['status_url'])[0], 200)
        rows = json.loads(self.request(f'/api/live/{self.run_id}/tasks')[2])['tasks']
        self.assertEqual(rows[0]['task_id'], submitted['task_id'])

    def test_command_requires_same_origin_host_and_idempotency_key(self):
        path = f'/api/live/{self.run_id}/commands'
        payload = json.dumps({'text': '前进 0.2 米'}).encode()
        base = {'Content-Type': 'application/json', 'Idempotency-Key': 'cmd_1'}
        self.assertEqual(self.request(path, 'POST', {**base, 'Origin': 'null'}, payload)[0], 403)
        self.assertEqual(self.request(path, 'POST', base, payload)[0], 403)
        self.assertEqual(self.request(path, 'POST', {**base, 'Origin': f'http://localhost:{self.server.address[1]}',
                            'Host': 'evil.test'}, payload)[0], 403)
        self.assertEqual(self.request(path, 'POST', {'Origin': f'http://localhost:{self.server.address[1]}',
                            'Content-Type': 'application/json'}, payload)[0], 400)

    def test_command_body_bounds_transfer_encoding_and_command_validation(self):
        self.assertEqual(self.post_command({'text': '向前随便走'} )[0], 422)
        self.assertEqual(self.post_command({'mode': 'g1_navigation', 'goal': {'x_m': 50, 'y_m': 0}})[0], 400)
        self.assertEqual(self.post_command({'text': '前进 0.2 米'}, Transfer_Encoding='chunked')[0], 400)
        headers = {'Origin': f'http://127.0.0.1:{self.server.address[1]}',
                   'Content-Type': 'application/json', 'Idempotency-Key': 'cmd_big',
                   'Content-Length': '16385'}
        self.assertEqual(self.request(f'/api/live/{self.run_id}/commands', 'POST', headers, b'{}')[0], 413)

    def test_idempotency_replay_and_conflict(self):
        payload = {'text': '前进 0.2 米'}
        first = json.loads(self.post_command(payload, key='same_key')[2])
        replay_status, _, replay_body = self.post_command(payload, key='same_key')
        self.assertEqual(replay_status, 202)
        self.assertEqual(json.loads(replay_body)['task_id'], first['task_id'])
        conflict, _, data = self.post_command({'text': '前进 0.3 米'}, key='same_key')
        self.assertEqual(conflict, 409)
        self.assertEqual(json.loads(data)['error']['code'], 'idempotency_conflict')

    def test_cancel_route_is_post_and_video_is_task_scoped(self):
        from wmal.monitor.commands import parse_command
        snap = self.service.submit(parse_command({'text': '前进 0.2 米'}), 'cancel_me', 'abc')
        origin = {'Origin': f'http://127.0.0.1:{self.server.address[1]}'}
        cancel_path = f'/api/live/{self.run_id}/tasks/{snap.task_id}/cancel'
        self.assertEqual(self.request(cancel_path, 'POST', origin, b'{}')[0], 200)
        self.assertEqual(self.request(f'/api/live/{self.run_id}/tasks/{snap.task_id}/video.mp4')[0], 404)
        self.assertEqual(self.request(f'/api/live/{self.run_id}/tasks/unknown/video.mp4')[0], 404)

    def test_video_is_only_served_for_ready_registered_regular_file(self):
        from wmal.monitor.commands import parse_command
        snap = self.service.submit(parse_command({'text': '前进 0.2 米'}), 'video_me', 'abc')
        video = self.task_root / snap.task_id / 'video.mp4'
        video.write_bytes(b'fake-mp4')
        # The service result is immutable from the route; use the supported runner path in a second service
        # result fixture by replacing the task record only for this endpoint contract test.
        self.service._tasks[snap.task_id]['result'] = {'status': 'succeeded', 'video': {'status': 'ready'}}
        self.service._tasks[snap.task_id]['state'] = 'succeeded'
        path = f'/api/live/{self.run_id}/tasks/{snap.task_id}/video.mp4'
        status, headers, body = self.request(path)
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'video/mp4')
        self.assertEqual(body, b'fake-mp4')
        video.unlink()
        outside = self.root / 'outside.mp4'
        outside.write_bytes(b'outside')
        video.symlink_to(outside)
        self.assertEqual(self.request(path)[0], 404)

    def test_live_task_service_is_scoped_to_its_attached_run(self):
        payload = {'text': '前进 0.2 米'}
        headers = {'Origin': f'http://127.0.0.1:{self.server.address[1]}',
                   'Content-Type': 'application/json', 'Idempotency-Key': 'history_command'}
        history_commands = f'/api/live/{self.historic_id}/commands'
        status, _, body = self.request(history_commands, 'POST', headers,
                                       json.dumps(payload).encode('utf-8'))
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(body)['error']['code'], 'wrong_live_run')
        task_list = f'/api/live/{self.historic_id}/tasks'
        status, _, body = self.request(task_list)
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(body)['error']['code'], 'wrong_live_run')
        self.assertEqual(len(self.service.list()), 0, 'rejected historical command must not reach the shared queue')

        live_list = json.loads(self.request(f'/api/live/{self.run_id}/tasks')[2])
        self.assertEqual(live_list['live_run_id'], self.run_id)
        from wmal.monitor.commands import parse_command
        task = self.service.submit(parse_command(payload), 'known_task', 'fp')
        detail_path = f'/api/live/{self.historic_id}/tasks/{task.task_id}'
        self.assertEqual(self.request(detail_path)[0], 409)
        cancel_path = f'/api/live/{self.historic_id}/tasks/{task.task_id}/cancel'
        self.assertEqual(self.request(cancel_path, 'POST', headers, b'{}')[0], 409)
        video_path = f'/api/live/{self.historic_id}/tasks/{task.task_id}/video.mp4'
        self.assertEqual(self.request(video_path)[0], 409)

    def test_shutdown_before_start_closes_socket_without_waiting_for_serve_forever(self):
        from wmal.monitor.server import MonitorServer
        unstarted = MonitorServer(self.catalog, port=0)
        unstarted.shutdown()


if __name__ == '__main__':
    unittest.main()
