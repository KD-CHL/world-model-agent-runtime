import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from wmal.monitor.commands import TaskSpec
from wmal.monitor.events import JsonlEventTail
from wmal.monitor.live_tasks import LiveTaskService, IdempotencyConflict


class LiveTaskServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'tasks'

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def spec(name='one'):
        return {'mode': 'fake', 'task': name, 'goal': [1.0, 2.0], 'waypoints': [],
                'options': {}, 'relative_intent': None}

    def wait_state(self, service, task_id, state, timeout=2):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            snap = service.get(task_id)
            if snap.state == state:
                return snap
            time.sleep(.005)
        self.fail(f'task did not reach {state}: {service.get(task_id)}')

    def test_runner_finishes_and_persists_json_safe_evidence(self):
        calls = []

        def runner(spec, context):
            self.assertIsInstance(spec, TaskSpec)
            calls.append(threading.current_thread().name)
            context.progress('predicting', .25)
            context.record_prediction({'pose': [1, 2]})
            context.record_observation({'pose': [1.1, 2]})
            return {'decision': 'goal_reached', 'count': 2}

        service = LiveTaskService(self.root, {'fake': runner})
        try:
            queued = service.submit(self.spec(), 'cmd-a', 'fingerprint-a')
            done = self.wait_state(service, queued.task_id, 'succeeded')
            self.assertEqual(done.result, {'decision': 'goal_reached', 'count': 2})
            self.assertTrue(calls)
            self.assertTrue(all(name == service.worker_name for name in calls))
            directory = self.root / queued.task_id
            manifest = json.loads((directory / 'manifest.json').read_text())
            event_lines = (directory / 'events.jsonl').read_text().splitlines()
            events = [json.loads(line) for line in event_lines]
            self.assertEqual(manifest['state'], 'succeeded')
            self.assertTrue(any(event['event'] == 'prediction' for event in events))
            self.assertTrue(any(event['event'] == 'observation' for event in events))
            tail = JsonlEventTail(directory / 'events.jsonl')
            tailed = tail.read_new()
            self.assertGreaterEqual(len(tailed), len(events))
            self.assertEqual(tail.status['error_count'], 0)
            prediction_row = next(row for row in tailed if row['event'] == 'prediction')
            raw_prediction = next(row for row in events if row['event'] == 'prediction')
            self.assertIsInstance(raw_prediction['timestamp'], float)
            self.assertEqual(prediction_row['raw']['timestamp'], raw_prediction['timestamp'])
            self.assertEqual(prediction_row['wall_time_utc'], raw_prediction['wall_time_utc'])
            self.assertTrue(prediction_row['wall_time_utc'].endswith('Z'))
            parsed_time = datetime.fromisoformat(prediction_row['wall_time_utc'].replace('Z', '+00:00'))
            self.assertEqual(parsed_time.tzinfo, timezone.utc)
            self.assertRegex(prediction_row['wall_time_utc'][11:23], r'^\d{2}:\d{2}:\d{2}\.\d{3}$',
                             'JsonlEventTail output must support the monitor timeline time slice')
        finally:
            service.close(2)

    def test_idempotency_returns_existing_or_conflicts_and_survives_restart(self):
        gate = threading.Event()
        def runner(spec, context):
            gate.wait(1)
            return {'ok': True}

        service = LiveTaskService(self.root, {'fake': runner})
        first = service.submit(self.spec(), 'cmd-a', 'fp-a')
        same = service.submit(self.spec(), 'cmd-a', 'fp-a')
        self.assertEqual(first.task_id, same.task_id)
        with self.assertRaises(IdempotencyConflict):
            service.submit(self.spec('other'), 'cmd-a', 'different')
        service.close(.01)
        gate.set()
        service.close(2)
        recovered = LiveTaskService(self.root, {'fake': lambda spec, context: {'ok': True}})
        try:
            self.assertEqual(recovered.submit(self.spec(), 'cmd-a', 'fp-a').task_id,
                             first.task_id)
        finally:
            recovered.close(2)

    def test_pending_queue_is_bounded_and_queued_cancel_is_immediate(self):
        started, release = threading.Event(), threading.Event()
        def runner(spec, context):
            started.set()
            release.wait(2)
            return {'ok': True}

        service = LiveTaskService(self.root, {'fake': runner}, max_pending=1)
        try:
            first = service.submit(self.spec('first'), 'a', 'a')
            self.assertTrue(started.wait(1))
            second = service.submit(self.spec('second'), 'b', 'b')
            with self.assertRaises(OverflowError):
                service.submit(self.spec('third'), 'c', 'c')
            cancelled = service.cancel(second.task_id)
            self.assertEqual(cancelled.state, 'cancelled')
            self.assertEqual(service.get(second.task_id).state, 'cancelled')
        finally:
            release.set()
            service.close(2)

    def test_running_cancel_is_visible_to_context_and_terminal_cancelled(self):
        started = threading.Event()
        def runner(spec, context):
            started.set()
            context.cancel_event.wait(2)
            return {'decision': 'stopped'}

        service = LiveTaskService(self.root, {'fake': runner})
        try:
            queued = service.submit(self.spec(), 'cmd', 'fp')
            self.assertTrue(started.wait(1))
            requested = service.cancel(queued.task_id)
            self.assertTrue(requested.cancel_requested)
            done = self.wait_state(service, queued.task_id, 'cancelled')
            self.assertEqual(done.result, {'decision': 'stopped'})
        finally:
            service.close(2)

    def test_fifo_and_single_runner_thread(self):
        order, active, max_active = [], 0, 0
        lock = threading.Lock()
        def runner(spec, context):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                order.append(spec.task)
            time.sleep(.02)
            with lock:
                active -= 1
            return {'ok': True}

        service = LiveTaskService(self.root, {'fake': runner})
        try:
            tasks = [service.submit(self.spec(str(i)), str(i), str(i)) for i in range(4)]
            for task in tasks:
                self.wait_state(service, task.task_id, 'succeeded')
            self.assertEqual(order, ['0', '1', '2', '3'])
            self.assertEqual(max_active, 1)
        finally:
            service.close(2)

    def test_exception_fails_and_recovery_marks_orphan_running_incomplete(self):
        service = LiveTaskService(self.root, {'fake': lambda spec, context: 1 / 0})
        try:
            task = service.submit(self.spec(), 'cmd', 'fp')
            failed = self.wait_state(service, task.task_id, 'failed')
            self.assertIn('ZeroDivisionError', failed.error['type'])
        finally:
            service.close(2)

        task_dir = self.root / task.task_id
        manifest = json.loads((task_dir / 'manifest.json').read_text())
        manifest['state'] = 'running'
        (task_dir / 'manifest.json').write_text(json.dumps(manifest))
        recovered = LiveTaskService(self.root, {'fake': lambda spec, context: {'ok': True}})
        try:
            incomplete = recovered.get(task.task_id)
            self.assertEqual(incomplete.state, 'incomplete')
            self.assertEqual(json.loads((task_dir / 'manifest.json').read_text())['state'], 'incomplete')
        finally:
            recovered.close(2)

    def test_close_timeout_does_not_claim_running_runner_stopped(self):
        started, release = threading.Event(), threading.Event()
        def runner(spec, context):
            started.set()
            release.wait(2)
            return {'ok': True}

        service = LiveTaskService(self.root, {'fake': runner})
        task = service.submit(self.spec(), 'cmd', 'fp')
        self.assertTrue(started.wait(1))
        closed = service.close(.01)
        self.assertFalse(closed)
        self.assertEqual(service.get(task.task_id).state, 'running')
        release.set()
        self.assertTrue(service.close(2))
        self.assertEqual(service.get(task.task_id).state, 'cancelled')

    def test_runner_status_maps_terminal_state_and_preserves_failure_evidence(self):
        results = {
            'success': {'status': 'succeeded', 'observations': [{'x': 1}]},
            'failure': {'status': 'failed', 'reason': 'unsafe target',
                        'video': {'status': 'ready'}},
            'cancel': {'status': 'cancelled', 'observations': [{'x': .5}]},
            'default': {'observations': [{'x': 2}]},
        }
        service = LiveTaskService(self.root, {'fake': lambda spec, context: results[spec.task]})
        try:
            snapshots = {name: service.submit(self.spec(name), name, name)
                         for name in results}
            expected = {'success': 'succeeded', 'failure': 'failed',
                        'cancel': 'cancelled', 'default': 'succeeded'}
            for name, task in snapshots.items():
                self.assertEqual(self.wait_state(service, task.task_id, expected[name]).state,
                                 expected[name])
            failed = service.get(snapshots['failure'].task_id)
            self.assertEqual(failed.result, results['failure'])
            self.assertIn('unsafe target', failed.error['message'])
            self.assertEqual(service.get(snapshots['cancel'].task_id).result,
                             results['cancel'])
        finally:
            service.close(2)

    def test_runner_close_hooks_execute_once_on_owner_after_task_finishes(self):
        calls = []
        class Runner:
            def __call__(self, spec, context):
                calls.append(('run', threading.current_thread().name))
                return {'ok': True}

            def close(self):
                calls.append(('close', threading.current_thread().name))

        service = LiveTaskService(self.root, {'fake': Runner()})
        task = service.submit(self.spec(), 'cmd', 'fp')
        self.wait_state(service, task.task_id, 'succeeded')
        self.assertTrue(service.close(2))
        self.assertEqual([kind for kind, _ in calls], ['run', 'close'])
        self.assertEqual({thread for _, thread in calls}, {service.worker_name})

    def test_slow_close_hook_does_not_hold_service_lock_or_block_repeated_close(self):
        hook_started, release_hook = threading.Event(), threading.Event()
        class SlowRunner:
            def __call__(self, spec, context):
                return {'ok': True}

            def close(self):
                hook_started.set()
                release_hook.wait(2)

        service = LiveTaskService(self.root, {'fake': SlowRunner()})
        task = service.submit(self.spec(), 'cmd', 'fp')
        self.wait_state(service, task.task_id, 'succeeded')
        self.assertFalse(service.close(.01))
        self.assertTrue(hook_started.wait(1))
        start = time.monotonic()
        self.assertEqual(service.get(task.task_id).state, 'succeeded')
        self.assertFalse(service.close(.01))
        self.assertLess(time.monotonic() - start, .2)
        release_hook.set()
        self.assertTrue(service.close(2))

    def test_task_id_traversal_and_symlink_task_directories_are_ignored(self):
        service = LiveTaskService(self.root, {'fake': lambda spec, context: {'ok': True}})
        try:
            with self.assertRaises(KeyError):
                service.get('../manifest.json')
        finally:
            service.close(2)
        outside = Path(self.temp.name) / 'outside'
        outside.mkdir()
        (outside / 'manifest.json').write_text('{}')
        self.root.mkdir(exist_ok=True)
        valid_task_id = 'a' * 32
        (self.root / valid_task_id).symlink_to(outside, target_is_directory=True)
        recovered = LiveTaskService(self.root, {'fake': lambda spec, context: {'ok': True}})
        try:
            self.assertEqual(recovered.list(), [])
            with self.assertRaises(KeyError):
                recovered.get(valid_task_id)
        finally:
            recovered.close(2)

    def test_semantically_malformed_recovery_evidence_does_not_abort_startup(self):
        self.root.mkdir(exist_ok=True)
        bad_spec_id, bad_events_id = 'b' * 32, 'c' * 32
        for task_id, spec in ((bad_spec_id, 'not-an-object'),
                              (bad_events_id, {'mode': 'fake'})):
            task_dir = self.root / task_id
            task_dir.mkdir()
            manifest = {'task_id': task_id, 'command_id': task_id,
                        'payload_fingerprint': 'fp', 'state': 'queued',
                        'spec': spec, 'created_at': 1.0, 'updated_at': 1.0,
                        'progress': None, 'result': None, 'error': None,
                        'cancel_requested': False}
            (task_dir / 'manifest.json').write_text(json.dumps(manifest))
            (task_dir / 'events.jsonl').write_text('{"type":"queued"}\nnot-json\n')

        service = LiveTaskService(self.root, {'fake': lambda spec, context: {'ok': True}})
        try:
            self.assertEqual(service.list(), [])
            self.assertTrue((self.root / bad_spec_id / 'manifest.json').exists())
            self.assertTrue((self.root / bad_events_id / 'events.jsonl').exists())
        finally:
            service.close(2)

    def test_recovered_queued_task_without_runner_becomes_incomplete_and_keeps_idempotency(self):
        task_id = 'd' * 32
        task_dir = self.root / task_id
        task_dir.mkdir(parents=True)
        manifest = {'task_id': task_id, 'command_id': 'orphan-command',
                    'payload_fingerprint': 'fp-orphan', 'state': 'queued',
                    'spec': {**self.spec(), 'mode': 'orphan'},
                    'created_at': 10.0, 'updated_at': 10.0, 'progress': None,
                    'result': None, 'error': None, 'cancel_requested': False}
        (task_dir / 'manifest.json').write_text(json.dumps(manifest))
        (task_dir / 'events.jsonl').write_text(json.dumps(
            {'event': 'queued', 'timestamp': 10.0, 'payload': {'mode': 'orphan'}}) + '\n')

        service = LiveTaskService(self.root, {})
        try:
            task = service.get(task_id)
            self.assertEqual(task.state, 'incomplete')
            self.assertEqual(task.error['type'], 'MissingRunner')
            retry = service.submit(manifest['spec'], 'orphan-command', 'fp-orphan')
            self.assertEqual(retry.task_id, task_id)
            self.assertEqual(retry.state, 'incomplete')
        finally:
            service.close(2)

    def test_recovery_rejects_nonfinite_and_malformed_snapshot_fields(self):
        self.root.mkdir(exist_ok=True)
        cases = (
            ('e' * 32, {'created_at': float('nan')}),
            ('f' * 32, {'updated_at': float('inf')}),
            ('1' * 32, {'cancel_requested': 0}),
            ('2' * 32, {'spec': {**self.spec(), 'goal': [float('nan'), 0]}}),
        )
        for index, (task_id, changes) in enumerate(cases):
            task_dir = self.root / task_id
            task_dir.mkdir()
            manifest = {'task_id': task_id, 'command_id': f'bad-{index}',
                        'payload_fingerprint': 'fp', 'state': 'queued',
                        'spec': self.spec(), 'created_at': 10.0, 'updated_at': 10.0,
                        'progress': None, 'result': None, 'error': None,
                        'cancel_requested': False}
            manifest.update(changes)
            (task_dir / 'manifest.json').write_text(json.dumps(manifest, allow_nan=True))
            (task_dir / 'events.jsonl').write_text(json.dumps(
                {'event': 'queued', 'timestamp': 10.0, 'payload': {}}) + '\n')
        bad_event_id = '3' * 32
        bad_event_dir = self.root / bad_event_id
        bad_event_dir.mkdir()
        valid_manifest = {'task_id': bad_event_id, 'command_id': 'bad-event',
                          'payload_fingerprint': 'fp', 'state': 'queued',
                          'spec': self.spec(), 'created_at': 10.0, 'updated_at': 10.0,
                          'progress': None, 'result': None, 'error': None,
                          'cancel_requested': False}
        (bad_event_dir / 'manifest.json').write_text(json.dumps(valid_manifest))
        (bad_event_dir / 'events.jsonl').write_text(
            '{"event":"queued","timestamp":Infinity,"payload":{}}\n')

        service = LiveTaskService(self.root, {'fake': lambda spec, context: {'ok': True}})
        try:
            self.assertEqual(service.list(), [])
            for task_id, _ in cases:
                self.assertTrue((self.root / task_id / 'manifest.json').exists())
        finally:
            service.close(2)


if __name__ == '__main__':
    unittest.main()
