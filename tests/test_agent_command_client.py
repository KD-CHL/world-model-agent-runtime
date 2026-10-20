"""CLI preflight and lifecycle checks without physics or model weights."""
import importlib.util
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, Mock

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/agent_command_client.py'
spec = importlib.util.spec_from_file_location('agent_command_client', SCRIPT)
client = importlib.util.module_from_spec(spec)
# Script remains importable without importing simulation/model dependencies.
if SCRIPT.exists():
    spec.loader.exec_module(client)


class ClientTests(unittest.TestCase):
    def test_parser_cpu_and_bounded_options(self):
        args = client.build_parser().parse_args([])
        self.assertEqual(args.device, 'cpu')
        self.assertTrue(args.realtime)
        for argv in (['--port','65536'], ['--g1-cycles','0'], ['--max-pending','0']):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                client.build_parser().parse_args(argv)

    def test_missing_assets_keep_capability_snapshot(self):
        args = client.build_parser().parse_args(['--g1-config','/missing/config.json'])
        assets = client.preflight(args)
        self.assertEqual(assets['capabilities']['available_modes'], [])
        self.assertIn('g1_navigation', assets['capabilities']['unavailable_reasons'])
        self.assertIn('arm_joint_targets', assets['capabilities']['unavailable_reasons'])

    def test_checkpoint_relative_to_config_repository(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root/'pyproject.toml').touch()
            (root/'configs').mkdir()
            (root/'runs').mkdir()
            (root/'runs/model.pt').touch()
            path = root/'configs/model.json'
            path.write_text(json.dumps({'world_model':{'factory':'local:load','config':{'checkpoint':'runs/model.pt'}}, 'planner':{}}))
            model = SimpleNamespace(version='test')
            with patch.object(client, '_check_simulation_assets'), patch.object(client, 'load_world_model', return_value=model) as load, patch.object(client, 'G1RolloutPlanner', return_value='planner'):
                args = client.build_parser().parse_args(['--g1-config',str(path)])
                assets = client.preflight(args)
            self.assertEqual(load.call_args.args[1]['checkpoint'], str(root/'runs/model.pt'))
            self.assertEqual(assets['capabilities']['available_modes'], ['g1_navigation'])

    def test_arm_rejects_incompatible_calibration_and_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights.pt'; path.touch()
            calibration=Path(folder)/'cal.json'; calibration.touch()
            model=SimpleNamespace(version='test', metadata={'training_source':{'task':'stack_block'},'max_horizon':4}, config=SimpleNamespace(image_size=64), semantics={'camera':'wrong'})
            with patch.object(client, '_load_visual', return_value=model), patch.object(client, '_load_calibration', return_value=Mock()):
                args=client.build_parser().parse_args(['--visual-checkpoint',str(path),'--visual-calibration',str(calibration)])
                assets=client.preflight(args)
            self.assertNotIn('arm_joint_targets',assets['capabilities']['available_modes'])
            self.assertIn('semantics',assets['capabilities']['unavailable_reasons']['arm_joint_targets'])

    def test_arm_requires_matching_version_and_exposes_calibrated_horizon(self):
        from wmal.envs.visual_workcell import JOINTS, ACTION_SCHEMA
        semantics={'action_schema':ACTION_SCHEMA,'action_order':list(JOINTS),
                   'state_order':list(JOINTS)+[joint+'_target' for joint in JOINTS],
                   'camera':'workcell_overview','period_s':.2,'image_size':64,'event_names':[]}
        model=SimpleNamespace(version='test', metadata={'training_source':{'task':'stack_block'},'max_horizon':8},
                              config=SimpleNamespace(image_size=64), semantics=semantics)
        calibration=SimpleNamespace(quantiles=(1,2,3,4),state_scale=(1,1,1,1),validate_for=Mock())
        self.assertEqual(client._validate_visual(model,calibration),4)
        calibration.validate_for.assert_called_once_with('test',semantics)
        calibration.validate_for.side_effect=ValueError('mismatch')
        with self.assertRaisesRegex(ValueError,'mismatch'):
            client._validate_visual(model,calibration)

    def test_missing_models_start_usable_loopback_service(self):
        import urllib.request
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()/'runs'
            args=client.build_parser().parse_args(['--g1-config','/missing/config.json',
                '--runs-root',str(root),'--run-dir','demo','--port','0'])
            runtime=client.start_client(args)
            with runtime:
                with urllib.request.urlopen(runtime.url,timeout=2) as response:
                    self.assertEqual(response.status,200)
                self.assertEqual(runtime.server.capabilities['available_modes'],[])
                self.assertEqual(runtime.task_service.task_root,runtime.task_service.task_root.parent/'.agent_tasks')

    def test_run_paths_and_hidden_task_discovery(self):
        from wmal.monitor.catalog import RunCatalog
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()/'runs'
            run=client.prepare_run(root,'demo',{'available_modes':[]})
            (run/'.agent_tasks'/'x').mkdir(parents=True)
            (run/'.agent_tasks'/'x'/'events.jsonl').touch()
            self.assertEqual(len(RunCatalog(root).list_runs()),1)
            for outside in (Path(folder)/'outside',root):
                with self.assertRaises(ValueError):
                    client.prepare_run(root,outside,{})
            (root/'linked').symlink_to(run,target_is_directory=True)
            with self.assertRaises(ValueError):
                client.prepare_run(root,'linked',{})
            with self.assertRaises(ValueError):
                client.prepare_run(root,'demo',{})

    def test_runtime_owner_closes_before_publisher_and_server(self):
        from wmal.monitor.runtime import MonitorRuntime
        order=[]
        runtime=MonitorRuntime.__new__(MonitorRuntime)
        runtime.run_id='current-run'
        runtime.publisher=SimpleNamespace(close=lambda:order.append('publisher'))
        runtime.server=SimpleNamespace(shutdown=lambda:order.append('server'),task_service=None,capabilities=None)
        service=SimpleNamespace(close=lambda timeout_s: order.append('owner') or True)
        runtime.attach_task_service(service, {'available_modes':[]})
        runtime.__exit__()
        self.assertEqual(order,['owner','publisher','server'])
        self.assertEqual(runtime.server.task_run_id,'current-run')


if __name__ == '__main__':
    unittest.main()
