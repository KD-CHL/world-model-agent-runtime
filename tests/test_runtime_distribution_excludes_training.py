"""Runtime wheel ownership and frozen model/provider compatibility."""
import configparser
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


class RuntimeDistributionTests(unittest.TestCase):
    def test_wheel_keeps_inference_and_monitor_without_training_commands(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            shutil.copy(root / 'pyproject.toml', build)
            shutil.copytree(root / 'src', build / 'src',
                            ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
            result = subprocess.run(
                [sys.executable, '-c',
                 "from setuptools.build_meta import build_wheel; build_wheel('dist')"],
                cwd=build, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with zipfile.ZipFile(next((build / 'dist').glob('*.whl'))) as wheel:
                names = wheel.namelist()
                self.assertTrue(any(name.startswith('wmal/models/') for name in names))
                self.assertTrue(any(name.startswith('wmal/monitor/') for name in names))
                self.assertFalse(any(name.startswith('wmal/training/') for name in names))
                entries = configparser.ConfigParser()
                entries.read_string(wheel.read(next(name for name in names
                                                    if name.endswith('/entry_points.txt'))).decode())
                self.assertEqual(set(entries['console_scripts']),
                                 {'wmal-agent', 'wmal-serve', 'wmal-monitor'})

    def test_joint_inference_roundtrip_has_no_fitting_api(self):
        import numpy as np
        from wmal.models.latent_dynamics import JointDynamics
        self.assertFalse(hasattr(JointDynamics, 'fit'))
        weights = np.zeros((2, 4, 1))
        weights[:, 3, 0] = .5
        model = JointDynamics(('joint',), weights, 'frozen-fixture')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            model.save(path)
            restored = JointDynamics.load(path)
        self.assertEqual(restored.version, model.version)
        self.assertEqual(restored.predict({'joint': 0.}, {'joint': .2}, .5), {'joint': .1})

    def test_g1_factory_loads_frozen_model_without_fitting_api(self):
        import numpy as np
        from wmal.locomotion.contracts import G1State, G1VelocityAction
        from wmal.locomotion.learned import LearnedG1Dynamics
        from wmal.locomotion.world_model import load_world_model
        self.assertFalse(hasattr(LearnedG1Dynamics, 'fit'))
        weights = np.zeros((2, 10, 6))
        weights[:, 7, 0] = .8
        weights[:, 0, 5] = .8
        model = LearnedG1Dynamics(weights, .5, 'frozen-g1')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'g1.json'
            model.save(path)
            provider = load_world_model('wmal.locomotion.learned:load', {'checkpoint': path})
        state = G1State('fixture', 0, 0., 0., 0., .8, 0., 0., 0., 0., 0., 0., 0., .8)
        prediction = provider.predict(state, G1VelocityAction(.2, 0., 0., .5), .5)
        self.assertEqual(provider.version, model.version)
        self.assertAlmostEqual(prediction.state.x, .08)


    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Install model-runtime extra')
    def test_neural_model_keeps_construction_prediction_and_checkpoint_loading(self):
        import torch
        from wmal.models.neural_dynamics import NeuralJointDynamics, _network
        self.assertFalse(hasattr(NeuralJointDynamics, 'fit'))
        network = _network(3, 1)
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network[-1].bias.fill_(.125)
        model = NeuralJointDynamics(('joint',), [network.state_dict()], 'frozen-neural')
        expected = model.predict({'joint': .2}, {'joint': .3}, .5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'neural.pt'
            model.save(path)
            restored = NeuralJointDynamics.load(path)
        self.assertEqual(restored.version, model.version)
        self.assertEqual(restored.predict({'joint': .2}, {'joint': .3}, .5), expected)
        self.assertAlmostEqual(expected['joint'], .325)
