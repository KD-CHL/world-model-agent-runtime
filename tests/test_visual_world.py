"""Behavioral regressions for action-conditioned visual prediction and trust."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def make_dataset(root):
    from wmal.datasets.visual_sequences import write_episode, write_manifest
    rng = np.random.default_rng(7)
    rows = []
    for split, count in [('train', 4), ('validation', 2), ('calibration', 5), ('test', 2)]:
        for i in range(count):
            actions = rng.uniform(-.15, .15, (8, 2)).astype('float32')
            states = np.concatenate([np.zeros((1, 2)), np.cumsum(actions, axis=0)]).astype('float32')
            rgb = np.zeros((9, 32, 32, 3), dtype='uint8')
            for t, state in enumerate(states):
                x = int(np.clip(16 + state[0] * 20, 2, 28))
                rgb[t, 10:18, x:x+3, 0] = 255
            episode = f'{split}_{i}'
            path = root / f'{episode}.npz'
            write_episode(path, rgb, states, actions, overwrite=True)
            rows.append({'episode_id': episode, 'split': split, 'path': path.name})
    return write_manifest(root/'manifest.json', rows, action_schema='test.delta.v1',
                          action_order=['x', 'y'], state_order=['x', 'y'],
                          camera='front', period_s=.2, image_size=32, source={'kind': 'controlled_test'})


class VisualDataTests(unittest.TestCase):
    def test_visual_dataset_capability_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('wmal.datasets.visual_sequences'))

    def test_windows_never_cross_episode_and_stats_ignore_test(self):
        from wmal.datasets.visual_sequences import VisualDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = VisualDataset(make_dataset(root), 'train', horizon=3)
            self.assertEqual(len(dataset), 24)
            sample = dataset[0]
            self.assertEqual(sample['rgb'].shape, (4, 3, 32, 32))
            self.assertEqual(sample['actions'].shape, (3, 2))
            norm = dataset.normalization()
            np.savez(root/'test_0.npz', rgb=np.zeros((9,32,32,3), dtype='uint8'),
                     states=np.ones((9,2))*1000, actions=np.ones((8,2))*1000)
            np.testing.assert_array_equal(dataset.normalization()['state_mean'], norm['state_mean'])

    def test_normalization_does_not_decompress_rgb_frames(self):
        from unittest.mock import patch
        from wmal.datasets.visual_sequences import VisualDataset
        with tempfile.TemporaryDirectory() as tmp:
            dataset = VisualDataset(make_dataset(Path(tmp)), 'train', horizon=3)
            original = np.lib.npyio.NpzFile.__getitem__

            def reject_rgb(archive, key):
                if key == 'rgb':
                    raise AssertionError('normalization decompressed RGB frames')
                return original(archive, key)

            with patch.object(np.lib.npyio.NpzFile, '__getitem__', reject_rgb):
                stats = dataset.normalization()
            self.assertEqual(set(stats), {'state_mean', 'state_scale', 'action_mean', 'action_scale'})

    def test_event_supervision_profile_reads_only_event_mask(self):
        from unittest.mock import patch
        from wmal.datasets.visual_sequences import VisualDataset, write_episode, write_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_dataset(root)
            payload = json.loads(manifest.read_text())
            for row in payload['episodes']:
                path = root / row['path']
                with np.load(path, allow_pickle=False) as arrays:
                    rgb, states, actions = (arrays['rgb'].copy(), arrays['states'].copy(),
                                            arrays['actions'].copy())
                events = np.zeros((8, 1), dtype=np.float32)
                event_mask = np.ones_like(events)
                write_episode(path, rgb, states, actions, events=events,
                              event_mask=event_mask, overwrite=True)
            semantics = dict(payload['semantics'])
            semantics.pop('event_names', None)
            write_manifest(manifest, payload['episodes'], source=payload['source'],
                           event_names=['seen'], **semantics)
            dataset = VisualDataset(manifest, 'train', horizon=3)
            original = np.lib.npyio.NpzFile.__getitem__

            def reject_rgb(archive, key):
                if key == 'rgb':
                    raise AssertionError('event supervision scan decompressed RGB frames')
                return original(archive, key)

            with patch.object(np.lib.npyio.NpzFile, '__getitem__', reject_rgb):
                mask = dataset.event_supervision_for_episode(0)
            np.testing.assert_array_equal(mask, [True])

    def test_event_context_is_aligned_to_prediction_origin_and_missing_reset_is_masked(self):
        from wmal.datasets.visual_sequences import VisualDataset, write_episode, write_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_dataset(root)
            payload = json.loads(manifest.read_text())
            rows = payload['episodes']
            for row in rows:
                path = root / row['path']
                with np.load(path, allow_pickle=False) as arrays:
                    rgb, states, actions = arrays['rgb'].copy(), arrays['states'].copy(), arrays['actions'].copy()
                events = np.zeros((8, 1), dtype=np.float32)
                events[0, 0] = 1.0
                write_episode(path, rgb, states, actions, events=events,
                              event_mask=np.ones_like(events), overwrite=True)
            semantics = dict(payload['semantics'])
            semantics.pop('event_names', None)
            write_manifest(manifest, rows, source=payload['source'],
                           event_names=['grasped'], **semantics)
            dataset = VisualDataset(manifest, 'validation', horizon=2)
            reset_window, after_transition = dataset[0], dataset[1]
            np.testing.assert_array_equal(reset_window['event_context'], [0.])
            np.testing.assert_array_equal(reset_window['event_context_mask'], [0.])
            np.testing.assert_array_equal(after_transition['event_context'], [1.])
            np.testing.assert_array_equal(after_transition['event_context_mask'], [1.])
            np.testing.assert_array_equal(after_transition['events'][0], [0.])

    def test_event_state_archive_rejects_future_label_misalignment(self):
        from wmal.datasets.visual_sequences import write_episode
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = np.zeros((3, 32, 32, 3), dtype=np.uint8)
            states = np.zeros((3, 1), dtype=np.float32)
            actions = np.zeros((2, 1), dtype=np.float32)
            events = np.asarray([[1.], [0.]], dtype=np.float32)
            event_states = np.zeros((3, 1), dtype=np.float32)
            with self.assertRaisesRegex(ValueError, 'transition labels'):
                write_episode(root/'misaligned.npz', rgb, states, actions,
                              events=events, event_mask=np.ones_like(events),
                              extra_arrays={'event_states': event_states,
                                            'event_state_mask': np.ones_like(event_states)})


    def test_duplicate_episode_or_content_and_wrong_alignment_are_rejected(self):
        from wmal.datasets.visual_sequences import VisualDataset, write_episode
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_dataset(root)
            payload = json.loads(manifest.read_text())
            payload['episodes'][-1]['episode_id'] = payload['episodes'][0]['episode_id']
            manifest.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                VisualDataset(manifest, 'train', horizon=3)
            manifest = make_dataset(root)
            payload = json.loads(manifest.read_text())
            payload['episodes'][-1]['sha256'] = payload['episodes'][0]['sha256']
            manifest.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                VisualDataset(manifest, 'train', horizon=3)
            with self.assertRaises(ValueError):
                write_episode(root/'bad.npz', np.zeros((3,32,32,3), dtype='uint8'),
                              np.zeros((3,2)), np.zeros((3,2)))


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional learning dependencies not installed')
class VisualNetworkTests(unittest.TestCase):


    def test_zero_residual_preserves_background_instead_of_relearning_static_world(self):
        import torch
        from wmal.models.visual_latent import NetworkConfig, VisualLatentMember
        model=VisualLatentMember(NetworkConfig(2,2,image_size=32,latent_dim=16,hidden_dim=24))
        for parameter in model.parameters():
            parameter.data.zero_()
        rgb=torch.rand(1,3,32,32)
        frames=model.imagine(rgb,torch.zeros(1,2),torch.zeros(1,2,2))['rgb']
        torch.testing.assert_close(frames[:,0],rgb)
        torch.testing.assert_close(frames[:,1],rgb)

    def test_imagination_depends_on_actions_without_future_observations(self):
        import torch
        from wmal.models.visual_latent import NetworkConfig, VisualLatentMember
        torch.manual_seed(4)
        torch.set_num_threads(1)
        model = VisualLatentMember(NetworkConfig(2, 2, image_size=32, latent_dim=16, hidden_dim=24))
        initial_rgb = torch.zeros(2,3,32,32)
        state = torch.zeros(2,2)
        actions = torch.stack([torch.zeros(3,2), torch.ones(3,2)*.1])
        prediction = model.imagine(initial_rgb, state, actions)
        self.assertEqual(tuple(prediction['rgb'].shape), (2,3,3,32,32))
        self.assertGreater(float((prediction['state'][0]-prediction['state'][1]).abs().max().detach()), 1e-5)
        self.assertGreater(float((prediction['rgb'][0]-prediction['rgb'][1]).abs().max().detach()), 1e-6)
        (prediction['rgb'].mean()+prediction['state'].mean()).backward()
        self.assertGreater(float(model.transition.weight_ih.grad.abs().sum()), 0.)


class HorizonCalibrationTests(unittest.TestCase):
    def test_episode_quantile_not_frame_quantile_and_model_binding(self):
        from wmal.models.horizon_calibration import fit_horizon_calibration
        rows=[]
        for i,error in enumerate([1.,2.,3.,4.,5.]):
            for _ in range(20 if i==0 else 1):
                rows.append({'episode_id':str(i),'split':'calibration',
                             'scores':[error,error*2]})
        calibration=fit_horizon_calibration(rows, model_version='v1', dataset_id='d',
                                             alpha=.2, state_scale=[1.,1.], std_floor=.1,
                                             semantics={'action_schema':'test.delta.v1'})
        self.assertEqual(calibration.quantiles,(5.,10.))
        np.testing.assert_allclose(calibration.bounds(np.zeros((2,2)),version='v1'),
                                   [[.5,.5],[1.,1.]])
        with self.assertRaises(ValueError):
            calibration.bounds(np.zeros((2,2)),version='v2')
        rows[0]['split']='test'
        with self.assertRaises(ValueError):
            fit_horizon_calibration(rows,model_version='v1',dataset_id='d',alpha=.2,
                                    state_scale=[1.,1.],std_floor=.1,semantics={})


if __name__=='__main__':
    unittest.main()
