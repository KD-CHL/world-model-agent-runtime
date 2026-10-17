"""Learning, replay provenance, resume and planner compatibility checks."""
from dataclasses import asdict, replace
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wmal.datasets.motion_sequences import load_motion_episodes, contiguous_sequences
from wmal.locomotion.contracts import G1State, G1VelocityAction, G1Goal

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


def fixture(path):
    rng, rows = np.random.default_rng(4), []
    for episode in range(8):
        split = 'train' if episode < 4 else ('validation' if episode < 6 else 'test')
        state = G1State(f'fixture-{episode}',0,0.,0.,0.,.8,0.,0.,0.,0.,0.,0.,0.,.8)
        for step in range(12):
            action = G1VelocityAction(float(rng.uniform(-.2,.3)),float(rng.uniform(-.1,.1)),
                                     float(rng.uniform(-.2,.2)),.5)
            c,s = math.cos(state.yaw),math.sin(state.yaw)
            bvx,bvy = .8*action.vx,.8*action.vy
            vx,vy = c*bvx-s*bvy,s*bvx+c*bvy
            after = replace(state,step_id=step+1,sim_time_s=state.sim_time_s+.5,
                x=state.x+vx*.5,y=state.y+vy*.5,yaw=state.yaw+action.yaw_rate*.5,
                vx=vx,vy=vy,yaw_rate=action.yaw_rate)
            rows.append({'split':split,'before':asdict(state),'action':asdict(action),'after':asdict(after)})
            state = after
    payload = {'schema':'wmal.g1.transitions.v1','rows':rows,'data_source':'synthetic_test_fixture'}
    path.write_text(json.dumps(payload))
    return payload


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'data.json'
        self.payload = fixture(self.path)

    def test_leakage_duplicates_and_discontinuous_states_are_rejected(self):
        for change in ('split','duplicate','state'):
            payload = json.loads(json.dumps(self.payload))
            if change == 'split': payload['rows'][1]['split'] = 'test'
            if change == 'duplicate': payload['rows'].append(payload['rows'][0])
            if change == 'state': payload['rows'][1]['before']['x'] += .1
            self.path.write_text(json.dumps(payload))
            with self.subTest(change=change), self.assertRaises(ValueError): load_motion_episodes(self.path)

    def test_sequences_do_not_cross_missing_transitions(self):
        self.payload['rows'].pop(5)
        self.path.write_text(json.dumps(self.payload))
        splits, duration = load_motion_episodes(self.path)
        windows = contiguous_sequences(splits['train'],3)
        self.assertEqual(duration,.5)
        self.assertEqual(len(windows['fixture-0']),7)
        self.assertTrue(all(all(a[2]==b[0] for a,b in zip(w,w[1:])) for w in windows['fixture-0']))


if __name__ == '__main__': unittest.main()
