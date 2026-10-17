"""Bootstrap ridge dynamics for policy-conditioned G1 base motion.

This lightweight learned baseline implements the same plugin contract as a
larger pretrained world model. It does not predict pixels or joint trajectories.
"""
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from wmal.logging.manifest import atomic_json
from wmal.locomotion.contracts import G1_ACTION_SCHEMA, G1_STATE_SCHEMA, G1Prediction


def wrap(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def features(state, action):
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    return np.array([1, c * state.vx + s * state.vy,
                     -s * state.vx + c * state.vy, state.yaw_rate,
                     state.roll, state.pitch, state.pelvis_height,
                     action.vx, action.vy, action.yaw_rate], dtype=float)


class LearnedG1Dynamics:
    state_schema, action_schema = G1_STATE_SCHEMA, G1_ACTION_SCHEMA

    def __init__(self, weights, duration_s, version):
        self.weights = np.asarray(weights, dtype=float)
        if (self.weights.ndim != 3 or self.weights.shape[1:] != (10, 6)
                or len(self.weights) < 2 or not np.isfinite(self.weights).all()):
            raise ValueError('Expected finite ensemble weights [members, 10, 6]')
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('Invalid trained action duration')
        self.duration_s, self.version = duration_s, version

    def predict(self, state, action, duration_s):
        if abs(duration_s - self.duration_s) > 1e-9:
            raise ValueError('Action duration differs from training duration')
        outputs = features(state, action) @ self.weights
        mean, variance = outputs.mean(axis=0), outputs.var(axis=0)
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        vx, vy = c * mean[0] - s * mean[1], s * mean[0] + c * mean[1]
        predicted = replace(state, step_id=state.step_id + 1,
                            sim_time_s=state.sim_time_s + duration_s,
                            x=float(state.x + vx * duration_s),
                            y=float(state.y + vy * duration_s),
                            yaw=wrap(state.yaw + float(mean[2]) * duration_s),
                            vx=float(vx), vy=float(vy), yaw_rate=float(mean[2]),
                            roll=float(mean[3]), pitch=float(mean[4]),
                            z=max(0., float(mean[5])), pelvis_height=max(0., float(mean[5])))
        return G1Prediction(predicted, {'position_variance': float(sum(variance[:2]) * duration_s**2),
                                        'posture_variance': float(sum(variance[3:]))},
                            uncertainty_kind='ensemble_spread')

    def save(self, path):
        atomic_json(path, {'schema': 'wmal.g1.ridge.v1', 'weights': self.weights.tolist(),
                           'duration_s': self.duration_s, 'version': self.version})


def load(config):
    payload = json.loads(Path(config['checkpoint']).read_text())
    if payload['schema'] != 'wmal.g1.ridge.v1':
        raise ValueError('Unknown G1 checkpoint schema')
    return LearnedG1Dynamics(payload['weights'], payload['duration_s'], payload['version'])
