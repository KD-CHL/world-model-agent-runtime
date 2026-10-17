"""Optional neural ensemble for nonlinear state/action transition prediction.

PyTorch is imported lazily so simulation protocols stay usable without it.
The model predicts state deltas and exposes member-wise rollouts to the planner.
"""
from pathlib import Path
import os
import tempfile

import numpy as np


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError('Install the optional model-runtime dependencies to use neural dynamics') from exc
    return torch, nn


def _network(width, output_width):
    torch, nn = _torch_modules()
    return nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 128),
                         nn.SiLU(), nn.Linear(128, output_width))


class NeuralJointDynamics:
    """Bootstrap ensemble with per-member MLP residual dynamics."""

    def __init__(self, joints, state_dicts, version, *, device='cpu'):
        torch, _ = _torch_modules()
        self.joints = tuple(joints)
        if not self.joints or len(set(self.joints)) != len(self.joints) or not state_dicts:
            raise ValueError('Invalid model dimensions')
        self.device = torch.device(device)
        self.models = []
        for state_dict in state_dicts:
            model = _network(1 + 2 * len(self.joints), len(self.joints)).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)
        self.ensemble_size = len(self.models)
        self.version = str(version)

    def _feature(self, state, targets, duration_s):
        if set(state) != set(self.joints) or not set(targets).issubset(self.joints) or duration_s <= 0:
            raise ValueError('State/action mismatch')
        row = [duration_s, *[state[joint] for joint in self.joints],
               *[targets.get(joint, state[joint]) for joint in self.joints]]
        if not np.isfinite(row).all():
            raise ValueError('Nonfinite model input')
        return row

    def predict_member(self, index, state, targets, duration_s):
        torch, _ = _torch_modules()
        if not 0 <= index < self.ensemble_size:
            raise IndexError('Unknown ensemble member')
        feature = torch.as_tensor(self._feature(state, targets, duration_s), dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        with torch.inference_mode():
            delta = self.models[index](feature).squeeze(0).cpu().numpy()
        return {joint: float(state[joint] + delta[i]) for i, joint in enumerate(self.joints)}

    def predict(self, state, targets, duration_s):
        predictions = [self.predict_member(i, state, targets, duration_s)
                       for i in range(self.ensemble_size)]
        return {joint: float(np.mean([item[joint] for item in predictions])) for joint in self.joints}

    def save(self, path):
        torch, _ = _torch_modules()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {'schema_version': 1, 'joints': self.joints, 'version': self.version,
                   'state_dicts': [{key: value.detach().cpu() for key, value in model.state_dict().items()}
                                   for model in self.models]}
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=destination.name + '.')
        os.close(descriptor)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path, *, device='cpu'):
        torch, _ = _torch_modules()
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get('schema_version') != 1:
            raise ValueError('Unknown neural dynamics checkpoint schema')
        return cls(payload['joints'], payload['state_dicts'], payload['version'], device=device)
