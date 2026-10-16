"""Runtime model/planner boundary of the split motion-network integration."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

@unittest.skipUnless(importlib.util.find_spec('torch'), 'Install model-runtime extra')
class MotionPlanningContractTests(unittest.TestCase):
    def test_saved_motion_model_loads_and_plans(self):
        import torch
        from wmal.models.motion_network import MotionNetwork, build_network, load, SCHEMA, cpu_states
        from wmal.locomotion.contracts import G1State, G1Goal
        from wmal.locomotion.planner import G1RolloutPlanner
        # Architecture and serialization fixture; no weight optimization.
        torch.manual_seed(12)
        networks = [build_network((16,16)) for _ in range(2)]
        with torch.no_grad():
            for network in networks:
                for parameter in network.parameters(): parameter.zero_()
                network[-1].bias[5] = .8
        normalization = {'x_mean': [0.]*10, 'x_scale': [1.]*10,
                         'y_mean': [0.]*6, 'y_scale': [1.]*6}
        model = MotionNetwork({'schema':SCHEMA, 'state_dicts':cpu_states(networks),
                              'normalization':normalization, 'duration_s':.5, 'hidden':[16,16]})
        state = G1State('fixture',0,0.,0.,0.,.8,0.,0.,0.,0.,0.,0.,0.,.8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            model.save(path)
            restored = load({'checkpoint':path})
            plan = G1RolloutPlanner(restored,samples=4,horizon=2).plan(state,G1Goal(.3,0.))
            self.assertEqual(plan.model_version,model.version)
            self.assertEqual(plan.uncertainty_kind,'ensemble_spread')
            self.assertEqual(plan.predicted_state.step_id,state.step_id+1)
