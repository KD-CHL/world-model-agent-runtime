"""Physical alignment, ensemble trajectories, resumability and execution-data provenance."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from test_motion_network import fixture, TORCH_AVAILABLE
from wmal.datasets.motion_sequences import load_motion_episodes


@unittest.skipUnless(TORCH_AVAILABLE,'Install learning extra')
class PlanningNetworkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset = self.root/'data.json'
        fixture(self.dataset)


    def test_velocity_targets_reconstruct_terminal_state_when_average_velocity_differs(self):
        import torch
        from wmal.models.motion_network import target_outputs,integrate
        before = torch.tensor([[1.,2.,1.1,.2,.3,.4,.02,.03,.8]])
        after = torch.tensor([[1.01,2.07,1.3,.1,.7,-.2,.04,.05,.79]])
        action = torch.tensor([[.2,.0,.2,.5]])
        outputs = target_outputs(before,after,action,'latent_state')
        self.assertEqual(outputs.shape[-1],9)
        torch.testing.assert_close(integrate(before,outputs,action),after)
        legacy = integrate(before,target_outputs(before,after,action),action)
        self.assertGreater(float((legacy[:,3:6]-after[:,3:6]).abs().sum()),.5)


    def test_rollout_retains_nonlinear_member_paths_instead_of_merging_every_step(self):
        import torch
        from wmal.models.motion_network import build_network,MotionNetwork,SCHEMA,cpu_states
        from wmal.locomotion.contracts import G1VelocityAction
        base = [build_network((4,)) for _ in range(2)]
        norm = {'x_mean':[0.]*10,'x_scale':[1.]*10,'y_mean':[0.]*6,'y_scale':[1.]*6}
        model = MotionNetwork({'schema':SCHEMA,'hidden':[4],'duration_s':.5,
                              'normalization':norm,'state_dicts':cpu_states(base)})
        class QuadraticMember(torch.nn.Module):
            def __init__(self,gain):
                super().__init__()
                self.gain=gain
            def forward(self,x):
                v = (x[...,0]+self.gain*x[...,6]).square()
                zero = torch.zeros_like(v)
                return torch.stack([v,zero,zero,zero,zero,zero+.8],dim=-1)
        # Analytical fixture: member gains .5 and 1.5, zero initial velocity, command .2.
        model.models = [QuadraticMember(.5),QuadraticMember(1.5)]
        state = next(iter(load_motion_episodes(self.dataset)[0]['train'].values()))[0][0]
        action = G1VelocityAction(.2,0.,0.,.5)
        result = model.rollout(state,[[action,action]])
        self.assertAlmostEqual(result.members[0,0,1,3],.0121,places=6)
        self.assertAlmostEqual(result.members[1,0,1,3],.1521,places=6)
        mean_feedback = model.predict(model.predict(state,action,.5).state,action,.5)
        self.assertGreater(abs(result.prediction(0,1).state.vx-mean_feedback.state.vx),.005)


class ExecutionExportTests(unittest.TestCase):
    def test_mean_path_is_checked_even_when_member_paths_clear_an_obstacle(self):
        from wmal.locomotion.contracts import G1State,G1Goal,G1_STATE_SCHEMA,G1_ACTION_SCHEMA
        from wmal.locomotion.rollouts import EnsembleRollout
        from wmal.locomotion.planner import G1RolloutPlanner
        state=G1State('episode',0,0.,0.,0.,.8,0.,0.,0.,0.,0.,0.,0.,.8)
        class Provider:
            state_schema,action_schema,version=G1_STATE_SCHEMA,G1_ACTION_SCHEMA,'fixture'
            def predict(self,*args): raise AssertionError('Batch endpoint expected')
            def rollout(self,observed,candidates):
                values=np.zeros((2,len(candidates),len(candidates[0]),9))
                values[...,8]=.8
                values[...,0]=.1
                values[:,0,:,0]=.5
                values[0,0,:,1]=.2
                values[1,0,:,1]=-.2
                return EnsembleRollout(self.version,observed,tuple(tuple(r) for r in candidates),values)
        class MeanObstacle:
            def segment_free(self,a,b):
                return not (abs(b[0]-.5)<.01 and abs(b[1])<.1)
        planner=G1RolloutPlanner(Provider(),samples=4,horizon=1)
        planner.scene=MeanObstacle()
        plan=planner.plan(state,G1Goal(.5,0.))
        self.assertNotEqual(plan.evidence['selected_candidate'],0)

    def test_individual_member_constraint_is_screened_and_wrong_rollout_episode_is_rejected(self):
        from wmal.locomotion.contracts import G1State,G1Goal,G1_STATE_SCHEMA,G1_ACTION_SCHEMA
        from wmal.locomotion.rollouts import EnsembleRollout
        from wmal.locomotion.planner import G1RolloutPlanner
        from wmal.locomotion.world_model import WorldModelAdapter
        state = G1State('episode',0,0.,0.,0.,.8,0.,0.,0.,0.,0.,0.,0.,.8)
        class Provider:
            state_schema,action_schema,version = G1_STATE_SCHEMA,G1_ACTION_SCHEMA,'fixture'
            def predict(self,*args): raise AssertionError('Batch endpoint should be used')
            def rollout(self,observed,candidates):
                values = np.zeros((2,len(candidates),len(candidates[0]),9))
                values[...,8]=.8
                values[...,0]=.1
                values[:,0,:,0]=.5
                values[0,0,:,6]=.8  # Mean tilt .4 is admissible, but one member violates .65.
                return EnsembleRollout(self.version,observed,tuple(tuple(r) for r in candidates),values)
        planner = G1RolloutPlanner(Provider(),samples=4,horizon=2)
        plan = planner.plan(state,G1Goal(.5,0.))
        self.assertNotEqual(plan.evidence['selected_candidate'],0)
        self.assertEqual(plan.evidence['candidates'][0]['rejection'],'predicted_posture_constraint')
        class StaleProvider(Provider):
            def rollout(self,observed,candidates):
                return super().rollout(replace(observed,episode_id='stale'),candidates)
        with self.assertRaises(ValueError):
            WorldModelAdapter(StaleProvider()).rollout(state,[[plan.action]])

    def test_language_recovery_receives_bounded_prediction_evidence(self):
        from wmal.agents.g1_coordinator import LanguageMissionPlanner
        from wmal.envs.indoor_scene import IndoorScene
        from wmal.locomotion.contracts import G1State,G1Goal
        class Client:
            model='offline-fixture'
            def request_json(self,payload):
                self.payload=payload
                return {'status':'planned','goals':[{'x':.3,'y':0.,'yaw':None}]}
        client=Client()
        state = G1State('episode',0,0.,0.,0.,.8,0.,0.,0.,0.,0.,0.,0.,.8)
        evidence={'model_version':'weights-v2','success_probability':None,'top_candidates':[]}
        goals=LanguageMissionPlanner(client).recover('move',G1Goal(.3,0.),state,IndoorScene(),
            'weights-v2','no_candidate',1,0,evidence)
        context=json.loads(client.payload['messages'][1]['content'])
        self.assertIn('"prediction_evidence"',context['instruction'])
        self.assertIn('"success_probability": null',context['instruction'])
        self.assertEqual(goals[-1],G1Goal(.3,0.))

    def test_episode_assignment_and_partial_execution_audit(self):
        from wmal.datasets.execution_export import export_execution_logs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = fixture(root/'fixture.json')['rows']
            events = [{'event':'executed_transition','payload':{'schema':'wmal.executed_transition.v1',
                       **{k:r[k] for k in ('before','action','after')},'model_version':'behavior-v1'}} for r in rows]
            events.append({'event':'command_uncertain','payload':{'command_id':'partial'}})
            log = root/'events.jsonl'
            log.write_text('\n'.join(json.dumps(e) for e in events))
            assignments = {r['before']['episode_id']:r['split'] for r in rows}
            report = export_execution_logs([log],root/'export.json',assignments,source='mujoco_interaction')
            self.assertEqual(report['transitions'],96)
            self.assertEqual(report['audit_events'],1)
            self.assertEqual(load_motion_episodes(root/'export.json')[1],.5)
            with self.assertRaises(ValueError):
                export_execution_logs([log],root/'bad.json',{'fixture-0':'train'},source='mujoco_interaction')
            self.assertFalse((root/'bad.json').exists())
            with self.assertRaises(ValueError):
                export_execution_logs([log,log],root/'dup.json',assignments,source='mujoco_interaction')
            self.assertFalse((root/'dup.json').exists())


if __name__=='__main__': unittest.main()
