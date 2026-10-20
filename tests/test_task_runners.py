import math
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace

from wmal.locomotion.agent import G1Agent
from wmal.locomotion.contracts import G1State, G1Goal


def state(x=0., yaw=0.):
    return G1State('test', 0, 0., x, 0., .8, yaw, 0., 0., 0., 0., 0., 0., .8)


class CancellationTests(unittest.TestCase):
    def test_cancel_before_planning_stops_and_keeps_real_final_state(self):
        class Session:
            stopped = False
            def observe(self): return state(.7)
            def stop(self): self.stopped = True
        class Planner:
            def plan(self, *args): raise AssertionError('cancelled task must not plan')
        session = Session()
        result = G1Agent(Planner()).run_goal(G1Goal(1., 0.), session, cancel_requested=lambda: True)
        self.assertEqual(result.status, 'cancelled')
        self.assertEqual(result.final_state.x, .7)
        self.assertTrue(session.stopped)


class GoalResolutionTests(unittest.TestCase):
    def test_relative_intent_uses_latest_pose_and_degrees(self):
        from wmal.monitor.task_runners import resolve_g1_goals
        spec = {'relative_intent': {'kind': 'translation', 'direction': 'forward', 'distance_m': .4}}
        goal = resolve_g1_goals(spec, state(1., math.pi / 2))[0]
        self.assertAlmostEqual(goal.x, 1.)
        self.assertAlmostEqual(goal.y, .4)
        goal = resolve_g1_goals({'goal': [1., 0., 90.], 'waypoints': []}, state())[0]
        self.assertAlmostEqual(goal.yaw, math.pi / 2)

    def test_relative_translation_tolerance_scales_and_rejects_half_finished_short_move(self):
        from wmal.monitor.task_runners import resolve_g1_goals
        for distance, expected in ((.1,.04),(.2,.05),(.4,.1),(2.,.15)):
            for direction in ('forward','backward'):
                with self.subTest(distance=distance,direction=direction):
                    goal=resolve_g1_goals({'relative_intent':{'kind':'translation',
                        'direction':direction,'distance_m':distance}},state())[0]
                    self.assertAlmostEqual(goal.position_tolerance_m,expected)
        goal=resolve_g1_goals({'relative_intent':{'kind':'translation',
            'direction':'forward','distance_m':.2}},state())[0]
        self.assertFalse(G1Agent._goal_reached(state(.104),goal))
        self.assertTrue(G1Agent._goal_reached(state(.16),goal))
        absolute=resolve_g1_goals({'goal':[.2,0.,None],'waypoints':[]},state())[0]
        self.assertEqual(absolute.position_tolerance_m,G1Goal(.2,0.).position_tolerance_m)

    def test_initial_leg_counts_toward_total_path(self):
        from wmal.monitor.task_runners import resolve_g1_goals
        with self.assertRaisesRegex(ValueError, '2 m'):
            resolve_g1_goals({'goal': [1.9, 0., None], 'waypoints': [[1., 0., None]]}, state(-.2))

    def test_absent_arm_model_does_not_register_arm(self):
        from wmal.monitor.task_runners import build_task_runners
        with tempfile.TemporaryDirectory() as tmp:
            runners = build_task_runners(task_root=tmp)
            self.assertNotIn('arm_joint_targets', runners)


if __name__ == '__main__':
    unittest.main()

class RunnerLifecycleTests(unittest.TestCase):
    def test_same_mode_retains_state_and_switch_closes_old_scene_on_owner(self):
        import numpy as np
        from unittest.mock import patch
        from wmal.locomotion.agent import G1TaskResult
        from wmal.monitor.task_runners import build_task_runners
        events=[]
        semantics={'camera':'workcell_overview','period_s':.2}
        class Context:
            cancel_event=threading.Event()
            task_id='task'
            def check_cancelled(self): pass
            def progress(self,*args,**kwargs): pass
            def record_evidence(self,*args): pass
            def record_observation(self,*args): pass
            def record_prediction(self,*args): pass
        class Session:
            fault_reason=None
            video_frame_sink=None
            def __init__(self, mode, **kwargs):
                self.mode,self.actual,self.semantics=mode,state(),semantics
                self.data=SimpleNamespace(time=0.)
                events.append(('create',mode,threading.get_ident()))
            def observe(self):
                if self.mode=='g1': return self.actual
                return SimpleNamespace(episode_id='arm',step_id=0,state=np.array([.35,.87,.35,.87]))
            def close(self): events.append(('close',self.mode,threading.get_ident()))
            def stop(self): pass
            def abort(self,*args): pass
        def run_goal(agent,goal,session,**kwargs):
            session.actual=replace(session.actual,x=goal.x,y=goal.y)
            return G1TaskResult('succeeded',1,'fixture',session.actual)
        def arm_task(*args,**kwargs): return {'state':{'status':'succeeded'}}
        model=SimpleNamespace(version='visual',semantics=semantics,metadata={'training_source':{'task':'stack_block'}})
        calibration=SimpleNamespace(validate_for=lambda *args: None)
        with tempfile.TemporaryDirectory() as tmp, patch('wmal.monitor.task_runners.G1Agent.run_goal',run_goal), \
             patch('wmal.monitor.task_runners.runpy.run_path',return_value={'run_task':arm_task}):
            runners=build_task_runners(task_root=tmp,g1_planner=object(),visual_model=model,
                visual_calibration=calibration,g1_session_factory=lambda **kw: Session('g1',**kw),
                arm_session_factory=lambda **kw: Session('arm',**kw))
            spec={'relative_intent':{'kind':'translation','direction':'forward','distance_m':.4}}
            a=runners['g1_navigation'](spec,Context())
            b=runners['g1_navigation'](spec,Context())
            self.assertAlmostEqual(a['final_observation']['x'],.4)
            self.assertAlmostEqual(b['final_observation']['x'],.8)
            runners['arm_joint_targets']({'goal':[.4,.9],'options':{}},Context())
            runners['g1_navigation'](spec,Context())
            runners['g1_navigation'].close()
        self.assertEqual([(e[0],e[1]) for e in events],
                         [('create','g1'),('close','g1'),('create','arm'),('close','arm'),('create','g1'),('close','g1')])
        self.assertEqual({e[2] for e in events},{threading.get_ident()})

    def test_publisher_delegate_close_does_not_close_shared_publisher(self):
        from wmal.monitor.task_runners import PublisherDelegate
        shared=SimpleNamespace(ready=lambda *a: True,closed=False)
        handle=PublisherDelegate(shared)
        self.assertTrue(handle.ready())
        handle.close()
        self.assertFalse(handle.ready())
        self.assertFalse(shared.closed)

class VideoFailureTests(unittest.TestCase):
    def test_final_rgb_failure_finishes_with_last_real_frame_and_closes_session(self):
        import numpy as np
        from unittest.mock import patch
        from wmal.locomotion.agent import G1TaskResult
        from wmal.monitor.task_runners import build_task_runners
        class Context:
            task_id='task'
            cancel_event=threading.Event()
            def check_cancelled(self): pass
            def progress(self,*args,**kwargs): pass
            def record_evidence(self,*args): pass
            def record_prediction(self,*args): pass
            def record_observation(self,*args): pass
        class Session:
            closed=False
            video_frame_sink=None
            calls=0
            data=SimpleNamespace(time=0.)
            def observe(self): return replace(state(),sim_time_s=self.data.time)
            def stop(self): pass
            def close(self): self.closed=True
            def capture_video_frame(self,force=False):
                self.calls+=1
                if self.calls>1: raise RuntimeError('renderer failed')
                return np.full((8,8,3),17,dtype=np.uint8),0.
        class Recorder:
            finished=False
            def submit(self,*args): pass
            def finish(self,t,rgb):
                self.finished=True
                self.end_time=t
                self.rgb=rgb.copy()
                return {'status':'ready','path':'video.mp4'}
        session,recorder=Session(),Recorder()
        def execute(*args,**kwargs):
            session.data.time=1.2
            return G1TaskResult('succeeded',1,'fixture',session.observe())
        with tempfile.TemporaryDirectory() as tmp, patch('wmal.monitor.task_runners.G1Agent.run_goal',side_effect=execute):
            runners=build_task_runners(task_root=tmp,g1_planner=object(),
                g1_session_factory=lambda **kw:session,video_recorder_factory=lambda _:recorder)
            result=runners['g1_navigation']({'goal':[0.,0.,None]},Context())
        self.assertTrue(recorder.finished)
        self.assertAlmostEqual(recorder.end_time,1.2)
        self.assertTrue(session.closed)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['video']['status'],'failed')
        self.assertTrue(result['video']['partial'])
        self.assertTrue(np.all(recorder.rgb==17))


class FaultRecoveryTests(unittest.TestCase):
    def test_failed_arm_task_recreates_faulted_session_before_next_same_mode_task(self):
        import numpy as np
        from unittest.mock import patch
        from wmal.monitor.task_runners import build_task_runners
        events=[]
        semantics={'camera':'workcell_overview','period_s':.2}
        class Context:
            cancel_event=threading.Event()
            task_id='first'
            def check_cancelled(self): pass
            def progress(self,*args,**kwargs): pass
            def record_evidence(self,*args): pass
            def record_observation(self,*args): pass
            def record_prediction(self,*args): pass
        class Session:
            fault_reason=None
            video_frame_sink=None
            semantics={'camera':'workcell_overview','period_s':.2}
            data=SimpleNamespace(time=0.)
            def __init__(self,**kwargs): events.append(('create',threading.get_ident()))
            def observe(self):
                return SimpleNamespace(episode_id='arm',step_id=0,state=np.array([.35,.87,.35,.87]))
            def abort(self,reason): self.fault_reason=reason
            def close(self): events.append(('close',threading.get_ident()))
        model=SimpleNamespace(version='visual',semantics=semantics,metadata={'training_source':{'task':'stack_block'}})
        calibration=SimpleNamespace(validate_for=lambda *args: None)
        reports=iter([{'state':{'status':'needs_review'}},{'state':{'status':'succeeded'}}])
        with tempfile.TemporaryDirectory() as tmp, patch('wmal.monitor.task_runners.runpy.run_path',
                return_value={'run_task':lambda *args,**kwargs:next(reports)}):
            runner=build_task_runners(task_root=tmp,visual_model=model,visual_calibration=calibration,
                                     arm_session_factory=Session)['arm_joint_targets']
            first=runner({'goal':[.4,.9],'options':{}},Context())
            context=Context();context.task_id='second'
            second=runner({'goal':[.4,.9],'options':{}},context)
            runner.close()
        self.assertEqual(first['status'],'failed')
        self.assertEqual(second['status'],'succeeded')
        self.assertEqual([e[0] for e in events],['create','close','create','close'])
        self.assertEqual({e[1] for e in events},{threading.get_ident()})

class ServiceRunnerContractTests(unittest.TestCase):
    def wait_terminal(self,service,task_id):
        import time
        until=time.monotonic()+3.
        while time.monotonic()<until:
            snapshot=service.get(task_id)
            if snapshot.state in ('succeeded','failed','cancelled'):
                return snapshot
            time.sleep(.005)
        self.fail('runner service did not finish')

    def test_parsed_g1_taskspec_runs_through_real_live_task_service(self):
        from wmal.monitor.commands import parse_command
        from wmal.monitor.live_tasks import LiveTaskService
        from wmal.monitor.task_runners import build_task_runners
        class Session:
            video_frame_sink=None
            def observe(self): return state()
            def stop(self): pass
            def close(self): pass
        with tempfile.TemporaryDirectory() as tmp:
            runners=build_task_runners(task_root=tmp,g1_planner=object(),g1_session_factory=lambda **kw:Session())
            service=LiveTaskService(tmp,runners)
            try:
                spec=parse_command({'mode':'g1_navigation','goal':{'x_m':0.,'y_m':0.,'yaw_deg':0.}})
                queued=service.submit(spec,'g1-spec','g1-spec-fingerprint')
                result=self.wait_terminal(service,queued.task_id)
                self.assertEqual(result.state,'succeeded',result)
                self.assertEqual(result.result['agent_status'],'succeeded')
            finally:
                service.close(2.)

    def test_parsed_arm_taskspec_options_run_through_real_live_task_service(self):
        import numpy as np
        from unittest.mock import patch
        from wmal.monitor.commands import parse_command
        from wmal.monitor.live_tasks import LiveTaskService
        from wmal.monitor.task_runners import build_task_runners
        semantics={'camera':'workcell_overview','period_s':.2}
        class Session:
            video_frame_sink=None
            fault_reason=None
            data=SimpleNamespace(time=0.)
            def __init__(self,**kwargs): self.semantics=semantics
            def observe(self):
                return SimpleNamespace(episode_id='arm',step_id=0,state=np.array([.35,.87,.35,.87]))
            def abort(self,reason): self.fault_reason=reason
            def close(self): pass
        model=SimpleNamespace(version='visual',semantics=semantics,metadata={'training_source':{'task':'stack_block'}})
        calibration=SimpleNamespace(validate_for=lambda *args:None)
        seen=[]
        def run_task(*args,**kwargs):
            seen.append(kwargs)
            return {'state':{'status':'succeeded'}}
        with tempfile.TemporaryDirectory() as tmp, patch('wmal.monitor.task_runners.runpy.run_path',return_value={'run_task':run_task}):
            runners=build_task_runners(task_root=tmp,visual_model=model,visual_calibration=calibration,arm_session_factory=Session)
            service=LiveTaskService(tmp,runners)
            try:
                spec=parse_command({'mode':'arm_joint_targets','task':'stack_block','goal_rad':[.4,.9],
                                    'baseline':'A3','horizon':2,'max_cycles':3,'error_budget':.5})
                queued=service.submit(spec,'arm-spec','arm-spec-fingerprint')
                result=self.wait_terminal(service,queued.task_id)
                self.assertEqual(result.state,'succeeded',result)
                self.assertEqual(seen[0]['horizon'],2)
                self.assertEqual(seen[0]['baseline'],'A3')
            finally:
                service.close(2.)
