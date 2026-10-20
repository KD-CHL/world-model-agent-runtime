"""Real local MuJoCo safe boundaries; no GUI or learned-model task acceptance."""
import io
import runpy
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np


class SessionCancellationTests(unittest.TestCase):
    def test_g1_cancel_preserves_one_executed_policy_tick_and_zero_command(self):
        from wmal.envs.g1_session import G1MuJoCoSession
        from wmal.locomotion.contracts import G1VelocityAction
        event=threading.Event()
        with G1MuJoCoSession(viewer=False,realtime=False) as session:
            initial=session.observe()
            control=session._control_interval
            def one_interval():
                control()
                event.set()
            session._control_interval=one_interval
            actual=session.step(G1VelocityAction(.1,0.,0.,.2),cancel_requested=event.is_set)
            self.assertAlmostEqual(actual.sim_time_s-initial.sim_time_s,.02,places=6)
            self.assertEqual(actual.step_id,1)
            self.assertIsNone(session._video_renderer)
            np.testing.assert_array_equal(session.policy.command,[0.,0.,0.])

    def test_arm_cancel_keeps_first_target_and_captures_real_endpoints_in_owner(self):
        from wmal.envs.visual_workcell import VisualWorkcellSession
        failures=[]
        def worker():
            try:
                event=threading.Event()
                frames=[]
                with VisualWorkcellSession(image_size=32,video_frame_sink=lambda rgb,t: frames.append((rgb.copy(),t))) as session:
                    before=session.observe()
                    session.capture_video_frame(force=True)
                    after=session.execute_actions([[.04,0.],[.04,0.]],episode_id=before.episode_id,
                        step_id=before.step_id,on_observation=lambda _: event.set(),cancel_requested=event.is_set)
                    session.capture_video_frame(force=True)
                    self.assertEqual(after.step_id,1)
                    self.assertEqual(session.fault_reason,'cancelled')
                    np.testing.assert_allclose(session.data.ctrl[session.aids],session.data.qpos[session.qadr])
                    self.assertGreater(frames[-1][1],frames[0][1])
                    self.assertEqual(frames[0][0].shape,(32,32,3))
                    self.assertEqual(frames[0][0].dtype,np.uint8)
            except BaseException as exc:
                failures.append(exc)
        thread=threading.Thread(target=worker)
        thread.start();thread.join(30)
        self.assertFalse(thread.is_alive())
        if failures: raise failures[0]

    def test_arm_task_cancel_before_plan_returns_actual_observation(self):
        from wmal.envs.visual_workcell import VisualWorkcellSession
        run_task=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/visual_skill_agent.py'))['run_task']
        with TemporaryDirectory() as tmp, VisualWorkcellSession(image_size=32) as session:
            before=session.observe()
            model=SimpleNamespace(version='a0_fixture',semantics=session.semantics)
            result=run_task(session,model,None,[.4,.9],baseline='A0',horizon=1,max_cycles=2,
                error_budget=1.,seed=0,log=io.StringIO(),artifact_dir=Path(tmp),cancel_requested=lambda: True)
            self.assertEqual(result['state']['status'],'cancelled')
            self.assertEqual(result['final_observation']['step_id'],before.step_id)
            self.assertEqual(session.fault_reason,'cancelled')


class G1VideoTimelineTests(unittest.TestCase):
    def test_aligned_action_end_does_not_submit_duplicate_real_frame_timestamp(self):
        import shutil
        from wmal.envs.g1_session import G1MuJoCoSession
        from wmal.locomotion.contracts import G1VelocityAction
        from wmal.monitor.video_recorder import TaskVideoRecorder
        ffmpeg,ffprobe=shutil.which('ffmpeg'),shutil.which('ffprobe')
        if ffmpeg is None or ffprobe is None:
            self.skipTest('ffmpeg and ffprobe are required for real task-video validation')
        with TemporaryDirectory() as tmp, G1MuJoCoSession(viewer=False,realtime=False) as session:
            recorder=TaskVideoRecorder(tmp,ffmpeg_path=ffmpeg,ffprobe_path=ffprobe,fps=15)
            timestamps=[]
            def sink(rgb,t):
                timestamps.append(t)
                recorder.submit(rgb,t)
            session.video_frame_sink=sink
            try:
                session.capture_video_frame(force=True)
                session.step(G1VelocityAction(.1,0.,0.,.16))
                session.video_frame_sink=None
                rgb,end=session.capture_video_frame(force=True)
                result=recorder.finish(end,rgb)
            finally:
                session.video_frame_sink=None
                recorder.abort('test cleanup')
            self.assertEqual(result.status,'ready',result)
            self.assertEqual(len(timestamps),len(set(timestamps)))
            self.assertAlmostEqual(result.end_sim_time_s-result.start_sim_time_s,.16,places=6)
            self.assertGreaterEqual(result.duration_s,.16)
