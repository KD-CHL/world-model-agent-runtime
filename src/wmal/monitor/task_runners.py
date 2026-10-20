"""Real local Agent runners, with simulator access confined to their owner thread.

The returned callables share one session manager. Same-mode tasks retain the
real state; mode changes close the old scene before constructing the next one.
No HTTP input can choose a model loader, file, or simulator implementation.
"""
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
import runpy
import threading

from wmal.locomotion.agent import G1Agent
from wmal.locomotion.contracts import G1Goal
from wmal.monitor.commands import MAX_G1_DISTANCE_M, TaskSpec


def _spec_mapping(spec):
    """Read immutable service TaskSpec without deepcopying MappingProxy fields."""
    if isinstance(spec, TaskSpec):
        return {'mode': spec.mode, 'task': spec.task, 'goal': spec.goal,
                'waypoints': spec.waypoints, 'options': spec.options,
                'relative_intent': spec.relative_intent}
    if isinstance(spec, Mapping):
        return spec
    raise TypeError('Runner requires a TaskSpec or task mapping')


def resolve_g1_goals(spec, initial_state):
    """Resolve relative requests only after dequeue against the real observation."""
    spec = _spec_mapping(spec)
    intent = spec.get('relative_intent')
    if intent:
        if intent['kind'] == 'translation':
            signed = intent['distance_m'] * (1 if intent['direction'] == 'forward' else -1)
            # Small requested moves need a smaller completion radius than the
            # legacy world-coordinate default; keep a 4 cm servo/noise floor.
            tolerance = min(.15, max(.04, intent['distance_m'] * .25))
            goals = [G1Goal(initial_state.x + math.cos(initial_state.yaw)*signed,
                            initial_state.y + math.sin(initial_state.yaw)*signed,
                            position_tolerance_m=tolerance)]
        else:
            signed = intent['angle_deg'] * (1 if intent['direction'] == 'left' else -1)
            yaw = (initial_state.yaw + math.radians(signed) + math.pi) % (2*math.pi)-math.pi
            goals = [G1Goal(initial_state.x, initial_state.y, yaw)]
    else:
        goals = [G1Goal(row[0], row[1], None if row[2] is None else math.radians(row[2]))
                 for row in list(spec.get('waypoints', ())) + [spec['goal']]]
    previous, distance = (initial_state.x, initial_state.y), 0.
    for goal in goals:
        distance += math.hypot(goal.x-previous[0], goal.y-previous[1])
        previous = (goal.x, goal.y)
    if distance > MAX_G1_DISTANCE_M + 1e-9:
        raise ValueError('Full path from current real pose must be at most 2 m')
    return goals


class PublisherDelegate:
    """Session-local handle; closing it never closes the Runtime publisher."""
    def __init__(self, publisher):
        self.publisher, self.closed = publisher, False

    def __getattr__(self, name):
        return getattr(self.publisher, name)

    def ready(self, *args, **kwargs):
        return not self.closed and self.publisher.ready(*args, **kwargs)

    def publish(self, *args, **kwargs):
        return False if self.closed else self.publisher.publish(*args, **kwargs)

    def close(self):
        self.closed = True


def _observation(session, mode):
    actual = session.observe()
    if mode == 'g1_navigation':
        return asdict(actual)
    return {'episode_id': actual.episode_id, 'step_id': actual.step_id,
            'state': actual.state.tolist(), 'sim_time_s': float(session.data.time)}


def _video_data(result):
    data = asdict(result) if is_dataclass(result) else dict(result)
    if data.get('path') is not None:
        # Only a task-local artifact name is exposed, never an arbitrary path.
        data['path'] = Path(data['path']).name
    return data


class _SessionOwner:
    def __init__(self, factories, publisher):
        self.factories, self.publisher = factories, publisher
        self.mode = self.session = self.thread_id = None

    def _assert_owner(self):
        current = threading.get_ident()
        if self.thread_id is None:
            self.thread_id = current
        elif current != self.thread_id:
            raise RuntimeError('Simulator runners must use one owner thread')

    def get(self, mode):
        self._assert_owner()
        faulted = self.session is not None and getattr(self.session, 'fault_reason', None) is not None
        if self.mode != mode or self.session is None or faulted:
            self.close()
            delegate = PublisherDelegate(self.publisher) if self.publisher is not None else None
            self.session = self.factories[mode](frame_publisher=delegate)
            self.mode = mode
        return self.session

    def close(self):
        self._assert_owner()
        if self.session is not None:
            session, self.session = self.session, None
            self.mode = None
            session.close()


class _TaskRunner:
    def __init__(self, mode, owner, task_root, execute, video_factory):
        self.mode, self.owner, self.root = mode, owner, Path(task_root)
        self.execute, self.video_factory = execute, video_factory

    def close(self):
        self.owner.close()

    def __call__(self, spec, context):
        spec = _spec_mapping(spec)
        context.check_cancelled()
        session = self.owner.get(self.mode)
        task_dir = self.root / context.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        recorder = None
        last_frame = None
        capture_failed = False
        outcome = {'status': 'failed', 'video': {'status': 'unavailable'}}
        try:
            context.record_observation({'phase': 'initial', **_observation(session, self.mode)})
            if self.video_factory is not None:
                # Capture before starting the encoder: a failed initial renderer
                # must not leave an encoder waiting forever for its first frame.
                rgb, timestamp = session.capture_video_frame(force=True)
                last_frame = (rgb, timestamp)
                recorder = self.video_factory(task_dir)
                def sink(frame, sim_time):
                    nonlocal last_frame
                    last_frame = (frame.copy(), sim_time)
                    return recorder.submit(frame, sim_time)
                sink(rgb, timestamp)
                session.video_frame_sink = sink
            context.progress('executing', 0., mode=self.mode)
            outcome.update(self.execute(session, spec, context, task_dir))
        except Exception as exc:
            outcome.update(status='cancelled' if context.cancel_event.is_set() else 'failed',
                           error={'type': type(exc).__name__, 'message': str(exc)})
        finally:
            try:
                if self.mode == 'g1_navigation':
                    session.stop()
                elif outcome['status'] != 'succeeded':
                    session.abort('cancelled' if context.cancel_event.is_set() else 'task_stopped')
            except Exception as exc:
                outcome.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
                capture_failed = True
            try:
                final = _observation(session, self.mode)
                outcome['final_observation'] = final
                context.record_observation({'phase': 'final', **final})
                if recorder is not None:
                    session.video_frame_sink = None
                    rgb, timestamp = session.capture_video_frame(force=True)
                    context.progress('encoding', 1.)
                    outcome['video'] = _video_data(recorder.finish(timestamp, rgb))
            except Exception as exc:
                capture_failed = True
                outcome.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
                if recorder is not None:
                    # Finish with the last actual frame to release the encoder.
                    # Hold that actual frame through the measured simulation
                    # endpoint. The timeline remains intact while missing final
                    # visual evidence is explicitly partial, never ready.
                    try:
                        rgb, timestamp = last_frame
                        last_frame_time = timestamp
                        end_time_source = 'last_actual_frame'
                        try:
                            actual_time = float(session.data.time)
                            if math.isfinite(actual_time) and actual_time >= timestamp:
                                timestamp = actual_time
                                end_time_source = 'session_simulation_time'
                        except (AttributeError, TypeError, ValueError, OverflowError):
                            pass
                        partial = _video_data(recorder.finish(timestamp, rgb))
                        partial.update(status='failed', partial=True,
                                       last_actual_frame_sim_time_s=last_frame_time,
                                       end_sim_time_source=end_time_source,
                                       error='Final actual RGB unavailable: '+str(exc))
                        outcome['video'] = partial
                    except Exception as finish_error:
                        if callable(getattr(recorder, 'abort', None)):
                            recorder.abort(str(finish_error))
                        outcome['video'] = {'status': 'failed', 'error': str(finish_error)}
            finally:
                session.video_frame_sink = None
                if capture_failed:
                    self.owner.close()
        if context.cancel_event.is_set():
            outcome['status'] = 'cancelled'
        return outcome


def build_task_runners(*, task_root, g1_model=None, g1_planner=None,
                       visual_model=None, visual_calibration=None,
                       frame_publisher=None, video_recorder_factory=None,
                       realtime=True, g1_session_factory=None, arm_session_factory=None,
                       g1_agent_options=None, g1_max_cycles=100):
    """Build closures from preflight-approved local models, never browser paths.

    The factory itself constructs no MuJoCo objects. Missing model/calibration
    resources omit the corresponding mode. Injectable sessions/recorders support
    deterministic lifecycle tests; production defaults always use real sessions.
    """
    if type(g1_max_cycles) is not int or not 1 <= g1_max_cycles <= 100:
        raise ValueError('G1 cycle budget must be in [1, 100]')
    factories, executors = {}, {}
    if g1_planner is None and g1_model is not None:
        from wmal.locomotion.planner import G1RolloutPlanner
        g1_planner = G1RolloutPlanner(g1_model)
    if g1_planner is not None:
        def make_g1(**kwargs):
            if g1_session_factory is not None:
                return g1_session_factory(**kwargs)
            from wmal.envs.g1_session import G1MuJoCoSession
            return G1MuJoCoSession(viewer=False, realtime=realtime, **kwargs)
        factories['g1_navigation'] = make_g1

        def execute_g1(session, spec, context, task_dir):
            goals = resolve_g1_goals(spec, session.observe())
            context.record_evidence('resolved_goals', {'goals': [asdict(goal) for goal in goals],
                                    'source': 'real_pose_after_dequeue'})
            def log(event, payload):
                if event == 'plan':
                    context.record_prediction(payload)
                elif event == 'executed_transition':
                    context.record_observation(payload)
                else:
                    context.record_evidence(event, payload)
            agent = G1Agent(g1_planner, log=log, **(g1_agent_options or {}))
            cycles, results = 0, []
            for goal in goals:
                context.check_cancelled()
                if hasattr(session, 'set_goal_marker'):
                    session.set_goal_marker(goal)
                result = agent.run_goal(goal, session, max_cycles=max(1, g1_max_cycles-cycles),
                                        cancel_requested=context.cancel_event.is_set)
                results.append(asdict(result))
                cycles += result.cycles
                if result.status != 'succeeded':
                    return {'status': 'cancelled' if result.status == 'cancelled' else 'failed',
                            'agent_status': result.status, 'goals': results, 'cycles': cycles}
                if cycles >= g1_max_cycles and len(results) < len(goals):
                    return {'status': 'failed', 'agent_status': 'budget_exhausted',
                            'goals': results, 'cycles': cycles}
            return {'status': 'succeeded', 'agent_status': 'succeeded', 'goals': results, 'cycles': cycles}
        executors['g1_navigation'] = execute_g1
    if visual_model is not None and visual_calibration is not None:
        if visual_model.metadata.get('training_source', {}).get('task') != 'stack_block':
            raise ValueError('Visual checkpoint is not scoped to stack_block')
        visual_calibration.validate_for(visual_model.version, visual_model.semantics)
        def make_arm(**kwargs):
            if arm_session_factory is not None:
                session = arm_session_factory(**kwargs)
            else:
                from wmal.envs.visual_workcell import VisualWorkcellSession
                session = VisualWorkcellSession('stack_block', image_size=visual_model.config.image_size,
                    camera=visual_model.semantics['camera'], period_s=visual_model.semantics['period_s'],
                    realtime=realtime, **kwargs)
            if session.semantics != visual_model.semantics:
                session.close()
                raise ValueError('Visual session/model camera or action semantics mismatch')
            return session
        factories['arm_joint_targets'] = make_arm

        def execute_arm(session, spec, context, task_dir):
            if session.fault_reason is not None:
                raise RuntimeError('Arm session remained faulted after owner recovery')
            run_task = runpy.run_path(str(Path(__file__).resolve().parents[3] / 'scripts/visual_skill_agent.py'))['run_task']
            def on_event(event, payload):
                if event == 'plan':
                    context.record_prediction(payload)
                elif event in ('observation', 'feedback', 'cancelled'):
                    context.record_observation(payload)
                else:
                    context.record_evidence(event, payload)
            with (task_dir / 'runner_events.jsonl').open('w', encoding='utf-8') as log:
                result = run_task(session, visual_model, visual_calibration, spec['goal'],
                    **spec['options'], seed=0, log=log, artifact_dir=task_dir,
                    waypoints=spec.get('waypoints', ()), cancel_requested=context.cancel_event.is_set,
                    on_event=on_event)
            status = result['state']['status']
            return {'status': 'succeeded' if status == 'succeeded' else 'cancelled' if status == 'cancelled' else 'failed',
                    'agent_status': status, 'agent_result': result}
        executors['arm_joint_targets'] = execute_arm
    owner = _SessionOwner(factories, frame_publisher)
    return {mode: _TaskRunner(mode, owner, task_root, execute, video_recorder_factory)
            for mode, execute in executors.items()}
