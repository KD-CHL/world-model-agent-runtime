"""Start the local instruction, world-model prediction and simulation client.

Only local experiment files are loaded. Physics is created later by the task
owner, never by HTTP handlers or model preflight.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import signal
import threading
import uuid

from wmal.locomotion.planner import G1RolloutPlanner
from wmal.locomotion.world_model import load_world_model

ROOT = Path(__file__).resolve().parents[1]


def _bounded_int(low, high):
    def parse(value):
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f'must be in [{low}, {high}]')
        return number
    return parse


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g1-config', default=str(ROOT/'configs/g1_planning_network.json'))
    parser.add_argument('--visual-checkpoint')
    parser.add_argument('--visual-calibration')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--runs-root', default=str(ROOT/'runs'))
    parser.add_argument('--run-dir', help='New/empty directory inside runs-root; relative paths use runs-root')
    parser.add_argument('--port', type=_bounded_int(0, 65535), default=8765)
    parser.add_argument('--g1-cycles', type=_bounded_int(1, 100), default=30)
    parser.add_argument('--max-pending', type=_bounded_int(1, 128), default=8)
    parser.add_argument('--realtime', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--video-fps', type=_bounded_int(5, 60), default=15)
    parser.add_argument('--shutdown-timeout', type=float, default=10.)
    return parser


def config_root(path):
    """Resolve repo-relative experiment assets from the owning configuration."""
    path = Path(path).expanduser().resolve()
    for parent in path.parents:
        if (parent/'pyproject.toml').is_file() or (parent/'.git').exists():
            return parent
    # Standalone configuration bundles resolve their local assets beside JSON.
    return path.parent


def _asset_path(value, base):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base/path).resolve()


def _load_visual(path, device):
    from wmal.models.visual_latent import VisualWorldModel
    return VisualWorldModel.load(path, device=device)


def _load_calibration(path):
    from wmal.models.horizon_calibration import HorizonCalibration
    return HorizonCalibration.load(path)


def _check_simulation_assets(mode):
    if importlib.util.find_spec('mujoco') is None:
        raise ValueError('MuJoCo 仿真依赖未安装')
    if mode == 'g1_navigation':
        from wmal.envs.g1_locomotion import POLICY_ONNX
        if not POLICY_ONNX.is_file():
            raise ValueError('G1 底层运动策略文件缺失')
        import onnxruntime
        # Validate the actual local policy without creating a physics session.
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        onnxruntime.InferenceSession(str(POLICY_ONNX), sess_options=options,
                                     providers=['CPUExecutionProvider'])


def _validate_visual(model, calibration):
    from wmal.envs.visual_workcell import JOINTS, ACTION_SCHEMA
    semantics = model.semantics
    period = semantics.get('period_s')
    expected = {'action_schema': ACTION_SCHEMA, 'action_order': list(JOINTS),
                'state_order': list(JOINTS)+[joint+'_target' for joint in JOINTS],
                'camera': semantics.get('camera'), 'period_s': period,
                'image_size': model.config.image_size, 'event_names': []}
    if (semantics != expected or expected['camera'] not in
            ('workcell_overview', 'workcell_top', 'workcell_front') or
            type(period) not in (int, float) or not math.isfinite(period) or
            not 0 < period <= 1 or abs(round(period/.002)*.002-period) > 1e-8 or
            model.config.image_size not in (32, 64, 128)):
        raise ValueError('Visual checkpoint camera/action/state semantics 不兼容')
    if model.metadata.get('training_source', {}).get('task') != 'stack_block':
        raise ValueError('Visual checkpoint 的训练场景必须为 stack_block')
    calibration.validate_for(model.version, semantics)
    horizon = model.metadata.get('max_horizon')
    if type(horizon) is not int or horizon < 1 or len(calibration.quantiles) < 1:
        raise ValueError('Visual checkpoint/calibration 缺少有效预测时域')
    if len(calibration.state_scale) != 4:
        raise ValueError('Visual calibration 状态维度不兼容')
    return min(12, horizon, len(calibration.quantiles))


def preflight(args):
    """Omit unavailable modes, retaining clear reasons and no model fallback."""
    result = {'g1_planner': None, 'g1_agent_options': {}, 'g1_scene': None,
              'visual_model': None, 'visual_calibration': None}
    modes, reasons, arm_horizon = [], {}, None
    try:
        config_path = Path(args.g1_config).expanduser().resolve(strict=True)
        config = json.loads(config_path.read_text(encoding='utf-8'),
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite config')))
        world = config['world_model']
        settings = dict(world.get('config', {}))
        settings['device'] = args.device
        base = config_root(config_path)
        if 'checkpoint' in settings:
            settings['checkpoint'] = str(_asset_path(settings['checkpoint'], base))
            if not Path(settings['checkpoint']).is_file():
                raise ValueError('G1 世界模型 checkpoint 文件缺失')
        _check_simulation_assets('g1_navigation')
        model = load_world_model(world['factory'], settings)
        if world.get('expected_version') and model.version != world['expected_version']:
            raise ValueError('G1 世界模型版本不匹配')
        planner = G1RolloutPlanner(model, **config['planner'])
        trained_duration = getattr(getattr(model, 'model', model), 'duration_s', None)
        if trained_duration is not None and abs(trained_duration-config['planner'].get('action_duration_s', .5)) > 1e-9:
            raise ValueError('G1 世界模型与 planner 动作时长不兼容')
        simulator = config.get('simulator', {})
        scene = simulator.get('scene')
        if scene not in (None, 'indoor'):
            raise ValueError('不支持的 G1 仿真场景')
        agent_config = config.get('agent', {})
        options = {key: agent_config[key] for key in ('max_tilt_rad','min_pelvis_height_m') if key in agent_config}
        from wmal.locomotion.agent import G1Agent
        G1Agent(planner, **options)
        if 'feedback' in config:
            from wmal.locomotion.feedback import ResidualFeedback
            options['feedback'] = ResidualFeedback(**config['feedback'])
        result.update(g1_planner=planner, g1_agent_options=options, g1_scene=scene)
        modes.append('g1_navigation')
    except Exception as exc:
        reasons['g1_navigation'] = f'G1 暂不可用：{exc}'
    try:
        if not args.visual_checkpoint or not args.visual_calibration:
            raise ValueError('需要同时提供视觉世界模型 checkpoint 与独立校准文件')
        base = config_root(args.g1_config)
        checkpoint = _asset_path(args.visual_checkpoint, base)
        calibration_path = _asset_path(args.visual_calibration, base)
        if not checkpoint.is_file() or not calibration_path.is_file():
            raise ValueError('视觉 checkpoint 或 calibration 文件缺失')
        model = _load_visual(checkpoint, args.device)
        calibration = _load_calibration(calibration_path)
        arm_horizon = _validate_visual(model, calibration)
        _check_simulation_assets('arm_joint_targets')
        result.update(visual_model=model, visual_calibration=calibration)
        modes.append('arm_joint_targets')
    except Exception as exc:
        arm_horizon = None
        reasons['arm_joint_targets'] = f'机械臂暂不可用：{exc}'
    result['capabilities'] = {'available_modes': modes, 'arm_max_horizon': arm_horizon,
                              'unavailable_reasons': reasons}
    return result


def _without_symlink(path):
    for item in [path, *path.parents]:
        if item.is_symlink():
            raise ValueError('Run directories may not traverse symbolic links')


def prepare_run(runs_root, run_dir, capabilities):
    from wmal.logging.manifest import atomic_json, build_manifest
    root = Path(runs_root).expanduser().absolute()
    _without_symlink(root)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    name = run_dir or ('agent_client_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    run = Path(name).expanduser()
    if not run.is_absolute():
        run = root/run
    _without_symlink(run)
    run = run.resolve()
    if run == root or not run.is_relative_to(root) or any(part.startswith('.') for part in run.relative_to(root).parts):
        raise ValueError('--run-dir must be a visible child directory of --runs-root')
    if run.exists() and (not run.is_dir() or any(run.iterdir())):
        raise ValueError('--run-dir must be new or empty')
    run.mkdir(parents=True, exist_ok=True)
    atomic_json(run/'run_manifest.json', build_manifest('agent_command_client', 0, {},
                parameters={'capabilities':capabilities}))
    event = {'event':'client_ready', 'wall_time_utc':datetime.now(timezone.utc).isoformat(),
             'payload':{'capabilities':capabilities}}
    (run/'events.jsonl').write_text(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    return run


def start_client(args):
    """Construct models and service, leaving all simulator work to its owner."""
    if not math.isfinite(args.shutdown_timeout) or not 0 < args.shutdown_timeout <= 60:
        raise ValueError('--shutdown-timeout must be in (0, 60] seconds')
    from wmal.monitor.runtime import MonitorRuntime
    from wmal.monitor.live_tasks import LiveTaskService
    from wmal.monitor.task_runners import build_task_runners
    from wmal.monitor.video_recorder import TaskVideoRecorder
    assets = preflight(args)
    run = prepare_run(args.runs_root, args.run_dir, assets['capabilities'])
    runtime = MonitorRuntime(run, args.runs_root, args.port, camera='agent_simulation',
                             capabilities=assets['capabilities'])
    task_root = run/'.agent_tasks'
    def g1_session_factory(**kwargs):
        from wmal.envs.g1_session import G1MuJoCoSession
        scene = None
        if assets['g1_scene'] == 'indoor':
            from wmal.envs.indoor_scene import IndoorScene
            scene = IndoorScene()
        return G1MuJoCoSession(viewer=False, realtime=args.realtime, scene=scene, **kwargs)
    try:
        runners = build_task_runners(task_root=task_root, g1_planner=assets['g1_planner'],
            g1_agent_options=assets['g1_agent_options'], g1_session_factory=g1_session_factory,
            visual_model=assets['visual_model'], visual_calibration=assets['visual_calibration'],
            frame_publisher=runtime.publisher, realtime=args.realtime, g1_max_cycles=args.g1_cycles,
            video_recorder_factory=lambda folder: TaskVideoRecorder(folder, fps=args.video_fps))
        service = LiveTaskService(task_root, runners, max_pending=args.max_pending)
        runtime.attach_task_service(service, assets['capabilities'], timeout_s=args.shutdown_timeout)
    except BaseException:
        runtime.__exit__()
        raise
    return runtime


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        runtime = start_client(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    stopped = threading.Event()
    previous = {}
    def stop(*_):
        stopped.set()
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, stop)
    try:
        with runtime:
            print(f'Agent client: {runtime.url}', flush=True)
            for reason in runtime.server.capabilities['unavailable_reasons'].values():
                print(reason, flush=True)
            while not stopped.wait(.25):
                pass
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
