"""Pure validation and parsing for the local, bounded agent command API."""

from dataclasses import dataclass
import json
import math
import re
from types import MappingProxyType
from typing import Mapping


MAX_BODY_BYTES = 16 * 1024
MAX_TEXT_CHARS = 256
MAX_G1_DISTANCE_M = 2.0
MAX_G1_COORDINATE_M = 20.0
MAX_G1_TURN_DEG = 180.0
MAX_WAYPOINTS = 8
MAX_ARM_CYCLES = 100
MAX_ARM_HORIZON = 12

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_G1_TRANSLATION = re.compile(rf"\s*(前进|后退)\s*({_NUMBER})\s*米\s*")
_G1_TURN = re.compile(rf"\s*(左转|右转)\s*({_NUMBER})\s*度\s*")
_G1_COORDINATE = re.compile(
    rf"\s*到达坐标\s+({_NUMBER})\s+({_NUMBER})"
    rf"(?:\s+朝向\s+({_NUMBER})\s*度)?\s*"
)
_ARM_TARGET = re.compile(
    rf"\s*肩关节到\s*({_NUMBER})\s*rad\s*[，,]?\s*"
    rf"肘关节到\s*({_NUMBER})\s*rad\s*"
)


class CommandError(ValueError):
    """A client command failure with an HTTP status and stable error code."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class TaskSpec:
    """Immutable, validated task request passed from HTTP to the task service."""

    mode: str
    task: str
    goal: tuple[float, ...] | None
    waypoints: tuple[tuple[float, ...], ...]
    options: Mapping[str, object]
    relative_intent: Mapping[str, object] | None = None

    def __post_init__(self):
        if self.goal is not None:
            object.__setattr__(self, 'goal', tuple(self.goal))
        object.__setattr__(self, 'waypoints', tuple(tuple(point) for point in self.waypoints))
        object.__setattr__(self, 'options', MappingProxyType(dict(self.options)))
        if self.relative_intent is not None:
            object.__setattr__(self, 'relative_intent', MappingProxyType(dict(self.relative_intent)))


@dataclass(frozen=True)
class CapabilitySnapshot:
    """Pure data snapshot; loading models and files belongs to the runtime."""

    available_modes: frozenset[str]
    arm_max_horizon: int | None = None


def _error(message: str, *, status: int = 400, code: str = 'invalid_command'):
    raise CommandError(status, code, message)


def _number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(f'{field} must be a finite number')
    try:
        result = float(value)
    except OverflowError:
        _error(f'{field} must be a finite number')
    if not math.isfinite(result):
        _error(f'{field} must be a finite number')
    return result


def _exact_keys(value, allowed, field: str, required=()):
    if not isinstance(value, dict):
        _error(f'{field} must be an object')
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown:
        _error(f'{field} contains unsupported keys: {", ".join(sorted(map(str, unknown)))}')
    if missing:
        _error(f'{field} is missing required keys: {", ".join(sorted(missing))}')


def _bounded_pair(value, field: str, low: tuple[float, float], high: tuple[float, float]):
    if not isinstance(value, list) or len(value) != 2:
        _error(f'{field} must contain exactly two numbers')
    pair = tuple(_number(item, f'{field}[{index}]') for index, item in enumerate(value))
    if any(item < lower or item > upper for item, lower, upper in zip(pair, low, high)):
        _error(f'{field} is outside the local joint range')
    return pair


def _parse_g1_target(value, field: str):
    _exact_keys(value, {'x_m', 'y_m', 'yaw_deg'}, field, required=('x_m', 'y_m'))
    x = _number(value['x_m'], f'{field}.x_m')
    y = _number(value['y_m'], f'{field}.y_m')
    if abs(x) > MAX_G1_COORDINATE_M or abs(y) > MAX_G1_COORDINATE_M:
        _error(f'{field} coordinates must be within ±{MAX_G1_COORDINATE_M:g} m')
    yaw = value.get('yaw_deg')
    if yaw is not None:
        yaw = _number(yaw, f'{field}.yaw_deg')
        if abs(yaw) > MAX_G1_TURN_DEG:
            _error(f'{field}.yaw_deg must be within ±{MAX_G1_TURN_DEG:g} degrees')
    return (x, y, yaw)


def _validate_g1_path(waypoints, goal):
    points = list(waypoints) + [goal]
    distance = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(points, points[1:]))
    if distance > MAX_G1_DISTANCE_M:
        _error(f'waypoint path must be no longer than {MAX_G1_DISTANCE_M:g} m')


def _structured_g1(payload):
    _exact_keys(payload, {'mode', 'goal', 'waypoints'}, 'command', required=('mode', 'goal'))
    goal = _parse_g1_target(payload['goal'], 'goal')
    raw_waypoints = payload.get('waypoints', [])
    if not isinstance(raw_waypoints, list) or len(raw_waypoints) > MAX_WAYPOINTS:
        _error(f'waypoints must be a list of at most {MAX_WAYPOINTS} coordinates')
    waypoints = tuple(_parse_g1_target(row, f'waypoints[{index}]')
                      for index, row in enumerate(raw_waypoints))
    _validate_g1_path(waypoints, goal)
    return TaskSpec('g1_navigation', 'world_goal', goal, waypoints, {})


def _structured_arm(payload):
    _exact_keys(payload, {'mode', 'task', 'goal_rad', 'waypoints_rad', 'baseline',
                          'horizon', 'max_cycles', 'error_budget'}, 'command',
                required=('mode', 'task', 'goal_rad'))
    if payload['task'] != 'stack_block':
        _error('task must be stack_block')
    goal = _bounded_pair(payload['goal_rad'], 'goal_rad', (.05, .55), (.65, 1.15))
    raw_waypoints = payload.get('waypoints_rad', [])
    if not isinstance(raw_waypoints, list) or len(raw_waypoints) > MAX_WAYPOINTS:
        _error(f'waypoints_rad must be a list of at most {MAX_WAYPOINTS} joint targets')
    waypoints = tuple(_bounded_pair(row, f'waypoints_rad[{index}]', (.05, .55), (.65, 1.15))
                      for index, row in enumerate(raw_waypoints))

    baseline = payload.get('baseline', 'A3')
    if baseline not in ('A0', 'A1', 'A2', 'A3'):
        _error('baseline must be one of A0, A1, A2, A3')
    horizon = payload.get('horizon', 4)
    if type(horizon) is not int or not 1 <= horizon <= MAX_ARM_HORIZON:
        _error(f'horizon must be an integer in [1, {MAX_ARM_HORIZON}]')
    max_cycles = payload.get('max_cycles', 30)
    if type(max_cycles) is not int or not 1 <= max_cycles <= MAX_ARM_CYCLES:
        _error(f'max_cycles must be an integer in [1, {MAX_ARM_CYCLES}]')
    error_budget = _number(payload.get('error_budget', 1.0), 'error_budget')
    if not 0 < error_budget <= 1:
        _error('error_budget must be in (0, 1]')
    if baseline == 'A0' and waypoints and horizon != 1:
        _error('A0 waypoint tasks require horizon=1')
    options = {'baseline': baseline, 'horizon': horizon,
               'max_cycles': max_cycles, 'error_budget': error_budget}
    return TaskSpec('arm_joint_targets', 'stack_block', goal, waypoints, options)


def _parse_text(text: str) -> TaskSpec:
    if len(text) > MAX_TEXT_CHARS:
        _error(f'text must be at most {MAX_TEXT_CHARS} characters')

    match = _G1_TRANSLATION.fullmatch(text)
    if match:
        amount = _number(float(match.group(2)), 'distance_m')
        if not 0 < amount <= MAX_G1_DISTANCE_M:
            _error(f'distance must be in (0, {MAX_G1_DISTANCE_M:g}] m')
        signed = amount if match.group(1) == '前进' else -amount
        direction = 'forward' if match.group(1) == '前进' else 'backward'
        return TaskSpec('g1_navigation', 'relative_translation', None, (), {},
                        {'kind': 'translation', 'direction': direction,
                         'distance_m': amount})

    match = _G1_TURN.fullmatch(text)
    if match:
        angle = _number(float(match.group(2)), 'turn_deg')
        if not 0 < angle <= MAX_G1_TURN_DEG:
            _error(f'turn angle must be in (0, {MAX_G1_TURN_DEG:g}] degrees')
        direction = 'left' if match.group(1) == '左转' else 'right'
        return TaskSpec('g1_navigation', 'relative_turn', None, (), {},
                        {'kind': 'turn', 'direction': direction, 'angle_deg': angle})

    match = _G1_COORDINATE.fullmatch(text)
    if match:
        x = _number(float(match.group(1)), 'x_m')
        y = _number(float(match.group(2)), 'y_m')
        yaw = _number(float(match.group(3)), 'yaw_deg') if match.group(3) else None
        if abs(x) > MAX_G1_COORDINATE_M or abs(y) > MAX_G1_COORDINATE_M:
            _error(f'coordinates must be within ±{MAX_G1_COORDINATE_M:g} m')
        if yaw is not None and abs(yaw) > MAX_G1_TURN_DEG:
            _error(f'yaw must be within ±{MAX_G1_TURN_DEG:g} degrees')
        return TaskSpec('g1_navigation', 'world_goal', (x, y, yaw), (), {})

    match = _ARM_TARGET.fullmatch(text)
    if match:
        return _structured_arm({'mode': 'arm_joint_targets', 'task': 'stack_block',
                                'goal_rad': [float(match.group(1)), float(match.group(2))]})

    _error('This command is outside the supported local command grammar',
           status=422, code='unsupported_intent')


def _capability_values(capabilities):
    if capabilities is None:
        return None, None
    if isinstance(capabilities, CapabilitySnapshot):
        modes, horizon = capabilities.available_modes, capabilities.arm_max_horizon
    elif isinstance(capabilities, Mapping):
        modes = capabilities.get('available_modes')
        horizon = capabilities.get('arm_max_horizon')
    else:
        _error('capabilities must be a capability snapshot')
    if modes is None:
        _error('capabilities.available_modes is required')
    if isinstance(modes, str) or not isinstance(modes, (set, frozenset, tuple, list)):
        _error('capabilities.available_modes must be a set of mode names')
    if any(mode not in ('g1_navigation', 'arm_joint_targets') for mode in modes):
        _error('capabilities.available_modes contains an unknown mode')
    try:
        modes = frozenset(modes)
    except TypeError:
        _error('capabilities.available_modes must be a set of mode names')
    if horizon is not None and (type(horizon) is not int or not 1 <= horizon <= MAX_ARM_HORIZON):
        _error(f'capabilities.arm_max_horizon must be an integer in [1, {MAX_ARM_HORIZON}]')
    return modes, horizon


def parse_command(payload, capabilities=None) -> TaskSpec:
    """Validate one JSON command and normalize it to an immutable ``TaskSpec``.

    ``capabilities`` is a pure snapshot supplied by the caller. This module
    never reads files, loads a model, or contacts a service.
    """
    if not isinstance(payload, dict):
        _error('command body must be a JSON object')
    try:
        body_size = len(json.dumps(payload, ensure_ascii=False, separators=(',', ':'),
                                   allow_nan=True).encode('utf-8'))
    except (TypeError, ValueError, UnicodeError):
        _error('command body must contain JSON values only')
    if body_size > MAX_BODY_BYTES:
        _error(f'command body must be at most {MAX_BODY_BYTES} bytes')

    if 'text' in payload:
        _exact_keys(payload, {'text'}, 'command', required=('text',))
        if not isinstance(payload['text'], str) or not payload['text'].strip():
            _error('text must be a non-empty string')
        spec = _parse_text(payload['text'])
    else:
        mode = payload.get('mode')
        if mode == 'g1_navigation':
            spec = _structured_g1(payload)
        elif mode == 'arm_joint_targets':
            spec = _structured_arm(payload)
        else:
            _error('command requires a supported mode or text field')

    available_modes, arm_max_horizon = _capability_values(capabilities)
    if available_modes is not None and spec.mode not in available_modes:
        _error(f'mode {spec.mode} is unavailable on this server',
               status=503, code='mode_unavailable')
    if (spec.mode == 'arm_joint_targets' and arm_max_horizon is not None
            and spec.options['horizon'] > arm_max_horizon):
        _error('horizon exceeds the loaded visual model maximum')
    return spec
