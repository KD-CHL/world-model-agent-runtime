"""Single-owner, durable lifecycle for bounded local agent tasks.

Runners are injected callables receiving ``(spec, context)``. This module does
not import or construct simulator sessions; the service's one worker is the
only thread that invokes a runner.
"""

from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid

from .commands import TaskSpec


_TASK_ID = re.compile(r'^[0-9a-f]{32}$')
_COMMAND_ID = re.compile(r'^[A-Za-z0-9_-]{1,128}$')
_TERMINAL = frozenset(('succeeded', 'failed', 'cancelled', 'incomplete'))


def _wall_time_utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


class IdempotencyConflict(ValueError):
    """A command key was already bound to another request fingerprint."""


class ServiceClosed(RuntimeError):
    """Raised when a task is submitted after service shutdown began."""


class TaskCancelled(Exception):
    """Convenience exception runners may raise at a safe stop boundary."""


def _json_value(value):
    """Copy supported values into strict, detached JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict) or hasattr(value, 'items'):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f'not JSON-safe: {type(value).__name__}')


def _encode(value):
    return json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _reject_json_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def _strict_json_loads(value):
    return json.loads(value, parse_constant=_reject_json_constant)


def _finite_timestamp(value):
    return type(value) in (int, float) and math.isfinite(value)


def _atomic_write(path: Path, data: bytes):
    """Replace one file atomically in its task directory."""
    if path.is_symlink():
        raise OSError(f'refusing symlink: {path.name}')
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    fd = os.open(temp, flags, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class TaskSnapshot:
    task_id: str
    command_id: str
    payload_fingerprint: str
    state: str
    spec: object
    created_at: float
    updated_at: float
    progress: object = None
    result: object = None
    error: object = None
    cancel_requested: bool = False
    events: tuple = ()


class TaskContext:
    """Runner-facing progress, evidence, and cooperative cancellation API."""

    def __init__(self, service, task_id, cancel_event):
        self._service = service
        self.task_id = task_id
        self.cancel_event = cancel_event

    def check_cancelled(self):
        if self.cancel_event.is_set():
            raise TaskCancelled('cancellation requested at runner boundary')

    def progress(self, stage, fraction=None, **details):
        payload = {'stage': str(stage)}
        if fraction is not None:
            payload['fraction'] = fraction
        payload.update(details)
        self._service._runner_event(self.task_id, 'progress', payload, progress=payload)

    def record_prediction(self, evidence):
        self._service._runner_event(self.task_id, 'prediction', evidence)

    def record_observation(self, evidence):
        self._service._runner_event(self.task_id, 'observation', evidence)

    def record_evidence(self, kind, evidence):
        self._service._runner_event(self.task_id, str(kind), evidence)


class LiveTaskService:
    """Thread-safe task registry with one FIFO runner-owner worker."""

    def __init__(self, task_root, runners, *, max_pending=8, worker_name='live-task-owner'):
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError('max_pending must be a positive integer')
        self.task_root = Path(task_root).expanduser().absolute()
        if self.task_root.is_symlink():
            raise ValueError('task_root may not be a symlink')
        self.task_root.mkdir(parents=True, exist_ok=True)
        if not self.task_root.is_dir():
            raise ValueError('task_root must be a directory')
        self._root_real = self.task_root.resolve(strict=True)
        self._runners = dict(runners)
        if not all(callable(runner) for runner in self._runners.values()):
            raise TypeError('each runner must be callable')
        self.max_pending = max_pending
        self.worker_name = worker_name
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._tasks = {}
        self._by_command = {}
        self._pending = deque()
        self._cancel_events = {}
        self._active_id = None
        self._closing = False
        self._closed = False
        self._runner_close_errors = []
        self._recover()
        self._worker = threading.Thread(target=self._work, name=worker_name, daemon=True)
        self._worker.start()

    def _task_dir(self, task_id, *, must_exist=False):
        if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
            raise KeyError(task_id)
        path = self.task_root / task_id
        if path.is_symlink():
            raise KeyError(task_id)
        if must_exist and not path.is_dir():
            raise KeyError(task_id)
        try:
            if path.exists() and path.resolve(strict=True).parent != self._root_real:
                raise KeyError(task_id)
        except (OSError, RuntimeError):
            raise KeyError(task_id) from None
        return path

    def _persist(self, record):
        task_dir = self._task_dir(record['task_id'])
        if task_dir.is_symlink():
            raise OSError('task directory may not be a symlink')
        task_dir.mkdir(mode=0o700, exist_ok=True)
        if task_dir.resolve(strict=True).parent != self._root_real:
            raise OSError('task directory escaped task root')
        # Events first: manifest is the commit marker for each updated snapshot.
        event_log = b''.join(_encode(event) + b'\n' for event in record['events'])
        _atomic_write(task_dir / 'events.jsonl', event_log)
        manifest = {key: value for key, value in record.items() if key != 'events'}
        _atomic_write(task_dir / 'manifest.json', _encode(manifest))

    def _snapshot(self, record):
        # Round-tripping gives callers detached data that cannot mutate service state.
        return TaskSnapshot(
            task_id=record['task_id'], command_id=record['command_id'],
            payload_fingerprint=record['payload_fingerprint'], state=record['state'],
            spec=_json_value(record['spec']), created_at=record['created_at'],
            updated_at=record['updated_at'], progress=_json_value(record.get('progress')),
            result=_json_value(record.get('result')), error=_json_value(record.get('error')),
            cancel_requested=record.get('cancel_requested', False),
            events=tuple(_json_value(record.get('events', []))))

    def _event(self, record, kind, payload):
        payload = _json_value(payload)
        if not isinstance(payload, dict):
            payload = {'data': payload}
        timestamp = time.time()
        event = {'event': str(kind), 'timestamp': timestamp,
                 'wall_time_utc': _wall_time_utc(timestamp), 'payload': payload}
        record['events'].append(event)
        record['updated_at'] = event['timestamp']
        self._persist(record)

    def submit(self, spec, command_id, payload_fingerprint):
        if not isinstance(command_id, str) or not _COMMAND_ID.fullmatch(command_id):
            raise ValueError('command_id must be 1-128 ASCII letters, digits, - or _')
        if not isinstance(payload_fingerprint, str) or not payload_fingerprint:
            raise ValueError('payload_fingerprint must be a non-empty string')
        normalized_spec = _json_value(spec)
        mode = normalized_spec.get('mode') if isinstance(normalized_spec, dict) else None
        with self._condition:
            existing_id = self._by_command.get(command_id)
            if existing_id:
                existing = self._tasks[existing_id]
                if existing['payload_fingerprint'] != payload_fingerprint:
                    raise IdempotencyConflict(command_id)
                return self._snapshot(existing)
            if self._closing or self._closed:
                raise ServiceClosed('task service is closing')
            if mode not in self._runners:
                raise ValueError(f'no runner registered for mode {mode!r}')
            if len(self._pending) >= self.max_pending:
                raise OverflowError('task queue is full')
            now = time.time()
            task_id = uuid.uuid4().hex
            record = {'task_id': task_id, 'command_id': command_id,
                      'payload_fingerprint': payload_fingerprint, 'state': 'queued',
                      'spec': normalized_spec, 'created_at': now, 'updated_at': now,
                      'progress': None, 'result': None, 'error': None,
                      'cancel_requested': False, 'events': []}
            queued_event = {'event': 'queued', 'timestamp': now,
                            'wall_time_utc': _wall_time_utc(now), 'payload': {'mode': mode}}
            record['events'].append(queued_event)
            # Persist the complete queued record before exposing its key or enqueueing it.
            self._persist(record)
            self._tasks[task_id] = record
            self._by_command[command_id] = task_id
            self._cancel_events[task_id] = threading.Event()
            self._pending.append(task_id)
            self._condition.notify()
            return self._snapshot(record)

    def get(self, task_id):
        with self._lock:
            self._task_dir(task_id, must_exist=True)
            if task_id not in self._tasks:
                raise KeyError(task_id)
            return self._snapshot(self._tasks[task_id])

    def list(self):
        with self._lock:
            return [self._snapshot(self._tasks[key]) for key in self._tasks]

    def cancel(self, task_id):
        with self._condition:
            self._task_dir(task_id, must_exist=True)
            record = self._tasks.get(task_id)
            if record is None:
                raise KeyError(task_id)
            if record['state'] in _TERMINAL:
                return self._snapshot(record)
            record['cancel_requested'] = True
            self._cancel_events[task_id].set()
            if record['state'] == 'queued':
                try:
                    self._pending.remove(task_id)
                except ValueError:
                    # Worker may have taken it; its locked transition determines state.
                    pass
                else:
                    record['state'] = 'cancelled'
                    self._event(record, 'cancelled', {'reason': 'cancelled_while_queued'})
            else:
                self._event(record, 'cancel_requested', {})
            self._condition.notify_all()
            return self._snapshot(record)

    def _runner_event(self, task_id, kind, payload, *, progress=None):
        with self._lock:
            record = self._tasks[task_id]
            if record['state'] != 'running':
                return
            if progress is not None:
                record['progress'] = _json_value(progress)
            self._event(record, kind, payload)

    def _transition(self, record, state, *, result=None, error=None, event_payload=None):
        record['state'] = state
        record['result'] = _json_value(result) if result is not None else None
        record['error'] = _json_value(error) if error is not None else None
        self._event(record, state, event_payload or {})

    def _work(self):
        while True:
            with self._condition:
                while not self._pending and not self._closing:
                    self._condition.wait()
                if self._closing and not self._pending:
                    finish_shutdown = True
                else:
                    finish_shutdown = False
                if finish_shutdown:
                    task_id = None
                else:
                    task_id = self._pending.popleft()
                    record = self._tasks[task_id]
                    if record['state'] != 'queued':
                        continue
                    if record['cancel_requested']:
                        self._transition(record, 'cancelled',
                                         event_payload={'reason': 'cancelled_while_queued'})
                        continue
                    record['state'] = 'running'
                    self._active_id = task_id
                    self._event(record, 'running',
                                {'owner_thread': threading.current_thread().name})
                    context = TaskContext(self, task_id, self._cancel_events[task_id])
                    runner = self._runners[record['spec']['mode']]
            if finish_shutdown:
                # A close hook may wait for native resources; never hold the registry lock.
                self._close_runner_hooks()
                with self._condition:
                    self._closed = True
                    self._condition.notify_all()
                return
            try:
                spec = self._restore_spec(record['spec'])
                result = runner(spec, context)
                result = _json_value(result)
                if not isinstance(result, dict):
                    raise TypeError('runner must return a JSON object')
                reported_state = result.get('status', 'succeeded')
                if reported_state not in ('succeeded', 'failed', 'cancelled'):
                    raise ValueError(f'runner returned invalid terminal status {reported_state!r}')
                error = None
                if reported_state == 'failed':
                    detail = result.get('error', result.get('reason', 'runner reported failure'))
                    message = detail if isinstance(detail, str) else json.dumps(
                        detail, ensure_ascii=False, allow_nan=False)
                    error = {'type': 'RunnerFailure', 'message': message}
                with self._condition:
                    if reported_state == 'cancelled' or (
                            context.cancel_event.is_set() and reported_state == 'succeeded'):
                        self._transition(record, 'cancelled', result=result,
                                         event_payload={'reason': 'runner_reached_cancel_boundary'})
                    elif reported_state == 'failed':
                        self._transition(record, 'failed', result=result, error=error,
                                         event_payload={'reason': error['message']})
                    else:
                        self._transition(record, 'succeeded', result=result)
            except TaskCancelled as exc:
                with self._condition:
                    self._transition(record, 'cancelled',
                                     event_payload={'reason': str(exc) or 'cancelled'})
            except BaseException as exc:
                with self._condition:
                    self._transition(record, 'failed',
                                     error={'type': type(exc).__name__, 'message': str(exc)})
            finally:
                with self._condition:
                    self._active_id = None
                    self._condition.notify_all()

    @staticmethod
    def _restore_spec(payload):
        """Rebuild the immutable command object for a runner from persisted JSON."""
        required = {'mode', 'task', 'goal', 'waypoints', 'options', 'relative_intent'}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError('persisted task spec does not match TaskSpec schema')
        if not isinstance(payload['mode'], str) or not payload['mode']:
            raise ValueError('task spec mode must be a non-empty string')
        if not isinstance(payload['task'], str) or not payload['task']:
            raise ValueError('task spec task must be a non-empty string')
        goal = payload['goal']
        if goal is not None and (not isinstance(goal, list) or any(
                type(number) not in (int, float) or not math.isfinite(number)
                for number in goal)):
            raise ValueError('task spec goal must be null or a finite numeric list')
        waypoints = payload['waypoints']
        if not isinstance(waypoints, list) or any(
                not isinstance(point, list) or any(
                    type(number) not in (int, float) or not math.isfinite(number)
                    for number in point) for point in waypoints):
            raise ValueError('task spec waypoints must be lists of finite numbers')
        if not isinstance(payload['options'], dict):
            raise ValueError('task spec options must be an object')
        if payload['relative_intent'] is not None and not isinstance(
                payload['relative_intent'], dict):
            raise ValueError('task spec relative_intent must be null or an object')
        return TaskSpec(mode=payload['mode'], task=payload['task'], goal=payload['goal'],
                        waypoints=payload['waypoints'], options=payload['options'],
                        relative_intent=payload['relative_intent'])

    def _close_runner_hooks(self):
        """Close runner-owned session state on the unique owner thread."""
        seen = set()
        for runner in self._runners.values():
            close = getattr(runner, 'close', None)
            if not callable(close) or id(runner) in seen:
                continue
            seen.add(id(runner))
            try:
                close()
            except BaseException as exc:
                self._runner_close_errors.append(
                    {'type': type(exc).__name__, 'message': str(exc)})

    def _recover(self):
        for directory in sorted(self.task_root.iterdir()):
            if directory.is_symlink() or not _TASK_ID.fullmatch(directory.name) or not directory.is_dir():
                continue
            try:
                if directory.resolve(strict=True).parent != self._root_real:
                    continue
                manifest_path = directory / 'manifest.json'
                events_path = directory / 'events.jsonl'
                if manifest_path.is_symlink() or events_path.is_symlink():
                    continue
                record = _strict_json_loads(manifest_path.read_text(encoding='utf-8'))
                manifest_fields = {'task_id', 'command_id', 'payload_fingerprint', 'state',
                                   'spec', 'created_at', 'updated_at', 'progress', 'result',
                                   'error', 'cancel_requested'}
                if not isinstance(record, dict) or set(record) != manifest_fields:
                    continue
                if record.get('task_id') != directory.name:
                    continue
                command_id = record.get('command_id')
                fingerprint = record.get('payload_fingerprint')
                if not isinstance(command_id, str) or not _COMMAND_ID.fullmatch(command_id):
                    continue
                if not isinstance(fingerprint, str) or not fingerprint:
                    continue
                if record.get('state') not in ('queued', 'running', *_TERMINAL):
                    continue
                if (not _finite_timestamp(record.get('created_at'))
                        or not _finite_timestamp(record.get('updated_at'))
                        or type(record.get('cancel_requested')) is not bool):
                    continue
                if record.get('error') is not None and not isinstance(record.get('error'), dict):
                    continue
                try:
                    self._restore_spec(record.get('spec'))
                    _json_value(record.get('progress'))
                    _json_value(record.get('result'))
                    _json_value(record.get('error'))
                except (TypeError, ValueError, RecursionError):
                    continue
                event_bytes = events_path.read_bytes()
                events = []
                valid_events = True
                for line in event_bytes.splitlines():
                    if not line.strip():
                        continue
                    event = _strict_json_loads(line.decode('utf-8'))
                    if (not isinstance(event, dict) or not isinstance(event.get('event'), str)
                            or not _finite_timestamp(event.get('timestamp'))
                            or not isinstance(event.get('payload'), dict)):
                        valid_events = False
                        break
                    events.append(event)
                if not valid_events:
                    continue
                record['events'] = events
                if record.get('state') == 'running':
                    record['state'] = 'incomplete'
                    record['error'] = {'type': 'ProcessRestart',
                                       'message': 'process ended while task was running'}
                    record['updated_at'] = time.time()
                    self._event(record, 'incomplete', {'reason': 'process_restart'})
                if command_id in self._by_command:
                    continue
                self._tasks[record['task_id']] = record
                self._by_command[command_id] = record['task_id']
                self._cancel_events[record['task_id']] = threading.Event()
                if record['state'] == 'queued':
                    mode = record['spec']['mode']
                    if mode not in self._runners:
                        record['state'] = 'incomplete'
                        record['error'] = {'type': 'MissingRunner',
                                           'message': f'no runner registered for mode {mode!r}'}
                        self._event(record, 'incomplete', {'reason': 'runner_unavailable',
                                                          'mode': mode})
                    elif len(self._pending) < self.max_pending:
                        self._pending.append(record['task_id'])
                    else:
                        record['state'] = 'incomplete'
                        record['error'] = {'type': 'RecoveryQueueFull',
                                           'message': 'recovered queue exceeds configured capacity'}
                        self._event(record, 'incomplete', {'reason': 'recovery_queue_full'})
            except (OSError, ValueError, TypeError, KeyError, RecursionError,
                    UnicodeDecodeError, json.JSONDecodeError):
                # Corrupt or incomplete evidence is left in place for inspection.
                continue

    def close(self, timeout_s=5.0):
        """Stop accepting work, cancel queued tasks, and join the owner boundedly.

        Returns False if an active runner is still executing after the timeout;
        its persisted state remains ``running`` until the owner actually returns.
        """
        timeout = max(0.0, float(timeout_s))
        with self._condition:
            if not self._closing:
                self._closing = True
                for task_id in list(self._pending):
                    self._pending.remove(task_id)
                    record = self._tasks[task_id]
                    record['cancel_requested'] = True
                    self._cancel_events[task_id].set()
                    self._transition(record, 'cancelled',
                                     event_payload={'reason': 'service_closing'})
                if self._active_id is not None:
                    self._cancel_events[self._active_id].set()
                    active = self._tasks[self._active_id]
                    active['cancel_requested'] = True
                    self._event(active, 'cancel_requested', {'reason': 'service_closing'})
                self._condition.notify_all()
        self._worker.join(timeout)
        return not self._worker.is_alive()

