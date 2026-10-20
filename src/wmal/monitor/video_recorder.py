"""Bounded task-video capture without simulator references or producer I/O.

Call ``submit`` on the simulation owner, starting before its first action.
Timestamps use simulation seconds. Call ``finish`` after stopping physics; only
that method waits for encoding. Images are HxWx3 uint8 RGB snapshots. A reserved
initial slot and final slot cannot be displaced by the ordinary bounded queue.
"""
from collections import deque
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from threading import Condition, Event, Thread
import time

import numpy as np


@dataclass(frozen=True)
class VideoResult:
    status: str
    path: Path | None
    error: str | None
    submitted_frames: int
    accepted_frames: int
    dropped_frames: int
    encoded_frames: int
    duration_s: float | None
    start_sim_time_s: float | None
    end_sim_time_s: float | None
    dropped_time_ranges: tuple[tuple[float, float], ...] = ()
    drop_ranges_truncated: bool = False


def _timestamp(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Simulation timestamp must be finite and nonnegative')
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError('Simulation timestamp must be finite and nonnegative') from None
    if not math.isfinite(result) or result < 0:
        raise ValueError('Simulation timestamp must be finite and nonnegative')
    return result


def _image(rgb):
    array = np.asarray(rgb)
    if (array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3
            or not 1 <= array.shape[0] <= 2048 or not 1 <= array.shape[1] <= 2048):
        raise ValueError('Video frame must be HxWx3 uint8 RGB, at most 2048x2048')
    return array.copy(order='C')


class TaskVideoRecorder:
    """One task's asynchronous ffmpeg recorder with bounded in-memory capture.

    ``submit`` never waits for an encoder, disk, or queue capacity. Invalid
    frames/timestamps raise ValueError; ordinary queue overload returns False.
    Gaps hold the last accepted observation, with loss reported in VideoResult.
    Existing final/partial files are never overwritten by a new recorder.
    """

    def __init__(self, task_dir, ffmpeg_path='ffmpeg', *, ffprobe_path='ffprobe',
                 fps=15, queue_size=8, timeout_s=30):
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 0 < fps <= 120:
            raise ValueError('fps must be finite and in (0, 120]')
        if type(queue_size) is not int or not 1 <= queue_size <= 4096:
            raise ValueError('queue_size must be an integer in [1, 4096]')
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or not 0 < timeout_s <= 300):
            raise ValueError('timeout_s must be finite and in (0, 300]')
        self.task_dir = Path(task_dir)
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._final_path = self.task_dir / 'video.mp4'
        self._partial_path = self.task_dir / 'video.mp4.part'
        if self._final_path.exists():
            raise FileExistsError(self._final_path)
        # Exclusive reservation also prevents two recorders owning a task path.
        self._partial_path.open('xb').close()
        self.ffmpeg_path, self.ffprobe_path = os.fspath(ffmpeg_path), os.fspath(ffprobe_path)
        self.fps, self.queue_size, self.timeout_s = float(fps), queue_size, float(timeout_s)
        self._condition = Condition()
        self._done, self._abort = Event(), Event()
        self._queue = deque()
        self._initial = self._final = None
        self._shape = self._last_time = self._start_time = self._end_time = None
        self._finished = self._failed = False
        self._abort_reason = None
        self._submitted = self._accepted = self._dropped = self._encoded = 0
        self._drop_ranges, self._previous_dropped, self._ranges_truncated = [], False, False
        self._process = self._result = self._deadline = None
        self._thread = Thread(target=self._run, name='task-video-encoder', daemon=True)
        self._thread.start()

    def submit(self, rgb, sim_time_s):
        timestamp, frame = _timestamp(sim_time_s), _image(rgb)
        with self._condition:
            if self._finished:
                raise RuntimeError('Video recorder is already finishing or finished')
            if self._shape is not None and frame.shape != self._shape:
                raise ValueError('All video frames must have the initial frame shape')
            if self._last_time is not None and timestamp <= self._last_time:
                raise ValueError('Submitted simulation timestamps must strictly increase')
            if self._failed:
                return False
            self._last_time = timestamp
            self._submitted += 1
            if self._initial is None:
                self._initial = (timestamp, frame)
                self._shape, self._start_time = frame.shape, timestamp
            elif len(self._queue) >= self.queue_size:
                self._dropped += 1
                if self._previous_dropped:
                    self._drop_ranges[-1] = (self._drop_ranges[-1][0], timestamp)
                elif len(self._drop_ranges) < 64:
                    self._drop_ranges.append((timestamp, timestamp))
                else:
                    # Bound diagnostic memory too. The last range summarizes
                    # all later losses; the flag makes its coarser scope clear.
                    self._drop_ranges[-1] = (self._drop_ranges[-1][0], timestamp)
                    self._ranges_truncated = True
                self._previous_dropped = True
                return False
            else:
                self._queue.append((timestamp, frame))
            self._accepted += 1
            self._previous_dropped = False
            self._condition.notify()
            return True

    def finish(self, end_sim_time_s, final_rgb):
        """Stop capture, preserve final observation, and wait for a verified MP4.

        The end may equal the latest submitted timestamp. This is idempotent;
        the first finish call owns the endpoint and later calls return its result.
        """
        with self._condition:
            if not self._finished:
                timestamp, frame = _timestamp(end_sim_time_s), _image(final_rgb)
                if self._shape is not None and frame.shape != self._shape:
                    raise ValueError('Final frame must have the initial frame shape')
                if self._last_time is not None and timestamp < self._last_time:
                    raise ValueError('End time cannot precede any submitted timestamp')
                self._end_time, self._final = timestamp, frame
                self._deadline = time.monotonic() + self.timeout_s
                self._finished = True
                self._condition.notify()
        if not self._done.wait(self.timeout_s):
            self.abort('Video finalization timed out')
        with self._condition:
            # An early encoder startup failure may have completed before finish.
            if self._result.end_sim_time_s is None:
                self._result = replace(self._result, end_sim_time_s=self._end_time)
            return self._result

    def abort(self, reason):
        """Discard unfinished capture and stop its worker/process boundedly.

        Returns a failed VideoResult with the reason. Repeated cleanup is safe;
        an already verified and published successful video is left untouched.
        This also interrupts a probe/decode running concurrently with finish.
        """
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1024:
            raise ValueError('Abort reason must be a nonempty string of at most 1024 characters')
        with self._condition:
            if self._result is not None and (self._done.is_set() or self._result.status == 'ready'):
                return self._result
            if self._abort_reason is None:
                self._abort_reason = reason
            self._abort.set()
            self._finished = True
            if self._end_time is None:
                self._end_time = self._last_time
            self._queue.clear()
            process = self._process
            self._condition.notify_all()
        self._kill(process)
        self._thread.join(timeout=min(2.0, self.timeout_s))
        if self._thread.is_alive():
            # Cover the race where a subprocess was being spawned at abort.
            with self._condition:
                process = self._process
            self._kill(process)
            self._thread.join(timeout=1)
        with self._condition:
            self._failed = True
            self._result = self._make_result('failed', f'Aborted: {self._abort_reason}')
        self._partial_path.unlink(missing_ok=True)
        return self._result

    @staticmethod
    def _kill(process):
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass

    def _start_process(self, command, **kwargs):
        process = subprocess.Popen(command, **kwargs)
        with self._condition:
            self._process = process
            aborted = self._abort.is_set()
        if aborted:
            self._kill(process)
            process.communicate(timeout=1)
            raise RuntimeError('Video capture aborted while starting subprocess')
        return process

    def _checked_output(self, command, *, capture_stdout=True):
        process = self._start_process(command, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
                                      stderr=subprocess.PIPE)
        try:
            output, errors = process.communicate(timeout=self._remaining())
        except BaseException:
            self._kill(process)
            process.communicate(timeout=1)
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, output, errors)
        self._remaining()
        return output

    def _make_result(self, status, error=None, duration=None):
        return VideoResult(status, self._final_path if status == 'ready' else None,
                           error, self._submitted, self._accepted, self._dropped,
                           self._encoded, duration, self._start_time, self._end_time,
                           tuple(self._drop_ranges), self._ranges_truncated)

    def _remaining(self):
        if self._abort.is_set():
            raise RuntimeError(f'Aborted: {self._abort_reason or "Video finalization timed out"}')
        if self._deadline is None:
            return self.timeout_s
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Video finalization timed out')
        return remaining

    def _run(self):
        process = None
        try:
            with self._condition:
                self._condition.wait_for(lambda: self._initial is not None or self._finished)
                self._remaining()
                if self._initial is None:
                    raise ValueError('No initial task frame was submitted')
                start, initial = self._initial
            height, width = initial.shape[:2]
            with tempfile.TemporaryFile() as errors:
                process = self._start_process([
                    self.ffmpeg_path, '-hide_banner', '-loglevel', 'error', '-y',
                    '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                    '-framerate', str(self.fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                    '-preset', 'ultrafast', '-crf', '18', '-pix_fmt', 'yuv420p',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-movflags', '+faststart',
                    '-f', 'mp4', str(self._partial_path)], stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=errors)
                # Keep one tail frame uncommitted, allowing finish to place the
                # true endpoint in the final CFR time cell without stretching
                # duration by two frame periods.
                process.stdin.write(initial.tobytes())
                self._encoded += 1
                pending, previous, next_index = None, initial, 1

                def append(frame):
                    nonlocal pending
                    if self._abort.is_set():
                        self._remaining()
                    if pending is not None:
                        process.stdin.write(pending.tobytes())
                        self._encoded += 1
                    pending = frame

                while True:
                    with self._condition:
                        self._condition.wait_for(lambda: self._queue or self._finished)
                        self._remaining()
                        if self._queue:
                            timestamp, frame = self._queue.popleft()
                        else:
                            end, final = self._end_time, self._final
                            break
                    target = math.ceil((timestamp - start) * self.fps - 1e-9)
                    while next_index < target:
                        append(previous)
                        next_index += 1
                    previous = frame
                last_index = math.floor((end - start) * self.fps + 1e-9)
                if next_index - 1 == last_index:
                    if last_index == 0 and not np.array_equal(initial, final):
                        raise ValueError('Task is too short at this fps to preserve both endpoint frames')
                    if last_index != 0:
                        pending = final
                else:
                    while next_index < last_index:
                        append(previous)
                        next_index += 1
                    append(final)
                if pending is not None:
                    process.stdin.write(pending.tobytes())
                    self._encoded += 1
                process.stdin.close()
                if process.wait(timeout=self._remaining()) != 0:
                    errors.seek(0, os.SEEK_END)
                    errors.seek(max(0, errors.tell() - 4096))
                    raise RuntimeError('ffmpeg encoding failed: ' + errors.read().decode('utf-8', 'replace'))
            duration = self._verify(width, height, end - start)
            with self._condition:
                self._remaining()
                os.replace(self._partial_path, self._final_path)
                self._result = self._make_result('ready', duration=duration)
        except Exception as exc:
            with self._condition:
                self._failed = True
                error = (f'Aborted: {self._abort_reason}' if self._abort_reason is not None
                         else f'{type(exc).__name__}: {exc}')
                self._result = self._make_result('failed', error)
        finally:
            if process is not None:
                self._kill(process)
                if process.stdin is not None and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                process.wait()
            if self._result is None or self._result.status != 'ready':
                self._partial_path.unlink(missing_ok=True)
            with self._condition:
                self._queue.clear()
                self._initial = self._final = None
            self._done.set()

    def _verify(self, width, height, expected_duration):
        probe = self._checked_output([
            self.ffprobe_path, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=width,height,nb_read_frames,duration:format=duration',
            '-of', 'json', str(self._partial_path)])
        data = json.loads(probe)
        stream = data['streams'][0]
        if (int(stream['width']) != width + width % 2
                or int(stream['height']) != height + height % 2
                or int(stream['nb_read_frames']) != self._encoded):
            raise ValueError('Encoded video shape/frame count differs from submitted timeline')
        duration = float(stream.get('duration', data.get('format', {}).get('duration')))
        if (not math.isfinite(duration) or duration <= 0
                or duration < expected_duration - 1e-5
                or duration - expected_duration > 1 / self.fps + 1e-5):
            raise ValueError('Encoded video duration differs from simulation duration')
        self._checked_output([self.ffmpeg_path, '-hide_banner', '-loglevel', 'error', '-xerror',
                              '-i', str(self._partial_path), '-f', 'null', '-'],
                             capture_stdout=False)
        return duration
