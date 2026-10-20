"""Task-video contract: real MP4 timelines, backpressure, and failure isolation."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest

import numpy as np

from wmal.monitor.video_recorder import TaskVideoRecorder


FFMPEG = shutil.which('ffmpeg')
FFPROBE = shutil.which('ffprobe')


def image(color, size=32):
    return np.full((size, size, 3), color, dtype=np.uint8)


class TaskVideoRecorderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def recorder(self, name='task', **kwargs):
        return TaskVideoRecorder(self.root / name, ffmpeg_path=FFMPEG or 'ffmpeg',
                                 ffprobe_path=FFPROBE or 'ffprobe', fps=10, **kwargs)

    def decoded(self, path, size=32):
        process = subprocess.run(
            [FFMPEG, '-v', 'error', '-i', str(path), '-f', 'rawvideo',
             '-pix_fmt', 'rgb24', 'pipe:1'], capture_output=True, check=True, timeout=10)
        return np.frombuffer(process.stdout, dtype=np.uint8).reshape(-1, size, size, 3)

    @unittest.skipUnless(FFMPEG and FFPROBE, 'ffmpeg and ffprobe are required')
    def test_real_clip_preserves_endpoints_and_simulation_duration(self):
        recorder = self.recorder()
        initial = image((240, 0, 0))
        self.assertTrue(recorder.submit(initial, 5.0))
        initial[:] = (0, 0, 0)  # Owner may immediately reuse its render buffer.
        self.assertTrue(recorder.submit(image((0, 240, 0)), 5.2))
        self.assertFalse((self.root / 'task' / 'video.mp4').exists())
        result = recorder.finish(5.45, image((0, 0, 240)))
        self.assertEqual(result.status, 'ready', result.error)
        self.assertEqual(result.encoded_frames, 5)
        self.assertAlmostEqual(result.duration_s, .5, places=3)
        self.assertEqual(result.dropped_frames, 0)
        self.assertEqual(result.start_sim_time_s, 5.0)
        self.assertEqual(result.end_sim_time_s, 5.45)
        frames = self.decoded(result.path)
        self.assertGreater(frames[0, :, :, 0].mean(), 220)
        self.assertGreater(frames[-1, :, :, 2].mean(), 220)
        self.assertFalse((self.root / 'task' / 'video.mp4.part').exists())
        self.assertEqual(recorder.finish(5.45, image((0, 0, 240))), result)
        self.assertEqual(recorder.abort('late cleanup after success'), result)
        self.assertTrue(result.path.is_file())

    def test_abort_wakes_idle_worker_is_idempotent_and_cleans_partial(self):
        recorder = self.recorder()
        result = recorder.abort('simulation did not start')
        self.assertEqual(result.status, 'failed')
        self.assertIn('simulation did not start', result.error)
        self.assertFalse(recorder._thread.is_alive())
        self.assertEqual(recorder.abort('repeated cleanup'), result)
        self.assertEqual(recorder.finish(.2, image(0)), result)
        self.assertFalse((self.root / 'task' / 'video.mp4').exists())
        self.assertFalse((self.root / 'task' / 'video.mp4.part').exists())

    def test_abort_kills_active_encoder_and_joins_worker_boundedly(self):
        started = self.root / 'started'
        encoder = self.root / 'stalled-ffmpeg'
        encoder.write_text(f'#!/bin/sh\ntouch "{started}"\nexec sleep 10\n')
        encoder.chmod(0o700)
        recorder = TaskVideoRecorder(self.root / 'task', ffmpeg_path=encoder, fps=10)
        recorder.submit(image(0, 256), 0.0)
        deadline = time.monotonic() + 3
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(started.exists())
        started_abort = time.monotonic()
        result = recorder.abort('renderer failed')
        self.assertLess(time.monotonic() - started_abort, 2.5)
        self.assertEqual(result.status, 'failed')
        self.assertIn('renderer failed', result.error)
        self.assertFalse(recorder._thread.is_alive())
        self.assertIsNotNone(recorder._process.poll())
        self.assertFalse((self.root / 'task' / 'video.mp4').exists())
        self.assertFalse((self.root / 'task' / 'video.mp4.part').exists())

    @unittest.skipUnless(FFMPEG, 'ffmpeg is required')
    def test_abort_during_probe_stops_probe_and_never_publishes(self):
        started = self.root / 'probe-started'
        probe = self.root / 'stalled-probe'
        probe.write_text(f'#!/bin/sh\ntouch "{started}"\nexec sleep 10\n')
        probe.chmod(0o700)
        recorder = TaskVideoRecorder(self.root / 'task', ffmpeg_path=FFMPEG,
                                     ffprobe_path=probe, fps=10)
        recorder.submit(image(0), 0.0)
        results = []
        finishing = threading.Thread(
            target=lambda: results.append(recorder.finish(.2, image(80))))
        finishing.start()
        # ffmpeg must finish encoding before this deliberately stalled probe
        # starts. Allow slower CI hosts to reach that boundary, but stop waiting
        # immediately if finalization fails before launching the probe.
        deadline = time.monotonic() + 10
        while (not started.exists() and finishing.is_alive()
               and time.monotonic() < deadline):
            time.sleep(.01)
        if not started.exists():
            recorder.abort('probe did not start')
            finishing.join(2)
            self.fail('ffprobe did not start before finalization ended or timed out')
        self.assertTrue(started.exists())
        result = recorder.abort('task owner stopped')
        finishing.join(2)
        self.assertFalse(finishing.is_alive())
        self.assertEqual(result.status, 'failed')
        self.assertIn('task owner stopped', result.error)
        self.assertFalse(recorder._thread.is_alive())
        self.assertIsNotNone(recorder._process.poll())
        self.assertEqual(results[0], result)
        self.assertFalse((self.root / 'task' / 'video.mp4').exists())
        self.assertFalse((self.root / 'task' / 'video.mp4.part').exists())

    @unittest.skipUnless(FFMPEG and FFPROBE, 'ffmpeg and ffprobe are required')
    def test_rejects_nonmonotonic_nonfinite_and_malformed_frames(self):
        recorder = self.recorder()
        recorder.submit(image(0), 1.0)
        for timestamp in (1.0, .9, float('nan'), float('inf'), True, 10**400):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                recorder.submit(image(0), timestamp)
        for bad in (np.zeros((32, 32, 3)), np.zeros((32, 32), dtype=np.uint8),
                    np.zeros((16, 16, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                recorder.submit(bad, 1.1)
        with self.assertRaises(ValueError):
            recorder.finish(.9, image(0))
        result = recorder.finish(1.2, image(0))
        self.assertEqual(result.status, 'ready', result.error)
        with self.assertRaises(RuntimeError):
            recorder.submit(image(0), 1.3)

    @unittest.skipUnless(FFMPEG and FFPROBE, 'ffmpeg and ffprobe are required')
    def test_zero_motion_still_has_one_decodable_frame(self):
        recorder = self.recorder()
        recorder.submit(image(80), 2.0)
        result = recorder.finish(2.0, image(80))
        self.assertEqual(result.status, 'ready', result.error)
        self.assertEqual(result.encoded_frames, 1)
        self.assertEqual(len(self.decoded(result.path)), 1)

    @unittest.skipUnless(FFMPEG and FFPROBE, 'ffmpeg and ffprobe are required')
    def test_full_queue_drops_without_waiting_and_preserves_timeline(self):
        # Real encoder is gated before reading stdin: its pipe fills on the
        # initial 256x256 frame, making backpressure deterministic.
        gate, started = self.root / 'gate', self.root / 'started'
        wrapper = self.root / 'gated-ffmpeg'
        wrapper.write_text('#!/bin/sh\n'
                           f'touch "{started}"\n'
                           f'while [ ! -f "{gate}" ]; do sleep 0.01; done\n'
                           f'exec "{FFMPEG}" "$@"\n')
        wrapper.chmod(0o700)
        recorder = TaskVideoRecorder(self.root / 'task', ffmpeg_path=wrapper,
                                     ffprobe_path=FFPROBE, fps=10, queue_size=1)
        recorder.submit(image((240, 0, 0), 256), 0.0)
        deadline = time.monotonic() + 3
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        try:
            self.assertTrue(started.exists())
            self.assertTrue(recorder.submit(image((0, 240, 0), 256), .1))
            started_submit = time.monotonic()
            self.assertFalse(recorder.submit(image((240, 240, 0), 256), .2))
            self.assertLess(time.monotonic() - started_submit, .1)
        finally:
            gate.touch()
        result = recorder.finish(.5, image((0, 0, 240), 256))
        self.assertEqual(result.status, 'ready', result.error)
        self.assertEqual(result.dropped_frames, 1)
        self.assertEqual(result.encoded_frames, 6)
        self.assertAlmostEqual(result.duration_s, .6, places=3)
        frames = self.decoded(result.path, 256)
        self.assertGreater(frames[0, :, :, 0].mean(), 220)
        self.assertGreater(frames[3, :, :, 1].mean(), 220)
        self.assertGreater(frames[-1, :, :, 2].mean(), 220)

    def test_missing_encoder_cleans_partial_and_never_publishes_video(self):
        recorder = TaskVideoRecorder(self.root / 'task', ffmpeg_path=self.root / 'missing',
                                     ffprobe_path=FFPROBE or 'ffprobe', fps=10)
        recorder.submit(image(0), 0.0)
        result = recorder.finish(.2, image(80))
        self.assertEqual(result.status, 'failed')
        self.assertTrue(result.error)
        self.assertIsNone(result.path)
        self.assertFalse((self.root / 'task' / 'video.mp4').exists())
        self.assertFalse((self.root / 'task' / 'video.mp4.part').exists())

    def test_encoder_exit_failure_cleans_partial(self):
        bad_encoder = self.root / 'bad-ffmpeg'
        bad_encoder.write_text('#!/bin/sh\nprintf "bad encoder" >&2\nexit 4\n')
        bad_encoder.chmod(0o700)
        recorder = TaskVideoRecorder(self.root / 'task', ffmpeg_path=bad_encoder,
                                     ffprobe_path=FFPROBE or 'ffprobe', fps=10)
        recorder.submit(image(0), 0.0)
        result = recorder.finish(.2, image(80))
        self.assertEqual(result.status, 'failed')
        self.assertFalse((self.root / 'task' / 'video.mp4').exists())
        self.assertFalse((self.root / 'task' / 'video.mp4.part').exists())

    @unittest.skipUnless(FFMPEG, 'ffmpeg is required')
    def test_failed_probe_does_not_publish_encoded_file(self):
        recorder = TaskVideoRecorder(self.root / 'task', ffmpeg_path=FFMPEG,
                                     ffprobe_path=self.root / 'missing-probe', fps=10)
        recorder.submit(image(0), 0.0)
        result = recorder.finish(.2, image(80))
        self.assertEqual(result.status, 'failed')
        self.assertFalse((self.root / 'task' / 'video.mp4').exists())
        self.assertFalse((self.root / 'task' / 'video.mp4.part').exists())

    def test_finishing_without_initial_frame_fails_without_publishing(self):
        recorder = self.recorder()
        result = recorder.finish(.2, image(80))
        self.assertEqual(result.status, 'failed')
        self.assertIn('initial', result.error)
        self.assertFalse((self.root / 'task' / 'video.mp4').exists())
        self.assertFalse((self.root / 'task' / 'video.mp4.part').exists())

    def test_stalled_encoder_has_bounded_finalization_and_no_partial(self):
        encoder = self.root / 'stalled-ffmpeg'
        encoder.write_text('#!/bin/sh\nexec sleep 10\n')
        encoder.chmod(0o700)
        recorder = TaskVideoRecorder(self.root / 'task', ffmpeg_path=encoder,
                                     fps=10, timeout_s=.2)
        recorder.submit(image(0), 0.0)
        started = time.monotonic()
        result = recorder.finish(.2, image(80))
        self.assertLess(time.monotonic() - started, 2.5)
        self.assertEqual(result.status, 'failed')
        self.assertFalse((self.root / 'task' / 'video.mp4').exists())
        self.assertFalse((self.root / 'task' / 'video.mp4.part').exists())

    def test_rejects_invalid_limits_and_existing_task_video(self):
        for kwargs in ({'fps': True}, {'fps': float('nan')}, {'queue_size': 0},
                       {'queue_size': True}, {'timeout_s': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TaskVideoRecorder(self.root / 'invalid', **kwargs)
        task = self.root / 'existing'
        task.mkdir()
        (task / 'video.mp4').write_bytes(b'original')
        with self.assertRaises(FileExistsError):
            TaskVideoRecorder(task)
        self.assertEqual((task / 'video.mp4').read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main()
