"""LeRobot metadata validation must recognize channel-first image features."""
import json
from pathlib import Path
import tempfile
import unittest
import csv

from wmal.datasets.lerobot import inspect_lerobot_v21


class LeRobotManifestTests(unittest.TestCase):
    def test_channel_first_camera_shape_is_recorded_as_width_height(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "meta").mkdir()
            (root / "data/chunk-000").mkdir(parents=True)
            (root / "videos/chunk-000/observation.images.cam_left_high").mkdir(parents=True)
            features = {
                "observation.state": {"dtype": "float32", "shape": [2], "names": [["j0", "j1"]]},
                "action": {"dtype": "float32", "shape": [2], "names": [["j0", "j1"]]},
                "observation.images.cam_left_high": {
                    "dtype": "video", "shape": [3, 480, 640],
                    "names": ["channels", "height", "width"],
                },
            }
            info = {"total_episodes": 5, "fps": 30, "features": features,
                    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
            (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
            (root / "meta/tasks.jsonl").write_text('{"task":"pick and place"}\n', encoding="utf-8")
            for episode in range(5):
                (root / f"data/chunk-000/episode_{episode:06d}.parquet").write_bytes(b"parquet")
                (root / f"videos/chunk-000/observation.images.cam_left_high/episode_{episode:06d}.mp4").write_bytes(b"video")

            manifest = inspect_lerobot_v21(
                root, dataset_id="fixture/dataset", revision="a" * 40,
                license_id="apache-2.0", camera_key="observation.images.cam_left_high")

            self.assertEqual(manifest["camera"], {
                "key": "observation.images.cam_left_high", "width": 640,
                "height": 480, "channels": 3})


if __name__ == "__main__":
    unittest.main()
