"""Run public collection and calibration commands; provenance survives the workflow."""
import importlib.util
import json
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT=Path(__file__).resolve().parents[1]


class ControllerWorkflowTests(unittest.TestCase):

    def test_task_cli_exposes_explicit_controller_selection(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/g1_task_agent.py'),'--help'],
                               capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--controller',result.stdout)
