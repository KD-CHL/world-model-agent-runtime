"""Controller transfer must never silently reuse pre-transfer calibration."""
import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
from wmal.models.horizon_calibration import HorizonCalibration


class ControllerCalibrationTests(unittest.TestCase):
    def calibration(self):
        return HorizonCalibration('v1', 'independent-set', .2, (1.,), (1.,), .05,
                                  ('c0', 'c1', 'c2', 'c3'), {'action_schema': 'example'})

    def validate(self, calibration, profile):
        method = getattr(calibration, 'validate_controller', None)
        self.assertIsNotNone(method, 'Calibration must validate its fixed controller binding')
        method(profile)

    def test_legacy_calibration_cannot_authorize_new_controller(self):
        calibration = self.calibration()
        self.validate(calibration, {'profile': 'g1_joint_pd_v1'})
        with self.assertRaises(ValueError):
            self.validate(calibration, {'profile': 'g1_joint_pd_gravity_v2'})

    def test_new_calibration_matches_entire_controller_descriptor(self):
        calibration = self.calibration()
        self.assertTrue(hasattr(calibration, 'controller_profile'), 'Missing controller provenance')
        profile = {'profile': 'g1_joint_pd_gravity_v2', 'parameters_sha256': 'fixed-gains'}
        calibration = replace(calibration, controller_profile=profile)
        self.validate(calibration, dict(profile))
        with self.assertRaises(ValueError):
            self.validate(calibration, {**profile, 'parameters_sha256': 'different-gains'})
        with self.assertRaises(ValueError):
            self.validate(calibration, {'profile': 'g1_joint_pd_v1'})


    def test_local_calibration_rejects_out_of_domain_context_or_commands(self):
        import numpy as np
        calibration=self.calibration()
        self.assertTrue(hasattr(calibration,'prediction_scope'),'Missing deployment-domain binding')
        scope={'command_lower':[.21,.73],'command_upper':[.49,1.01],
               'measured_margin':.15,'max_action_delta':.06}
        calibration=replace(calibration,prediction_scope=scope)
        self.assertTrue(calibration.allows_scope(np.array([[.35,.87,.35,.87]]),np.array([[.04,.04]])))
        self.assertFalse(calibration.allows_scope(np.array([[.35,.87,.5,.87]]),np.zeros((1,2))))
        self.assertFalse(calibration.allows_scope(np.array([[.35,.87,.48,.87]]),np.array([[.04,0.]])))
        self.assertFalse(calibration.allows_scope(np.array([[.35,.87,.35,.87]]),np.array([[.07,0.]])))
        self.assertFalse(calibration.allows_scope(np.array([[.9,.87,.35,.87]]),np.zeros((1,2))))
