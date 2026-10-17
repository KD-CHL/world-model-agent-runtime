from dataclasses import replace
import numpy as np
import unittest

from wmal.locomotion.contracts import G1State, G1VelocityAction, G1Goal
from wmal.locomotion.learned import LearnedG1Dynamics
from wmal.locomotion.feedback import ResidualFeedback
from wmal.locomotion.planner import G1RolloutPlanner


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.state = G1State('episode', 0, 0., 0., 0., .8, 0., 0., 0., 0., 0., 0., 0., .8)
        self.rows = []
        for i in range(60):
            action = G1VelocityAction((i % 10 - 4) * .04, (i % 3 - 1) * .05, 0., .5)
            after = replace(self.state, step_id=1, sim_time_s=.5,
                            x=action.vx * .4, y=action.vy * .4)
            self.rows.append((self.state, action, after))


    def test_feedback_changes_candidate_bounds_and_rejects_stale(self):
        weights = np.zeros((2, 10, 6))
        weights[:, 7, 0] = .8
        weights[:, 8, 1] = .8
        weights[:, 0, 5] = .8
        model = LearnedG1Dynamics(weights, .5, 'feedback-fixture')
        planner = G1RolloutPlanner(model)
        feedback = ResidualFeedback(alpha=1.)
        predicted = self.rows[0][2]
        feedback.update(predicted, replace(predicted, x=predicted.x + 1), planner)
        self.assertEqual(planner.feedback_scale, .25)
        sequences = planner._sequences(self.state, G1Goal(1., 0.))
        self.assertLessEqual(abs(sequences[:, :, 0]).max(), .45 * .25)
        with self.assertRaises(ValueError):
            feedback.update(predicted, self.state, planner)
        feedback.update(predicted, predicted, planner)
        self.assertEqual(planner.feedback_scale, 1.)
