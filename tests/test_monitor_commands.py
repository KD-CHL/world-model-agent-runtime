import math
import unittest
from dataclasses import FrozenInstanceError

from wmal.monitor.commands import CommandError, TaskSpec, parse_command


class MonitorCommandTests(unittest.TestCase):
    def test_parses_g1_forward_as_body_relative_distance(self):
        spec = parse_command({'text': '前进 0.4 米'})
        self.assertIsNone(spec.goal)
        self.assertEqual(spec.task, 'relative_translation')
        self.assertEqual(dict(spec.relative_intent),
                         {'kind': 'translation', 'direction': 'forward', 'distance_m': 0.4})
        with self.assertRaises(TypeError):
            spec.relative_intent['direction'] = 'backward'

        backward = parse_command({'text': '后退 0.25 米'})
        self.assertIsNone(backward.goal)
        self.assertEqual(dict(backward.relative_intent),
                         {'kind': 'translation', 'direction': 'backward', 'distance_m': 0.25})

    def test_parses_g1_turn_and_absolute_coordinate_commands(self):
        left = parse_command({'text': '左转 90 度'})
        self.assertEqual(left.mode, 'g1_navigation')
        self.assertEqual(left.task, 'relative_turn')
        self.assertIsNone(left.goal)
        self.assertEqual(dict(left.relative_intent),
                         {'kind': 'turn', 'direction': 'left', 'angle_deg': 90.0})
        right = parse_command({'text': '右转 30 度'})
        self.assertEqual(right.task, 'relative_turn')
        self.assertIsNone(right.goal)
        self.assertEqual(dict(right.relative_intent),
                         {'kind': 'turn', 'direction': 'right', 'angle_deg': 30.0})
        absolute = parse_command({'text': '到达坐标 1.25 -0.5 朝向 30 度'})
        self.assertEqual((absolute.task, absolute.goal), ('world_goal', (1.25, -0.5, 30.0)))
        no_yaw = parse_command({'text': '到达坐标 1.25 -0.5'})
        self.assertEqual(no_yaw.goal, (1.25, -0.5, None))

    def test_parses_arm_joint_instruction_in_radians(self):
        spec = parse_command({'text': '肩关节到 0.45 rad，肘关节到 0.95 rad'})
        self.assertEqual(spec, TaskSpec(
            mode='arm_joint_targets', task='stack_block', goal=(0.45, 0.95),
            waypoints=(), options={'baseline': 'A3', 'horizon': 4,
                                   'max_cycles': 30, 'error_budget': 1.0}))

    def test_parses_structured_g1_and_arm_commands(self):
        g1 = parse_command({'mode': 'g1_navigation',
                            'goal': {'x_m': 1.0, 'y_m': -0.25, 'yaw_deg': 45},
                            'waypoints': []})
        self.assertEqual(g1.task, 'world_goal')
        self.assertEqual(g1.goal, (1.0, -0.25, 45.0))
        arm = parse_command({'mode': 'arm_joint_targets', 'task': 'stack_block',
                             'goal_rad': [0.45, 0.95], 'waypoints_rad': [[0.4, 0.9]],
                             'baseline': 'A2', 'horizon': 3, 'max_cycles': 24,
                             'error_budget': 0.5})
        self.assertEqual(arm.goal, (0.45, 0.95))
        self.assertEqual(arm.waypoints, ((0.4, 0.9),))
        self.assertEqual(dict(arm.options), {'baseline': 'A2', 'horizon': 3,
                                             'max_cycles': 24, 'error_budget': 0.5})

    def test_task_spec_is_frozen(self):
        spec = parse_command({'text': '前进 0.2 米'})
        with self.assertRaises(FrozenInstanceError):
            spec.mode = 'arm_joint_targets'

    def test_rejects_missing_or_ambiguous_schema_with_client_error(self):
        for payload in ({}, {'text': '前进 0.2 米', 'mode': 'g1_navigation'},
                        {'mode': 'g1_navigation'}, [], '前进 0.2 米'):
            with self.subTest(payload=payload):
                with self.assertRaises(CommandError) as raised:
                    parse_command(payload)
                self.assertEqual(raised.exception.status, 400)
                self.assertEqual(raised.exception.code, 'invalid_command')

    def test_rejects_unsupported_language_instead_of_guessing(self):
        for text in ('向前走一点', '先前进 0.2 米再左转 10 度',
                     '肩关节到 0.45，肘关节到 0.95 rad'):
            with self.subTest(text=text):
                with self.assertRaises(CommandError) as raised:
                    parse_command({'text': text})
                self.assertEqual(raised.exception.status, 422)
                self.assertEqual(raised.exception.code, 'unsupported_intent')

    def test_rejects_unknown_keys_for_all_structured_schemas(self):
        cases = (
            {'mode': 'g1_navigation', 'goal': {'x_m': 0, 'y_m': 0}, 'magic': True},
            {'mode': 'g1_navigation', 'goal': {'x_m': 0, 'y_m': 0, 'yaw_deg': None,
                                               'z': 0}},
            {'mode': 'arm_joint_targets', 'task': 'stack_block', 'goal_rad': [0.4, 0.9],
             'python': 'os.system'},
        )
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(CommandError) as raised:
                    parse_command(payload)
                self.assertEqual(raised.exception.status, 400)

    def test_rejects_boolean_non_finite_and_out_of_range_values(self):
        invalid_payloads = (
            {'mode': 'g1_navigation', 'goal': {'x_m': True, 'y_m': 0}},
            {'mode': 'g1_navigation', 'goal': {'x_m': math.nan, 'y_m': 0}},
            {'mode': 'g1_navigation', 'goal': {'x_m': math.inf, 'y_m': 0}},
            {'mode': 'g1_navigation', 'goal': {'x_m': 10 ** 400, 'y_m': 0}},
            {'mode': 'g1_navigation', 'goal': {'x_m': 20.01, 'y_m': 0}},
            {'mode': 'g1_navigation', 'goal': {'x_m': 0, 'y_m': 0, 'yaw_deg': 181}},
            {'text': '前进 2.01 米'},
            {'text': '右转 181 度'},
            {'mode': 'arm_joint_targets', 'task': 'stack_block', 'goal_rad': [True, 0.9]},
            {'mode': 'arm_joint_targets', 'task': 'stack_block', 'goal_rad': [math.nan, 0.9]},
            {'mode': 'arm_joint_targets', 'task': 'stack_block', 'goal_rad': [10 ** 400, 0.9]},
            {'mode': 'arm_joint_targets', 'task': 'stack_block', 'goal_rad': [0.66, 0.9]},
            {'mode': 'arm_joint_targets', 'task': 'stack_block', 'goal_rad': [0.4, 1.16]},
            {'mode': 'arm_joint_targets', 'task': 'stack_block', 'goal_rad': [0.4, 0.9],
             'waypoints_rad': [[0.4, 1.16]]},
            {'mode': 'arm_joint_targets', 'task': 'stack_block', 'goal_rad': [0.4, 0.9],
             'max_cycles': 101},
            {'mode': 'g1_navigation', 'goal': {'x_m': 0, 'y_m': 0},
             'waypoints': [{'x_m': 1, 'y_m': 0}, {'x_m': 2.1, 'y_m': 0}]},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(CommandError):
                    parse_command(payload)

    def test_rejects_invalid_arm_options_and_task(self):
        invalid_payloads = (
            {'mode': 'arm_joint_targets', 'task': 'unknown', 'goal_rad': [0.4, 0.9]},
            {'mode': 'arm_joint_targets', 'task': 'stack_block', 'goal_rad': [0.4, 0.9],
             'baseline': 'A4'},
            {'mode': 'arm_joint_targets', 'task': 'stack_block', 'goal_rad': [0.4, 0.9],
             'horizon': True},
            {'mode': 'arm_joint_targets', 'task': 'stack_block', 'goal_rad': [0.4, 0.9],
             'error_budget': math.inf},
            {'mode': 'arm_joint_targets', 'task': 'stack_block', 'goal_rad': [0.4, 0.9],
             'baseline': 'A0', 'waypoints_rad': [[0.4, 0.9]], 'horizon': 2},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(CommandError):
                    parse_command(payload)

    def test_enforces_body_and_text_size_limits(self):
        too_long = 'a' * 257
        with self.assertRaises(CommandError) as raised:
            parse_command({'text': too_long})
        self.assertEqual(raised.exception.status, 400)
        self.assertEqual(raised.exception.code, 'invalid_command')
        oversized_payload = {'mode': 'g1_navigation',
                             'goal': {'x_m': 0, 'y_m': 0},
                             'padding': 'x' * (16 * 1024)}
        with self.assertRaises(CommandError) as raised:
            parse_command(oversized_payload)
        self.assertEqual(raised.exception.status, 400)

    def test_checks_capabilities_without_loading_models(self):
        with self.assertRaises(CommandError) as raised:
            parse_command({'mode': 'arm_joint_targets', 'task': 'stack_block',
                           'goal_rad': [0.4, 0.9]},
                          capabilities={'available_modes': {'g1_navigation'},
                                        'arm_max_horizon': None})
        self.assertEqual(raised.exception.status, 503)
        self.assertEqual(raised.exception.code, 'mode_unavailable')

        with self.assertRaises(CommandError):
            parse_command({'mode': 'arm_joint_targets', 'task': 'stack_block',
                           'goal_rad': [0.4, 0.9], 'horizon': 5},
                          capabilities={'available_modes': {'g1_navigation', 'arm_joint_targets'},
                                        'arm_max_horizon': 4})

    def test_rejects_malformed_capability_snapshot_as_client_error(self):
        request = {'mode': 'arm_joint_targets', 'task': 'stack_block', 'goal_rad': [0.4, 0.9]}
        for capabilities in (
                {'available_modes': 'g1_navigation'},
                {'available_modes': {'g1_navigation', 'arm_joint_targets'}, 'arm_max_horizon': True},
                {'available_modes': {'g1_navigation', 'arm_joint_targets'}, 'arm_max_horizon': 0},
                {'available_modes': {'g1_navigation', 'arm_joint_targets'}, 'arm_max_horizon': 1.5}):
            with self.subTest(capabilities=capabilities):
                with self.assertRaises(CommandError) as raised:
                    parse_command(request, capabilities=capabilities)
                self.assertEqual(raised.exception.status, 400)


if __name__ == '__main__':
    unittest.main()
