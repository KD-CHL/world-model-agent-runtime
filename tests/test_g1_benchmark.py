import unittest
from wmal.envs.indoor_scene import IndoorScene, Box


def make_scene(name):
    scene = IndoorScene()
    if name == 'open':
        scene.boxes = tuple(b for b in scene.boxes if b.name.startswith('wall_'))
    elif name == 'corridor':
        scene.boxes = tuple(b for b in scene.boxes if b.name.startswith('wall_')) + (
            Box('aisle_left', 2.5, 1., 1., .25, 1.),
            Box('aisle_right', 2.5, -1., 1., .25, 1.))
    elif name != 'indoor':
        raise ValueError('Unknown scene')
    return scene


class BenchmarkTests(unittest.TestCase):
    def test_all_ten_missions_have_feasible_routes(self):
        suite = {'schema': 'wmal.g1.experiments.v1',
         'experiments': [{'id': '01_forward',
                          'name': '直行到点',
                          'scene': 'open',
                          'goals': [[1.5, 0]],
                          'cycles': 80},
                         {'id': '02_reverse',
                          'name': '后退到点',
                          'scene': 'open',
                          'goals': [[-0.6, 0]],
                          'cycles': 100},
                         {'id': '03_lateral',
                          'name': '侧向目标',
                          'scene': 'open',
                          'goals': [[0, 1]],
                          'cycles': 100},
                         {'id': '04_diagonal',
                          'name': '斜向到点',
                          'scene': 'open',
                          'goals': [[1.5, 1]],
                          'cycles': 100},
                         {'id': '05_heading',
                          'name': '指定朝向停靠',
                          'scene': 'open',
                          'goals': [[1, 0, 1.5707963267948966]],
                          'cycles': 150},
                         {'id': '06_obstacle',
                          'name': '中央障碍绕行',
                          'scene': 'indoor',
                          'goals': [[4, 0]],
                          'cycles': 150},
                         {'id': '07_corridor',
                          'name': '狭窄通道',
                          'scene': 'corridor',
                          'goals': [[4, 0]],
                          'cycles': 150},
                         {'id': '08_patrol',
                          'name': '多点巡航',
                          'scene': 'open',
                          'goals': [[1, 0], [1, 1], [0, 1]],
                          'cycles': 100},
                         {'id': '09_return',
                          'name': '绕障往返',
                          'scene': 'indoor',
                          'goals': [[4, 0], [0, 0]],
                          'cycles': 150},
                         {'id': '10_delivery',
                          'name': '多站点配送',
                          'scene': 'indoor',
                          'goals': [[0.5, -1.5], [3.2, -0.9], [4, 0]],
                          'cycles': 150}]}
        self.assertEqual(len({e['id'] for e in suite['experiments']}), 10)
        for experiment in suite['experiments']:
            scene, point = make_scene(experiment['scene']), (0., 0.)
            scene.robot_radius += .15
            for goal in experiment['goals']:
                route = scene.route(point, tuple(goal[:2]))
                self.assertTrue(all(scene.segment_free(a,b) for a,b in zip(route, route[1:])))
                point = tuple(goal[:2])
