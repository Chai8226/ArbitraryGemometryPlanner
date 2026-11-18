#!/usr/bin/python3
"""ROS-independent migration checks against the shipped map/shape configuration."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('runner', ROOT / 'scripts/run_known_map.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NoeticConfigTests(unittest.TestCase):
    def test_all_map_shape_bindings_and_parameter_types(self):
        count = 0
        for map_id in runner.SUPPORTED_KNOWN_MAPS:
            cfg = runner.load_map_config(ROOT / 'config/mighty.yaml', map_id)
            for shape in runner.SHAPES_BY_SHAPE_SET[cfg['shape_set']]:
                runner.require_shape_for_shape_set(shape, cfg['shape_set'])
                cfg['shape_aware_astar_shape'] = shape
                params = runner.resolved_parameters(ROOT / 'config/mighty.yaml', cfg,
                        SimpleNamespace(start=cfg['default_start'], start_yaw=90.0))
                planner = params['mighty_node']
                self.assertEqual(planner['shape_aware_astar/shape'], shape)
                self.assertEqual(planner['known_map/id'], map_id)
                self.assertEqual(planner['v_max'], 2.0)
                self.assertEqual(planner['shape_aware_astar/timeout_seconds'], 1.0)
                self.assertIsInstance(planner['time_weight'], float)  # inherited lower layer
                self.assertTrue(all('.' not in k for k in planner))
                self.assertIsInstance(planner['drone_bbox'], list)
                self.assertIsInstance(planner['force_goal_z'], bool)
                self.assertAlmostEqual(params['fake_sim']['start_yaw'], 1.5707963267948966)
                self.assertTrue((ROOT / 'shape' / cfg['shape_set'] / 'ply' / (shape + '.polymap.json')).is_file())
                count += 1
        self.assertEqual(count, 18)

    def test_setup_symlink_must_keep_overlay_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            private = folder / "private"
            private.mkdir()
            (private / "setup.bash").write_text("true\n")
            link = folder / "setup.bash"
            link.symlink_to(private / "setup.bash")
            self.assertEqual(runner.find_setup_bash(str(link)), link)
            self.assertEqual(runner.discover_underlays([str(link)]), [link])

    def test_unsupported_shape_rejected(self):
        with self.assertRaises(SystemExit):
            runner.require_shape_for_shape_set('long2', 'normal')

    def test_invalid_asset_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.yaml'
            doc = yaml.safe_load((ROOT / 'config/mighty.yaml').read_text())
            doc['known_maps']['hole']['pcd_path'] = 'missing.pcd'
            p.write_text(yaml.safe_dump(doc))
            with self.assertRaises(SystemExit):
                runner.load_map_config(p, 'hole')

    def test_extends_override_and_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a.yaml', Path(tmp) / 'b.yaml'
            b.write_text('section:\n  value: 2\n  retained: true\n')
            a.write_text('extends: b.yaml:section\nvalue: 3\n')
            self.assertEqual(runner._resolve_extends(a), {'value': 3, 'retained': True})
            b.write_text('extends: a.yaml\n')
            with self.assertRaises(SystemExit):
                runner._resolve_extends(a)

    def test_dry_run_and_nan_rejection(self):
        cmd = ['/usr/bin/python3', str(ROOT / 'scripts/run_known_map.py'), '--map', 'hole', '--dry-run']
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('mighty_node', yaml.safe_load(result.stdout))
        result = subprocess.run(cmd + ['--goal', 'nan', '0', '1'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_rviz_topics_are_native_ros1_and_map_is_snapshot(self):
        def walk(value):
            if isinstance(value, dict):
                yield value
                for v in value.values(): yield from walk(v)
            elif isinstance(value, list):
                for v in value: yield from walk(v)
        for file in (ROOT / 'config').glob('*.rviz'):
            document = yaml.safe_load(file.read_text())
            entries = list(walk(document))
            self.assertFalse(any('rviz_default_plugins/' in v.get('Class', '') for v in entries))
            surface = [v for v in entries if v.get('Marker Topic', '').endswith('/polymap')]
            self.assertEqual(len(surface), 1)
            self.assertEqual(surface[0]['Class'], 'rviz/MarkerArray')
            goal = next(v for v in entries if v.get('Class') == 'rviz/SetGoal')
            self.assertTrue(goal['Topic'].endswith('/term_goal'))


if __name__ == '__main__':
    unittest.main()
