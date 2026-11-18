#!/usr/bin/python3
"""Exercise repaired planning paths and retain rejection of colliding starts."""
import json
import math
import subprocess
import sys
import unittest

BENCHMARK = sys.argv.pop(1)


class PlannerRegressions(unittest.TestCase):
    def request(self, map_id, shape, start, goal, start_yaw_deg=None):
        options = [] if start_yaw_deg is None else ["--start-yaw-deg", str(start_yaw_deg)]
        result = subprocess.run(
            [BENCHMARK, "--offline-request", map_id, shape,
             *map(str, start), *map(str, goal), *options],
            capture_output=True, text=True, timeout=15)
        data = json.loads(result.stdout.splitlines()[-1])
        return result.returncode, data

    def test_repaired_routes(self):
        cases = [
            ("block", "t", [0, 0, 1], [3.75, -0.92, 1]),
            ("block", "cuboid", [3.7, -0.89, 1], [-0.92, 0.08, 1]),
            ("narrow", "cross", [0, 0, 1], [9.08, 0, 1]),
            ("hole", "cross", [-6.18, -0.1, 1], [0.24, 3.14, 1]),
        ]
        for map_id, shape, start, goal in cases:
            with self.subTest(map=map_id, shape=shape):
                code, data = self.request(map_id, shape, start, goal)
                self.assertEqual(code, 0, data["reason"])
                self.assertTrue(data["success"], data["reason"])
                self.assertFalse(data["refinement_timed_out"])
                self.assertTrue(data["backend"])
                self.assertTrue(all(math.isfinite(x) for p in data["backend"] for x in p))
                for actual, expected in zip(data["backend"][0][:3], start):
                    self.assertAlmostEqual(actual, expected, places=6)
                for actual, expected in zip(data["backend"][-1][:3], goal):
                    self.assertAlmostEqual(actual, expected, places=6)

    def test_colliding_fixed_start_still_fails(self):
        code, data = self.request("block", "cuboid", [-3.59, 3.49, 1], [-0.09, 2.42, 1])
        self.assertNotEqual(code, 0)
        self.assertFalse(data["success"])
        self.assertEqual(data["reason"], "exact start pose is in collision")
        self.assertEqual(data["start_yaw"], 0.0)
        self.assertFalse(data["backend"])

    def test_approved_starting_orientations(self):
        cases = [
            ("cuboid", [-3.59, 3.49, 1], [-0.09, 2.42, 1], -10),
            ("cuboid", [-0.09, 2.42, 1], [3.70, -0.89, 1], 10),
            ("v", [-3.98, -4.29, 1], [1.88, -5.50, 1], 30),
        ]
        for shape, start, goal, yaw in cases:
            with self.subTest(shape=shape, yaw=yaw):
                code, data = self.request("block", shape, start, goal, yaw)
                self.assertEqual(code, 0, data["reason"])
                self.assertTrue(data["success"])
                self.assertFalse(data["free_start_yaw"])
                self.assertAlmostEqual(data["start_yaw"], math.radians(yaw))
                self.assertAlmostEqual(data["backend"][0][3], math.radians(yaw))
                for actual, expected in zip(data["backend"][0][:3], start):
                    self.assertAlmostEqual(actual, expected, places=6)


if __name__ == "__main__":
    unittest.main()
