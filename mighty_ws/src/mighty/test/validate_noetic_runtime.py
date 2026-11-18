#!/usr/bin/python3
"""ROS1 migration checks; requires an isolated, already-running ROS master."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import yaml
import rospy
import rosnode
from visualization_msgs.msg import MarkerArray
from dynus_interfaces.msg import State, Trajectory, Goal
from geometry_msgs.msg import PoseStamped

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_known_map as runner
from quantitative_test import CaseNode


def wait_until(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not rospy.is_shutdown():
        if predicate(): return True
        time.sleep(.05)
    return False


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--representatives-only', action='store_true')
    p.add_argument('--case', action='append', help='Run selected cases, including execution')
    p.add_argument('--goal-yaw', type=float, help='Explicit goal yaw in degrees for regression checks')
    args = p.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    rospy.init_node('noetic_migration_validation', anonymous=True)
    cases = yaml.safe_load((ROOT / 'scripts/start2end.yaml').read_text())['cases']
    representatives = {}
    for name, case in cases.items(): representatives.setdefault(case['map'], name)
    selected = {k:v for k,v in cases.items() if not args.representatives_only or k in representatives.values()}
    if args.case:
        selected = {k:cases[k] for k in args.case}
    report = []
    for index, (name, case) in enumerate(selected.items()):
        namespace = 'MIGRATION%02d' % index
        cfg = runner.load_map_config(ROOT / 'config/mighty.yaml', case['map'])
        cfg['shape_aware_astar_shape'] = case['shape']
        params = runner.resolved_parameters(ROOT / 'config/mighty.yaml', cfg,
                SimpleNamespace(start=case['pairs'][0][0], start_yaw=0.0))
        params_file = args.results / (name + '.yaml')
        params_file.write_text(yaml.safe_dump(params))
        entry = {'case':name, 'map':case['map'], 'shape':case['shape']}
        log = (args.results / (name + '.log')).open('w')
        proc = subprocess.Popen(['roslaunch', 'mighty', 'known_map.launch',
                  'namespace:=' + namespace, 'params_file:=' + str(params_file.resolve())],
                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        node = None
        try:
            node = CaseNode(namespace)
            executable_messages = []
            trajectory_sub = rospy.Subscriber('/' + namespace + '/trajectory', Trajectory,
                                              lambda m: executable_messages.append(m))
            ready = wait_until(lambda: proc.poll() is not None or (node.ready and node.frontend_ready and node.goal.get_num_connections()), 45)
            if not ready or proc.poll() is not None:
                raise RuntimeError('startup failed or timed out; see case log')
            entry['loaded'] = True
            # Subscribe only after startup: verifies full latched snapshots.
            surface = rospy.wait_for_message('/' + namespace + '/polymap', MarkerArray, timeout=5)
            boundary = rospy.wait_for_message('/' + namespace + '/polymap/boundary', MarkerArray, timeout=5)
            entry['late_map_markers'] = len([m for m in surface.markers if m.action == m.ADD])
            entry['late_map_triangles'] = sum(len(m.points)//3 for m in surface.markers if m.action == m.ADD)
            entry['late_boundary_markers'] = len([m for m in boundary.markers if m.action == m.ADD])
            assert entry['late_map_triangles'] > 0
            if case['map'] == 'narrow': assert entry['late_boundary_markers'] == 2
            if name == representatives[case['map']] or args.case:
                start, goal = case['pairs'][0]
                node.run_pair(start, goal, 120, goal_yaw=args.goal_yaw)  # inspect structured result below
                entry['task'] = {'reached':node.reached, 'failure':node.failure,
                    'plan_ok':node.plan_ok, 'executable_trajectories':len(executable_messages), 'frontend_points':len(node.front_path),
                    'backend_points':len(node.backend_path), 'frontend_ms':node.front_time,
                    'backend_ms':node.backend_time,
                    'moving_state_samples':sum(v > .05 for v in node.velocities)}
                if node.failure:
                    entry['failure_held_position'] = not executable_messages and not any(v > .05 for v in node.velocities)
                entry['execution_passed'] = bool(node.reached and node.plan_ok and node.backend_path and any(v>.05 for v in node.velocities))
                if node.reached:
                    measured = rospy.wait_for_message('/' + namespace + '/state', State, timeout=3)
                    command = rospy.wait_for_message('/' + namespace + '/goal', Goal, timeout=3)
                    q = measured.quat
                    measured_yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                    error = abs(math.remainder(measured_yaw-command.yaw, 2*math.pi))
                    entry['task']['endpoint_yaw_error_rad'] = error
                    assert error <= .04, 'arrival must retain the configured yaw tolerance'
                    if args.goal_yaw is not None:
                        hint_error = abs(math.remainder(measured_yaw-math.radians(args.goal_yaw), 2*math.pi))
                        entry['task']['requested_yaw_error_rad'] = hint_error
                        assert hint_error <= .04, 'explicit-yaw regression fixture must reach requested yaw'
            else:
                entry['execution_tested'] = False
        except Exception as error:
            entry['error'] = str(error)
        finally:
            if node:
                for sub in node.subscribers: sub.unregister()
                node.goal.unregister()
                trajectory_sub.unregister()
            stop(proc)
            log.close()
            entry['clean_exit'] = wait_until(lambda: not any(n.startswith('/' + namespace + '/') for n in rosnode.get_node_names()), 5)
        report.append(entry)
        (args.results / 'runtime.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    if not all(e.get('loaded') and e.get('clean_exit') and 'error' not in e for e in report):
        return 1
    return 0 if all(e.get('execution_passed', True) for e in report) else 2


if __name__ == '__main__':
    raise SystemExit(main())
