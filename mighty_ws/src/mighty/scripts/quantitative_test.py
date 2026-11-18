#!/usr/bin/env python3
"""Run start2end.yaml cases and write CSV plus interactive Plotly reports."""
import argparse
import csv
import json
import math
import os
import re
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import rospy
import yaml
from dynus_interfaces.msg import ComputationTimes, GlobalPlanStatus, State
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath
from std_msgs.msg import Empty, Float64, String
from visualization_msgs.msg import Marker, MarkerArray

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "maps").is_dir():
    import rospkg
    ROOT = Path(rospkg.RosPack().get_path("mighty"))
FIELDS = ["caseid", "平均前端路径长度", "平均前端计算时间_ms", "平均中端轨迹长度",
          "平均中端计算时间_ms", "平均后端轨迹长度", "平均后端计算时间_ms", "总平均飞速度_mps"]
STATUS_RE = re.compile(r"\b(path_length|elapsed)=([^\s]+)")


def finite_mean(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
    return sum(values) / len(values) if values else None


def points_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def marker_points(msg):
    """Extract the longest rendered trajectory polyline from a MarkerArray."""
    candidates = []
    for marker in msg.markers:
        if marker.action != Marker.ADD or len(marker.points) < 2:
            continue
        pts = [[p.x, p.y, p.z] for p in marker.points]
        # MIGHTY's stateVector2ColoredMarkerArray emits connected LINE_LIST
        # pairs: [p0,p0], [p0,p1], [p1,p2], ... . Keep each segment end.
        if marker.type == Marker.LINE_LIST:
            pts = [pts[0]] + pts[1::2]
        candidates.append(pts)
    return max(candidates, key=len, default=[])


def downsample(points, maximum=6000):
    return points[::max(1, len(points) // maximum)]


def segmented_line(points, segment_size):
    """Render a 3-D dashed/dotted line without unsupported scatter3d dash."""
    if segment_size <= 1:
        return points
    rendered = []
    for index in range(0, len(points) - 1, segment_size * 2):
        rendered.extend(points[index:index + segment_size + 1])
        rendered.append(None)
    return rendered


def read_pcd_xyz(path):
    """Read XYZ from the project's ascii or binary XYZ PCD files."""
    raw = Path(path).read_bytes()
    end = raw.find(b"DATA ")
    end = raw.find(b"\n", end) + 1
    header = raw[:end].decode("ascii", "replace")
    if "DATA ascii" in header:
        return downsample([[float(v) for v in line.split()[:3]]
                           for line in raw[end:].decode("ascii", "replace").splitlines() if line.strip()])
    fields = next(line.split()[1:] for line in header.splitlines() if line.startswith("FIELDS "))
    sizes = list(map(int, next(line.split()[1:] for line in header.splitlines() if line.startswith("SIZE "))))
    types = next(line.split()[1:] for line in header.splitlines() if line.startswith("TYPE "))
    count = int(next(line.split()[1] for line in header.splitlines() if line.startswith("POINTS ")))
    if fields[:3] != ["x", "y", "z"] or sizes[:3] != [4, 4, 4] or types[:3] != ["F", "F", "F"]:
        return []
    stride = sum(sizes)
    step = max(1, count // 6000)
    return [list(struct.unpack_from("<fff", raw, end + i * stride)) for i in range(0, count, step)]


class CaseNode:
    def __init__(self, namespace):
        self.subscribers = []
        p = "/" + namespace.strip("/")
        self.goal = rospy.Publisher(p + "/term_goal", PoseStamped, queue_size=1)
        self.ready = False; self.ready_at = None; self.frontend_ready = False; self.reset()
        self.subscribers.append(rospy.Subscriber(p + "/state", State, self._state, queue_size=10))
        self.subscribers.append(rospy.Subscriber(p + "/goal_reached", Empty, self._reached, queue_size=10))
        self.subscribers.append(rospy.Subscriber(p + "/shape_aware_astar/path", RosPath, self._front, queue_size=10))
        self.subscribers.append(rospy.Subscriber(p + "/shape_aware_astar/status", String, self._front_status, queue_size=10))
        self.subscribers.append(rospy.Subscriber(p + "/computation_times", ComputationTimes, self._times, queue_size=10))
        self.subscribers.append(rospy.Subscriber(p + "/middle_stage_time_ms", Float64, self._middle_time, queue_size=10))
        self.subscribers.append(rospy.Subscriber(p + "/traj_middle", MarkerArray, self._middle, queue_size=10))
        self.subscribers.append(rospy.Subscriber(p + "/traj_backend", MarkerArray, self._backend, queue_size=10))
        self.subscribers.append(rospy.Subscriber(p + "/global_plan_status", GlobalPlanStatus, self._plan, queue_size=10))

    def reset(self):
        self.sent_at = None; self.reached = False; self.failure = None; self.front_path = []; self.middle_path = []; self.backend_path = []
        self.front_length = self.front_time = self.middle_time = self.backend_time = None; self.velocities = []; self.plan_ok = False

    def _after_send(self): return self.sent_at is not None
    def _state(self, m):
        if not self.ready:
            self.ready = True
            self.ready_at = time.monotonic()
        if self._after_send(): self.velocities.append(math.sqrt(m.vel.x*m.vel.x + m.vel.y*m.vel.y + m.vel.z*m.vel.z))
    def _reached(self, _):
        if self._after_send(): self.reached = True
    def _front(self, m):
        if self._after_send(): self.front_path = [[q.pose.position.x, q.pose.position.y, q.pose.position.z] for q in m.poses]
    def _front_status(self, m):
        if "ready" in m.data.lower(): self.frontend_ready = True
        if self._after_send() and "success" in m.data:
            values = dict(STATUS_RE.findall(m.data))
            if "path_length" in values: self.front_length = float(values["path_length"])
            if "elapsed" in values: self.front_time = float(values["elapsed"]) * 1000
        elif self._after_send() and any(x in m.data.lower() for x in ("failed", "no_path", "rejected")): self.failure = m.data
    def _times(self, m):
        if self._after_send():
            self.backend_time = m.trajectory_optimization_ms
            self.front_time = self.front_time if self.front_time is not None else m.frontend_planning_ms
    def _middle_time(self, m):
        if self._after_send() and math.isfinite(m.data): self.middle_time = m.data
    def _middle(self, m):
        if self._after_send(): self.middle_path = marker_points(m)
    def _backend(self, m):
        if self._after_send(): self.backend_path = marker_points(m)
    def _plan(self, m):
        if self._after_send():
            self.plan_ok = m.outcome in (GlobalPlanStatus.NOMINAL, GlobalPlanStatus.SAFE_DEGRADED)
            if m.outcome == GlobalPlanStatus.FAILED: self.failure = m.reason or "global planner failed"

    def run_pair(self, start, goal, timeout, goal_yaw=None):
        self.reset()
        msg = PoseStamped(); msg.header.frame_id = "map"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(value) for value in goal)
        # A zero quaternion deliberately requests free yaw in MIGHTY.
        if goal_yaw is not None:
            angle = math.radians(goal_yaw)
            msg.pose.orientation.z = math.sin(angle / 2)
            msg.pose.orientation.w = math.cos(angle / 2)
        self.sent_at = time.monotonic(); self.goal.publish(msg)
        deadline = self.sent_at + timeout
        while time.monotonic() < deadline and not self.failure and not self.reached:
            time.sleep(0.1)
        if not self.reached and not self.failure: self.failure = "timeout waiting for goal_reached"
        return {"start": start, "goal": goal, "status": "FAILED" if self.failure else "OK", "reason": self.failure or "",
                "frontend_length": self.front_length if self.front_length is not None else points_length(self.front_path),
                "frontend_ms": self.front_time, "middle_length": points_length(self.middle_path), "middle_ms": self.middle_time,
                "backend_length": points_length(self.backend_path), "backend_ms": self.backend_time,
                "mean_speed": sum(self.velocities)/len(self.velocities) if self.velocities else None,
                "frontend": self.front_path, "middle": self.middle_path, "backend": self.backend_path}


def load_polymap(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["polyhedra"]


def mesh_trace(polyhedra, translation, name, color, opacity):
    """Convert all PolyMap convex components to one Plotly triangle mesh."""
    x, y, z, i, j, k = [], [], [], [], [], []
    yaw = translation[3] if len(translation) > 3 else 0.0
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    for component in polyhedra:
        offset = len(x)
        for vx, vy, vz in component["vertices"]:
            x.append(cos_yaw * vx - sin_yaw * vy + translation[0])
            y.append(sin_yaw * vx + cos_yaw * vy + translation[1]); z.append(vz + translation[2])
        for a, b, c in component["triangles"]:
            i.append(offset + a); j.append(offset + b); k.append(offset + c)
    return {"type":"mesh3d", "name":name, "x":x, "y":y, "z":z, "i":i, "j":j, "k":k,
            "color":color, "opacity":opacity, "flatshading":True, "hoverinfo":"name"}


def report(path, case, rows):
    map_poly = load_polymap(ROOT / "maps" / "polymap" / f"{case['map']}.polymap.json")
    scale = "small" if case["map"] == "hole" else "normal"
    body_poly = load_polymap(ROOT / "shape" / scale / "ply" / f"{case['shape']}.polymap.json")
    traces = [mesh_trace(map_poly, [0, 0, 0], "map PolyMap", "#6c757d", 0.22)]
    for i, row in enumerate(rows, 1):
        for key, color in (("frontend", "#1f77b4"), ("middle", "#ff7f0e"), ("backend", "#2ca02c")):
            pts = row[key]
            if pts:
                rendered = segmented_line(pts, {"frontend":1, "middle":4, "backend":1}[key])
                traces.append({"type":"scatter3d","mode":"lines","name":f"pair {i} {key}","x":[p[0] if p else None for p in rendered],"y":[p[1] if p else None for p in rendered],"z":[p[2] if p else None for p in rendered],"line":{"color":color,"width":6}})
        body_path = row["backend"] or row["middle"] or row["frontend"]
        for index, point in enumerate(body_path[::max(1, len(body_path)//12)]):
            traces.append(mesh_trace(body_poly, point, "shape PolyMap" if i == 1 and index == 0 else "shape", "#d62728", 0.42))
    payload = json.dumps({"traces": traces, "rows": rows}, ensure_ascii=False)
    html = """<!doctype html><meta charset='utf-8'><title>{title}</title>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script><div id='plot' style='width:100%;height:100vh'></div>
<script>const d={payload}, plot=document.querySelector('#plot');
const layout={{title:'{title}',scene:{{aspectmode:'data',xaxis:{{visible:false}},yaxis:{{visible:false}},zaxis:{{visible:false}}}}}};
Plotly.newPlot(plot,d.traces,layout);
</script>""".format(title=path.stem, payload=payload)
    path.write_text(html, encoding="utf-8")


def launch(case, namespace, master_uri):
    command = [sys.executable, str(ROOT / "scripts" / "run_known_map.py"), "--map", case["map"], "--shape", case["shape"], "--start", *map(str, case["pairs"][0][0]), "--namespace", namespace, "--ros-master-uri", str(master_uri), "--no-rviz", "--kill-stale", "--detached"]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def stop_session():
    owner = subprocess.run(["tmux", "show-option", "-v", "-t", "mighty_known_map", "@mighty_project"],
                           capture_output=True, text=True)
    if owner.returncode != 0:
        return
    if owner.stdout.strip() != str(ROOT):
        raise RuntimeError("refusing to stop a tmux session owned by another checkout")
    subprocess.run(["tmux", "kill-session", "-t", "mighty_known_map"], check=True)
    time.sleep(1.0)



def main():
    p = argparse.ArgumentParser(); p.add_argument("--runs", type=int, default=10); p.add_argument("--timeout", type=float, default=120); p.add_argument("--results", type=Path, default=ROOT / "scripts" / "results"); p.add_argument("--namespace", default="MIGHTY_BENCH"); p.add_argument("--ros-master-uri", default=os.environ.get("ROS_MASTER_URI", "http://localhost:11311")); p.add_argument("--case", help="Run one caseid from start2end.yaml."); p.add_argument("--pair-limit", type=int, help="Limit pairs per selected case (diagnostic use)."); args = p.parse_args()
    cases = yaml.safe_load((ROOT / "scripts" / "start2end.yaml").read_text())["cases"]
    if args.case:
        if args.case not in cases: p.error(f"unknown case: {args.case}")
        cases = {args.case: cases[args.case]}
    if args.pair_limit is not None:
        if args.pair_limit < 1: p.error("--pair-limit must be positive")
        cases = {caseid: {**case, "pairs": case["pairs"][:args.pair_limit]}
                 for caseid, case in cases.items()}
    os.environ["ROS_MASTER_URI"] = args.ros_master_uri
    # A master must already be running for repeated benchmark sessions.
    import rosgraph
    if not rosgraph.is_master_online():
        raise RuntimeError("start roscore before running quantitative tests")
    rospy.init_node("mighty_quantitative_test", anonymous=True)
    args.results.mkdir(parents=True, exist_ok=True); all_rows = {caseid: [] for caseid in cases}
    for run in range(1, args.runs + 1):
        output = args.results / f"val{run}"; output.mkdir(exist_ok=True)
        for caseid, case in cases.items():
            print(f"[val{run}] {caseid}", flush=True); stop_session()
            if launch(case, args.namespace, args.ros_master_uri):
                rows = [{"status":"FAILED", "reason":"launch failed", "frontend":[], "middle":[], "backend":[]}]
            else:
                node = None
                try:
                    node = CaseNode(args.namespace); deadline = time.monotonic() + 30
                    while (not node.ready or not node.frontend_ready or node.goal.get_num_connections() == 0) and time.monotonic() < deadline: time.sleep(0.2)
                    rows = ([node.run_pair(start, goal, args.timeout) for start, goal in case["pairs"]]
                            if node.ready and node.frontend_ready and node.goal.get_num_connections() else [{"status":"FAILED", "reason":"planner startup timeout", "frontend":[], "middle":[], "backend":[]}])
                except Exception as exc:
                    rows = [{"status":"FAILED", "reason":f"runner error: {exc}", "frontend":[], "middle":[], "backend":[]}]
                finally:
                    if node is not None:
                        for sub in node.subscribers: sub.unregister()
                        node.goal.unregister()

            report(output / f"{caseid}.html", case, rows); all_rows[caseid].extend(rows); stop_session()
    with (args.results / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader()
        for caseid, rows in all_rows.items(): writer.writerow({FIELDS[0]:caseid, FIELDS[1]:finite_mean(rows,"frontend_length"), FIELDS[2]:finite_mean(rows,"frontend_ms"), FIELDS[3]:finite_mean(rows,"middle_length"), FIELDS[4]:finite_mean(rows,"middle_ms"), FIELDS[5]:finite_mean(rows,"backend_length"), FIELDS[6]:finite_mean(rows,"backend_ms"), FIELDS[7]:finite_mean(rows,"mean_speed")})

if __name__ == "__main__":
    try: main()
    finally: stop_session()
