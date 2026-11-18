#!/usr/bin/python3
"""Low-noise ANSI dashboard for a running MIGHTY agent."""

import argparse
import math
import re
import shutil
import sys
import time

import rospy
from dynus_interfaces.msg import ComputationTimes, GlobalPlanStatus, State
from std_msgs.msg import Float64, String, UInt8
from visualization_msgs.msg import Marker, MarkerArray


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
COLORS = {
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "red": "\033[31m",
    "gray": "\033[90m",
    "white": "\033[97m",
}
FSM_NAMES = {
    0: ("YAWING", "yellow"),
    1: ("TRAVELING", "green"),
    2: ("GOAL_SEEN", "cyan"),
    3: ("GOAL_REACHED", "blue"),
}
STATUS_FIELD_RE = re.compile(r"\b([a-z_]+)=([^\s]+)")


def paint(text, color=None, *, bold=False, dim=False, enabled=True):
    if not enabled:
        return text
    prefix = ""
    if bold:
        prefix += BOLD
    if dim:
        prefix += DIM
    if color:
        prefix += COLORS[color]
    return f"{prefix}{text}{RESET}"


class DashboardModel:
    """Latest state and the current global-once planning pipeline."""

    def __init__(self):
        self.state = None
        self.state_at = None
        self.acceleration = None
        self.fsm = None
        self.fsm_at = None
        self.plan = None
        self.plan_at = None
        self.plan_status = None
        self.plan_status_at = None
        self.shape_aware_astar_status = None
        self.shape_aware_astar_status_at = None
        self.middle_stage_ms = None
        self.middle_trajectory_at = None
        self.middle_trajectory_ready = False
        self.backend_trajectory_at = None
        self.backend_trajectory_ready = False

    def update_state(self, msg, now=None):
        received_at = time.monotonic() if now is None else now
        if self.state is not None and self.state_at is not None:
            elapsed = received_at - self.state_at
            if 1.0e-3 <= elapsed <= 1.0:
                previous = self.state.vel
                current = msg.vel
                self.acceleration = (
                    (current.x - previous.x) / elapsed,
                    (current.y - previous.y) / elapsed,
                    (current.z - previous.z) / elapsed,
                )
            elif elapsed > 1.0:
                self.acceleration = None
        self.state = msg
        self.state_at = received_at

    def update_fsm(self, msg, now=None):
        self.fsm = int(msg.data)
        self.fsm_at = time.monotonic() if now is None else now

    def update_plan(self, msg, now=None):
        self.plan = msg
        self.plan_at = time.monotonic() if now is None else now

    def update_plan_status(self, msg, now=None):
        self.plan_status = msg
        self.plan_status_at = time.monotonic() if now is None else now
        if msg.outcome == GlobalPlanStatus.FAILED:
            self.middle_stage_ms = None
            self.middle_trajectory_ready = False
            self.backend_trajectory_ready = False

    def update_shape_aware_astar_status(self, msg, now=None):
        self.shape_aware_astar_status = str(msg.data)
        self.shape_aware_astar_status_at = time.monotonic() if now is None else now
        if self.shape_aware_astar_status.strip().lower() == "running":
            self.middle_stage_ms = None
            self.middle_trajectory_ready = False
            self.backend_trajectory_ready = False

    @staticmethod
    def _trajectory_ready(msg):
        return any(marker.action == Marker.ADD for marker in msg.markers)

    def update_middle_trajectory(self, msg, now=None):
        self.middle_trajectory_ready = self._trajectory_ready(msg)
        self.middle_trajectory_at = time.monotonic() if now is None else now

    def update_middle_stage_time(self, msg):
        value = float(msg.data)
        self.middle_stage_ms = value if math.isfinite(value) and value >= 0.0 else None

    def update_backend_trajectory(self, msg, now=None):
        self.backend_trajectory_ready = self._trajectory_ready(msg)
        self.backend_trajectory_at = time.monotonic() if now is None else now


def _number(value, unit=""):
    if value is None:
        return "--"
    return f"{value:.2f}{unit}"


def _clip(text, width):
    """Keep a rendered line inside the pane instead of letting tmux wrap it."""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[:width - 1] + "…"


def _shape_aware_astar_status_color(status):
    if status is None:
        return "gray"
    lowered = status.lower()
    if any(token in lowered for token in
           ("failed", "unavailable", "rejected", "collision", "no_path")):
        return "red"
    if "running" in lowered:
        return "yellow"
    if "success" in lowered or "ready" in lowered:
        return "green"
    return "gray"


def _status_fields(status):
    return {} if status is None else dict(STATUS_FIELD_RE.findall(status))


def _vector_text(vector, unit):
    return f"({vector[0]:+.2f}, {vector[1]:+.2f}, {vector[2]:+.2f}) {unit}"


def _yaw_from_quaternion(quaternion):
    """Return the normalized attitude yaw, or ``None`` for invalid input."""
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return None
    x, y, z, w = (component / norm for component in (x, y, z, w))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_text(yaw):
    if yaw is None:
        return "--"
    return f"{yaw:+.2f} rad ({math.degrees(yaw):+.1f}°)"


def _section(title, width):
    return f"── {title} " + "─" * max(0, width - len(title) - 4)


def _outcome_name(plan_status):
    if plan_status is None:
        return "WAITING"
    names = {
        GlobalPlanStatus.NOMINAL: "NOMINAL",
        GlobalPlanStatus.SAFE_DEGRADED: "SAFE DEGRADED",
        GlobalPlanStatus.FAILED: "FAILED",
        GlobalPlanStatus.NO_ACTION: "NO ACTION",
    }
    return names.get(plan_status.outcome, "UNKNOWN")


def _outcome_color(plan_status):
    if plan_status is None:
        return "gray"
    if plan_status.outcome == GlobalPlanStatus.FAILED:
        return "red"
    if plan_status.outcome == GlobalPlanStatus.SAFE_DEGRADED:
        return "yellow"
    if plan_status.outcome in (
            GlobalPlanStatus.NOMINAL, GlobalPlanStatus.NO_ACTION):
        return "green"
    return "gray"


def _route_name(plan_status):
    if plan_status is None:
        return ""
    names = {
        GlobalPlanStatus.ROUTE_WHOLE_BODY_REFINED: "REFINED",
        GlobalPlanStatus.ROUTE_NONE: "",
    }
    return names.get(plan_status.route, "")


def _trajectory_state(ready, plan_status, label):
    if ready:
        return "READY", "green"
    if plan_status is not None and plan_status.outcome == GlobalPlanStatus.FAILED:
        return f"NOT PRODUCED ({label} failed)", "red"
    return "WAITING", "gray"


def _frontend_summary(status, global_ms):
    if status is None:
        return "WAITING", "gray"

    fields = _status_fields(status)
    lowered = status.lower()
    if "success" in lowered:
        details = ["SUCCESS"]
        if global_ms is not None:
            details.append(_number(global_ms, " ms"))
        if "expansions" in fields:
            details.append(f"{fields['expansions']} expansions")
        return " · ".join(details), "green"
    if "running" in lowered:
        return "RUNNING", "yellow"
    if "ready" in lowered:
        return "READY", "green"
    if "failure" in lowered or "failed" in lowered or "unavailable" in lowered:
        return _clip(status.replace("failure:", "FAILED: "), 58), "red"
    return _clip(status, 58), _shape_aware_astar_status_color(status)


def render_dashboard(model, agent, shape=None, now=None, color=True,
                     width=88, height=24):
    """Render a compact UAV and planning-pipeline status panel."""
    now = time.monotonic() if now is None else now
    width = max(20, width)
    height = max(5, height)
    shape_status = f" · SHAPE {shape.upper()}" if shape else ""

    state_age = None if model.state_at is None else max(0.0, now - model.state_at)
    online = state_age is not None and state_age < 2.0
    health = "● ONLINE" if online else "● WAITING"
    health_color = "green" if online else "red"

    if model.fsm is None:
        fsm_name, fsm_color = "WAITING FOR FSM", "gray"
    else:
        fsm_name, fsm_color = FSM_NAMES.get(
            model.fsm, (f"UNKNOWN({model.fsm})", "red"))

    if model.state is None:
        position = velocity = acceleration = yaw = "--"
    else:
        pos = model.state.pos
        vel = model.state.vel
        position = _vector_text((pos.x, pos.y, pos.z), "m")
        velocity = _vector_text((vel.x, vel.y, vel.z), "m/s")
        acceleration = (
            "--" if model.acceleration is None
            else _vector_text(model.acceleration, "m/s²"))
        yaw = _yaw_text(_yaw_from_quaternion(model.state.quat))

    latest = model.plan
    latest_ms = None if latest is None else float(latest.total_planning_ms)
    global_ms = None if latest is None else float(latest.frontend_planning_ms)
    local_ms = None if latest is None else float(latest.trajectory_optimization_ms)
    frontend, frontend_color = _frontend_summary(
        model.shape_aware_astar_status, global_ms)
    middle_ready = (
        model.middle_trajectory_ready or model.middle_stage_ms is not None)
    middle, middle_color = _trajectory_state(
        middle_ready, model.plan_status, "middle stage")
    backend, backend_color = _trajectory_state(
        model.backend_trajectory_ready, model.plan_status, "back-end")
    route = _route_name(model.plan_status)
    outcome = _outcome_name(model.plan_status)
    outcome_color = _outcome_color(model.plan_status)
    middle_detail = (
        f"{middle} · certified warm start · {_number(model.middle_stage_ms, ' ms')}"
        if middle_ready else middle)
    backend_detail = backend
    if model.backend_trajectory_ready:
        backend_detail += f" · {outcome}"
        if route:
            backend_detail += f" · {route}"
        backend_detail += f" · refine {_number(local_ms, ' ms')}"

    # Each tuple is (plain text, color, bold, dim). Color is applied only
    # after clipping, so ANSI escape bytes never affect width calculations.
    if width < 38 or height < 8:
        rows = [
            (f"MIGHTY · {agent}{shape_status}  {health}",
             health_color, True, False),
            (f"UAV  {fsm_name}", fsm_color, True, False),
            (f"POS  {position}", "white", False, False),
            (f"YAW  {yaw}", "cyan", False, False),
            (f"VEL  {velocity}", "green", False, False),
            (f"ACC* {acceleration}", "yellow", False, False),
            (f"F {frontend}", frontend_color, False, False),
            (f"M {middle_detail}", middle_color, False, False),
            (f"B {backend}", backend_color, False, False),
        ]
    elif width < 68 or height < 13:
        rows = [
            (f"MIGHTY · {agent}{shape_status}", "cyan", True, False),
            (f"{health}  ·  {fsm_name}", fsm_color if online else "red", True, False),
            (_section("UAV", width), "cyan", True, False),
            (f"POS    {position}", "white", False, False),
            (f"YAW    {yaw}", "cyan", False, False),
            (f"VEL    {velocity}", "green", False, False),
            (f"ACC*   {acceleration}", "yellow", False, False),
            (_section("PLAN", width), "magenta", True, False),
            (f"FRONT  {frontend}", frontend_color, False, False),
            (f"MIDDLE {middle_detail}", middle_color, False, False),
            (f"BACK   {backend}", backend_color, False, False),
        ]
    else:
        rows = [
            (f"MIGHTY STATUS · {agent}{shape_status}    {health}",
             "cyan", True, False),
            (_section("UAV STATE", width), "cyan", True, False),
            (f"FSM           {fsm_name}", fsm_color, True, False),
            (f"POSITION      {position}", "white", False, False),
            (f"YAW (att.)    {yaw}", "cyan", False, False),
            (f"VELOCITY      {velocity}", "green", False, False),
            (f"ACCEL (est.)  {acceleration}", "yellow", False, False),
            (_section("PLANNING", width), "magenta", True, False),
            (f"FRONT-END     {frontend}", frontend_color, True, False),
            (f"MIDDLE        {middle_detail}", middle_color, False, False),
            (f"BACK-END      {backend_detail}", backend_color, False, False),
            (f"PIPELINE      {outcome} · total {_number(latest_ms, ' ms')}",
             outcome_color, True, False),
        ]

    rows = rows[:max(1, height - 1)]
    return "\n".join(
        paint(_clip(text, width), row_color, bold=bold, dim=dim, enabled=color)
        for text, row_color, bold, dim in rows)


def compose_terminal_frame(screen, first_draw=False):
    """Build an in-place frame and erase stale suffixes on every old row."""
    prefix = "\033[2J\033[H\033[?25l" if first_draw else "\033[H"
    # EL (ESC[K) is required on each row. ED (ESC[J) at the end only clears
    # rows below the cursor and cannot remove characters left behind when a
    # newly rendered row is shorter than its previous contents.
    body = "\033[K\n".join(screen.splitlines())
    return prefix + body + "\033[K\033[J"


class DashboardNode:
    def __init__(self, agent, shape, refresh_hz, color):
        self.subscribers = []
        self.agent = agent.strip("/")
        self.shape = shape
        self.color = color
        self.model = DashboardModel()
        prefix = f"/{self.agent}"

        self.subscribers.append(rospy.Subscriber(f"{prefix}/state", State, self.model.update_state, queue_size=10))
        self.subscribers.append(rospy.Subscriber(f"{prefix}/fsm_state", UInt8, self.model.update_fsm, queue_size=10))
        self.subscribers.append(rospy.Subscriber(f"{prefix}/computation_times", ComputationTimes, self.model.update_plan, queue_size=10))
        self.subscribers.append(rospy.Subscriber(f"{prefix}/global_plan_status", GlobalPlanStatus, self.model.update_plan_status, queue_size=10))
        self.subscribers.append(rospy.Subscriber(f"{prefix}/shape_aware_astar/status", String, self.model.update_shape_aware_astar_status, queue_size=10))
        self.subscribers.append(rospy.Subscriber(f"{prefix}/middle_stage_time_ms", Float64, self.model.update_middle_stage_time, queue_size=10))
        self.subscribers.append(rospy.Subscriber(f"{prefix}/traj_middle", MarkerArray, self.model.update_middle_trajectory, queue_size=10))
        self.subscribers.append(rospy.Subscriber(f"{prefix}/traj_backend", MarkerArray, self.model.update_backend_trajectory, queue_size=10))
        self.first_draw = True


    def draw(self):
        size = shutil.get_terminal_size((88, 24))
        screen = render_dashboard(
            self.model, self.agent, shape=self.shape,
            color=self.color,
            width=size.columns, height=size.lines)
        frame = compose_terminal_frame(screen, first_draw=self.first_draw)
        self.first_draw = False
        sys.stdout.write(frame)
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="NX01")
    parser.add_argument("--shape", help="Selected visualization body shape")
    parser.add_argument("--refresh-hz", type=float, default=2.0)
    parser.add_argument("--no-color", action="store_true")
    args, ros_args = parser.parse_known_args()
    if not 0.2 <= args.refresh_hz <= 10.0:
        parser.error("--refresh-hz must be between 0.2 and 10")

    rospy.init_node("mighty_sim_dashboard", anonymous=True)
    node = DashboardNode(
        args.agent, args.shape, args.refresh_hz,
        color=sys.stdout.isatty() and not args.no_color)
    try:
        while not rospy.is_shutdown():
            node.draw()
            time.sleep(1.0 / args.refresh_hz)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()
        for sub in node.subscribers:
            sub.unregister()


if __name__ == "__main__":
    main()
