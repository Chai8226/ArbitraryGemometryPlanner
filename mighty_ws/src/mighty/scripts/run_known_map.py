#!/usr/bin/env python3
"""Launch single-UAV MIGHTY with a complete immutable PCD map."""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


SESSION_NAME = "mighty_known_map"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if not (PACKAGE_ROOT / "config/mighty.yaml").is_file():
    import rospkg
    PACKAGE_ROOT = Path(rospkg.RosPack().get_path("mighty"))
SHAPES_BY_SHAPE_SET = {
    "small": frozenset({"cross", "cuboid", "long1", "long2", "t", "v"}),
    "normal": frozenset({"cross", "cuboid", "t", "v"}),
}
SUPPORTED_SHAPE_AWARE_ASTAR_SHAPES = frozenset().union(
    *SHAPES_BY_SHAPE_SET.values())
SUPPORTED_KNOWN_MAPS = ("hole", "forest", "narrow", "block")


def fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(2)


def find_setup_bash(value: str | None) -> Path:
    value = value or os.environ.get("MIGHTY_SETUP_BASH")
    if value:
        path = Path(value).expanduser().absolute()
    else:
        candidates = [PACKAGE_ROOT.parent.parent / "devel" / "setup.bash"]
        candidates.extend(Path(p) / "setup.bash" for p in os.environ.get("CMAKE_PREFIX_PATH", "").split(":") if p and p != "/opt/ros/noetic")
        path = next((p for p in candidates if p.is_file()), candidates[0])
    if not path.is_file():
        fail(f"setup.bash not found: {path}")
    return path


def vector3(value, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        fail(f"{name} must be a YAML list with exactly three numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        fail(f"{name} contains a non-numeric value")
    if not all(math.isfinite(item) for item in result):
        fail(f"{name} must contain finite values")
    return result


def load_map_config(path: Path, map_id: str) -> dict:
    """Read the known_maps.<map_id> entry from the unified config file."""
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        fail(f"cannot read config {path}: {exc}")
    known_maps = document.get("known_maps") if isinstance(document, dict) else None
    if not isinstance(known_maps, dict):
        fail("config must contain a 'known_maps' mapping")
    cfg = known_maps.get(map_id)
    if not isinstance(cfg, dict):
        fail(f"known_maps has no entry for {map_id!r}")

    if map_id not in SUPPORTED_KNOWN_MAPS:
        fail(f"map id must be one of: {', '.join(SUPPORTED_KNOWN_MAPS)}")
    if cfg.get("id") != map_id:
        fail(f"known_maps.{map_id}.id must be {map_id!r}")
    shape_set = cfg.get("shape_set")
    if not isinstance(shape_set, str) or shape_set not in ("small", "normal", "real"):
        fail(f"known_maps.{map_id}.shape_set must be 'small', 'normal', or 'real'")

    pcd_value = cfg.get("pcd_path")
    if not isinstance(pcd_value, str) or not pcd_value:
        fail("known_map.pcd_path is required")
    pcd_path = Path(pcd_value).expanduser()
    if not pcd_path.is_absolute():
        pcd_path = (path.parent / pcd_path).resolve()
    if not pcd_path.is_file():
        fail(f"PCD file not found: {pcd_path}")

    try:
        resolution = float(cfg["resolution"])
    except (KeyError, TypeError, ValueError):
        fail("known_map.resolution must be numeric")
    if not math.isfinite(resolution) or resolution <= 0:
        fail("known_map.resolution must be finite and positive")

    bounds = cfg.get("bounds")
    if not isinstance(bounds, dict):
        fail("known_map.bounds must contain min and max")
    minimum = vector3(bounds.get("min"), "known_map.bounds.min")
    maximum = vector3(bounds.get("max"), "known_map.bounds.max")
    for axis, (lo, hi) in enumerate(zip(minimum, maximum)):
        if hi <= lo:
            fail(f"known_map bounds axis {axis} is not strictly increasing")

    frame_id = str(cfg.get("frame_id", "map")).strip()
    if not frame_id:
        fail("known_map.frame_id must not be empty")

    default_start = vector3(cfg.get("default_start", [0.0, 0.0, 1.0]),
                            "known_map.default_start")
    if not all(lo <= value < hi for value, lo, hi in
               zip(default_start, minimum, maximum)):
        fail("known_map.default_start must be inside the half-open bounds")

    visualization = cfg.get("visualization", {})
    if visualization is None:
        visualization = {}
    if not isinstance(visualization, dict):
        fail("visualization must be a mapping")

    polymap_value = visualization.get("polymap_path")
    if polymap_value is None:
        # This covers the normal package layout: maps/foo.pcd and
        # maps/polymap/foo.polymap.json.
        polymap_path = (
            pcd_path.parent / "polymap"
            / f"{pcd_path.stem}.polymap.json"
        ).resolve()
    elif isinstance(polymap_value, str) and polymap_value.strip():
        polymap_path = Path(polymap_value).expanduser()
        if not polymap_path.is_absolute():
            polymap_path = (path.parent / polymap_path).resolve()
        else:
            polymap_path = polymap_path.resolve()
    else:
        fail("visualization.polymap_path must be a non-empty string")

    boundary_solids = visualization.get("boundary_solids", [])
    if boundary_solids is None:
        boundary_solids = []
    if not isinstance(boundary_solids, list):
        fail("visualization.boundary_solids must be a list")
    boundary_entries = []
    seen_names = set()
    for entry in boundary_solids:
        if not isinstance(entry, dict):
            fail("every visualization.boundary_solids entry must be a mapping")
        name = entry.get("name")
        if (not isinstance(name, str) or not name or
                not re.fullmatch(r"[A-Za-z0-9_]+", name)):
            fail("boundary_solid name must be a non-empty identifier")
        if name in seen_names:
            fail(f"duplicate boundary_solid name {name!r}")
        seen_names.add(name)
        z_value = entry.get("z")
        if not isinstance(z_value, list) or len(z_value) != 2:
            fail(f"boundary_solid {name!r} z must contain exactly two numbers")
        try:
            z_lo, z_hi = (float(value) for value in z_value)
        except (TypeError, ValueError):
            fail(f"boundary_solid {name!r} z contains a non-numeric value")
        if not (math.isfinite(z_lo) and math.isfinite(z_hi) and z_hi > z_lo):
            fail(f"boundary_solid {name!r} z must be finite and increasing")
        if not (minimum[2] <= z_lo and z_hi <= maximum[2]):
            fail(f"boundary_solid {name!r} z must lie inside known_map bounds")
        boundary_entries.append((name, z_lo, z_hi))

    return {
        "map_id": map_id,
        "shape_set": shape_set,
        "pcd_path": pcd_path,
        "polymap_path": polymap_path,
        "resolution": resolution,
        "minimum": minimum,
        "maximum": maximum,
        "frame_id": frame_id,
        "default_start": default_start,
        "boundary_solids": boundary_entries,
    }


def _parse_extends(value: str, base_path: Path):
    """Parse extends reference 'file.yaml[:fragment/path]' into absolute path and keys."""
    if ":" in value:
        file_part, frag_part = value.split(":", 1)
    else:
        file_part, frag_part = value, ""
    file_path = Path(file_part).expanduser()
    if not file_path.is_absolute():
        file_path = (base_path.parent / file_path).resolve()
    else:
        file_path = file_path.resolve()
    fragments = [part for part in frag_part.split("/") if part]
    return file_path, fragments


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge mappings; override replaces scalars and sequences."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        existing = result.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_extends(path: Path, seen=None) -> dict:
    """Load a YAML file and recursively resolve its 'extends' references."""
    seen = set() if seen is None else set(seen)
    path = path.expanduser().resolve()
    if path in seen:
        fail(f"config extends cycle at {path}")
    seen.add(path)
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        fail(f"cannot read config {path}: {exc}")
    if not isinstance(document, dict):
        fail(f"config must be a mapping: {path}")

    extends = document.get("extends")
    if extends is None:
        return document
    if not isinstance(extends, str) or not extends:
        fail(f"config extends must be a non-empty string: {path}")

    base_path, fragments = _parse_extends(extends, path)
    base_doc = _resolve_extends(base_path, seen)
    for fragment in fragments:
        if not isinstance(base_doc, dict) or fragment not in base_doc:
            fail(f"config extends fragment {fragment!r} not found in {base_path}")
        base_doc = base_doc[fragment]

    return _deep_merge(base_doc, document)


def load_planner_parameters(path: Path) -> dict:
    """Load mighty_node.ros__parameters with extends resolution."""
    document = _resolve_extends(path)
    node = document.get("mighty_node")
    ros_parameters = node.get("ros__parameters") if isinstance(node, dict) else None
    if not isinstance(ros_parameters, dict):
        fail(f"planner config lacks mighty_node.ros__parameters: {path}")
    return ros_parameters


def selected_shape_aware_astar_shape(planner_config: Path,
                                     override: str | None = None) -> str:
    if override is not None:
        shape = override.strip().lower()
    else:
        parameters = load_planner_parameters(planner_config)
        shape = parameters.get("shape_aware_astar.shape", "t")
        if not isinstance(shape, str):
            fail("shape_aware_astar.shape must be a string")
        shape = shape.strip().lower()
    if shape not in SUPPORTED_SHAPE_AWARE_ASTAR_SHAPES:
        supported = ", ".join(sorted(SUPPORTED_SHAPE_AWARE_ASTAR_SHAPES))
        fail(f"shape must be one of: {supported}")
    return shape


def require_shape_for_shape_set(shape: str, shape_set: str) -> None:
    allowed = SHAPES_BY_SHAPE_SET.get(shape_set)
    if allowed is None:
        fail(f"unknown shape_set {shape_set!r}")
    if shape not in allowed:
        supported = ", ".join(sorted(allowed))
        fail(f"shape {shape!r} is not available in shape_set "
             f"{shape_set!r}; use one of: {supported}")


def require_inside(point, cfg: dict, label: str) -> None:
    if not all(lo <= value < hi for value, lo, hi in
               zip(point, cfg["minimum"], cfg["maximum"])):
        fail(f"{label} {tuple(point)} is outside the half-open map bounds")


def discover_underlays(explicit: list[str]) -> list[Path]:
    underlays = []
    for value in explicit:
        path = Path(value).expanduser().absolute()
        if not path.is_file():
            fail(f"underlay setup.bash not found: {path}")
        underlays.append(path)
    return underlays


def resolved_parameters(config, cfg, args):
    """Resolve the existing config schema to ROS1 node-private parameter trees."""
    params = load_planner_parameters(config)
    params.update({
        "planning_mode": "global_once", "map_mode": "known_pcd", "sim_env": "fake_sim",
        "map_frame_id": cfg["frame_id"], "known_map_pcd_path": str(cfg["pcd_path"]),
        "known_map_polymap_path": str(cfg["polymap_path"]),
        "known_map.id": cfg["map_id"], "known_map.shape_set": cfg["shape_set"],
        "known_map.resolution": cfg["resolution"],
        "known_map.boundary_solids": ",".join(f"{n}:{lo}:{hi}" for n, lo, hi in cfg["boundary_solids"]),
        "shape_aware_astar.shape": cfg["shape_aware_astar_shape"],
    })
    for i, axis in enumerate("xyz"):
        params[axis + "_min"] = cfg["minimum"][i]
        params[axis + "_max"] = cfg["maximum"][i]
    return {"mighty_node": {k.replace(".", "/"): v for k, v in params.items()},
            "fake_sim": {"start_pos": list(args.start), "start_yaw": math.radians(args.start_yaw),
                         "map_frame_id": cfg["frame_id"], "visual_level": 1}}


def render_rviz_config(namespace, frame_id, output_path, overwrite=False):
    source = output_path if output_path.is_file() and not overwrite else PACKAGE_ROOT / "rviz/mighty.rviz"
    content = source.read_text()
    content = re.sub(r"(?m)^(\s*(?:Topic|Marker Topic): )/[^/\s]+/", lambda m: m[1] + "/" + namespace + "/", content)
    content = re.sub(r"(?m)^(\s*Fixed Frame: ).*$", lambda m: m[1] + frame_id, content)
    if not output_path.is_file() or output_path.read_text() != content:
        output_path.write_text(content)
    return output_path


def command_prefix(setup, underlays, master, log_dir):
    q = shlex.quote
    return [
        "export PATH=/usr/bin:/bin",
        "unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH ROS_PACKAGE_PATH QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH",
        "source /opt/ros/noetic/setup.bash",
        *["source " + q(str(p)) for p in underlays],
        "source " + q(str(setup)),
        "export ROS_MASTER_URI=" + q(master),
        "export MIGHTY_SIM_LOG_DIR=" + q(str(log_dir)),
        "stty -ixon 2>/dev/null || true",
    ]


def launch_yaml(args, cfg, setup, underlays, params_file, rviz_config, log_dir):
    q = shlex.quote
    onboard = ("roslaunch mighty known_map.launch namespace:=" + q(args.namespace)
               + " params_file:=" + q(str(params_file)))
    def pane(command, name):
        # Exec leaves roslaunch directly parented to the tmux pane.
        return {"shell_command": ["exec " + command + " > " + q(str(log_dir / (name + ".log"))) + " 2>&1"]}
    panes = [pane(onboard, args.namespace)]
    if rviz_config:
        panes.append(pane("rviz -d " + q(str(rviz_config)), "rviz"))
    if args.goal:
        goal = "rosrun mighty send_goal.py --namespace " + q(args.namespace)
        goal += " --frame " + q(cfg["frame_id"]) + " --goal " + " ".join(map(str, args.goal))
        if args.goal_yaw is not None:
            goal += " --yaw " + str(args.goal_yaw)
        panes.append(pane(goal, "goal"))
    return yaml.safe_dump({"session_name": SESSION_NAME,
        "options": {"mouse": "on"},
        "shell_command_before": command_prefix(setup, underlays, args.ros_master_uri, log_dir),
        "windows": [{"window_name": "known-map", "layout": "tiled", "panes": panes},
                    {"window_name": "status", "panes": [{"shell_command": [
                        "exec rosrun mighty sim_dashboard.py --agent " + q(args.namespace)
                        + " --shape " + q(cfg["shape_aware_astar_shape"])]}]}]}, sort_keys=False)


def check_existing(args):
    """Never kill unrelated nodes or replace an active ROS1 node registration."""
    from xmlrpc.client import ServerProxy, Fault, ProtocolError
    import socket
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(2)
    try:
        master = ServerProxy(args.ros_master_uri)
        for node in ("mighty_node", "fake_sim"):
            name = "/" + args.namespace + "/" + node
            code, _, uri = master.lookupNode("/mighty_launch_check", name)
            if code != 1:
                continue
            try:
                ServerProxy(uri).getPid("/mighty_launch_check")
            except ConnectionRefusedError:
                # A new node registration replaces this dead node's URI.
                continue
            except (OSError, Fault, ProtocolError) as error:
                fail("cannot verify registered node: " + name + "; " + str(error))
            fail("node already running on selected ROS master: " + name)
    except (OSError, Fault, ProtocolError):
        pass  # roslaunch starts a local master if none is available
    finally:
        socket.setdefaulttimeout(previous)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MIGHTY with a complete immutable obstacle PCD")
    parser.add_argument("--config", type=Path,
                        default=PACKAGE_ROOT / "config" / "mighty.yaml",
                        help="unified config file (planner params, known maps, "
                             "build profiles)")
    parser.add_argument("--map", required=True,
                        choices=list(SUPPORTED_KNOWN_MAPS),
                        help="known map id (hole, forest, narrow, block)")
    parser.add_argument(
        "--shape",
        choices=sorted(SUPPORTED_SHAPE_AWARE_ASTAR_SHAPES),
        help="body shape for Shape-aware Weighted A* "
             "(default: shape_aware_astar.shape from --config; "
             "hole/small supports long1/long2/v, forest|narrow/normal do not support long1/long2)")
    parser.add_argument("--setup-bash")
    parser.add_argument(
        "--underlay-bash", action="append", default=[],
        help="Additional setup.bash to source first; may be repeated.")
    parser.add_argument("--start", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--start-yaw", type=float, default=0.0,
                        help="Initial yaw in degrees")
    parser.add_argument("--goal", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--goal-yaw", type=float,
        help="Optional terminal yaw in degrees; requires --goal")
    parser.add_argument("--namespace", default="NX01")
    parser.add_argument("--ros-master-uri", default=os.environ.get("ROS_MASTER_URI", "http://localhost:11311"))
    parser.add_argument(
        "--rviz-config", type=Path,
        help="Use this persistent RViz config instead of the map YAML's sibling .rviz file")
    parser.add_argument(
        "--reset-rviz-config", action="store_true",
        help="Regenerate the default persistent RViz config from rviz/mighty.rviz")
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument(
        "--kill-stale", action="store_true",
        help="SIGINT any mighty/fake_sim process already running on this "
             "project session instead of refusing to start")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--detached", action="store_true",
        help="Create the tmux session without attaching (for automation).")
    return parser.parse_args()


def main():
    args = parse_args()
    config = args.config.expanduser().resolve()
    cfg = load_map_config(config, args.map)
    args.start = cfg["default_start"] if args.start is None else args.start
    require_inside(args.start, cfg, "start")
    if args.goal:
        require_inside(args.goal, cfg, "goal")
    if args.goal_yaw is not None and (not args.goal or not math.isfinite(args.goal_yaw)):
        fail("--goal-yaw requires --goal and must be finite")
    if not math.isfinite(args.start_yaw):
        fail("--start-yaw must be finite")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", args.namespace):
        fail("namespace must be a single ROS name")
    cfg["shape_aware_astar_shape"] = selected_shape_aware_astar_shape(config, args.shape)
    require_shape_for_shape_set(cfg["shape_aware_astar_shape"], cfg["shape_set"])
    if not cfg["polymap_path"].is_file():
        fail("PolyMap artifact not found: " + str(cfg["polymap_path"]))
    params = resolved_parameters(config, cfg, args)
    if args.dry_run:
        print(yaml.safe_dump(params, sort_keys=False))
        return
    setup = find_setup_bash(args.setup_bash)
    underlays = discover_underlays(args.underlay_bash)
    if not shutil.which("tmux") or not shutil.which("tmuxp"):
        fail("install tmux and tmuxp before running")
    exists = subprocess.run(["tmux", "has-session", "-t", SESSION_NAME],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if exists:
        if not args.kill_stale:
            fail("project session already running; use --kill-stale to replace it")
        owner = subprocess.run(["tmux", "show-option", "-v", "-t", SESSION_NAME, "@mighty_project"],
                               capture_output=True, text=True).stdout.strip()
        if owner != str(PACKAGE_ROOT):
            fail("existing tmux session is not owned by this checkout; refusing to terminate it")
        subprocess.run(["tmux", "kill-session", "-t", SESSION_NAME], check=True)
        time.sleep(3)  # only our session is terminated; allow ROS registrations to drain
    check_existing(args)
    log_dir = Path(tempfile.mkdtemp(prefix="mighty_noetic_"))
    params_file = log_dir / "parameters.yaml"
    params_file.write_text(yaml.safe_dump(params, sort_keys=False))
    rviz_config = None
    if not args.no_rviz:
        if args.rviz_config and args.reset_rviz_config:
            fail("--rviz-config cannot be combined with --reset-rviz-config")
        rviz_config = (args.rviz_config or PACKAGE_ROOT / "config" / ("known_map." + args.map + ".rviz")).resolve()
        if args.rviz_config and not rviz_config.is_file():
            fail("RViz config not found: " + str(rviz_config))
        render_rviz_config(args.namespace, cfg["frame_id"], rviz_config, args.reset_rviz_config)
    session = log_dir / "session.yaml"
    session.write_text(launch_yaml(args, cfg, setup, underlays, params_file, rviz_config, log_dir))
    print("[INFO] ROS master: " + args.ros_master_uri, flush=True)
    print("[INFO] Logs: " + str(log_dir), flush=True)
    command = ["tmuxp", "load", "-y", "-d", str(session)]
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:/bin"
    result = subprocess.run(command, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)
    subprocess.run(["tmux", "set-option", "-t", SESSION_NAME,
                    "@mighty_project", str(PACKAGE_ROOT)], check=True)
    if not args.detached:
        raise SystemExit(subprocess.run(["tmux", "attach-session", "-t", SESSION_NAME], env=env).returncode)


if __name__ == "__main__":
    main()
