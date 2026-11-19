# Arbitrary Geometry Planner

Known-map trajectory planning for non-spherical bodies, jointly planning three-dimensional position and yaw. This release contains the shape-aware search frontend, trajectory optimization backend, ROS1 simulation, RViz interface, and prepared map and body geometry.

PolyMap construction, geometry generation, competing planners, real-world experiment assets, and manuscript sources are outside this release. The supplied final maps and shapes can be loaded directly; no map-building step is required.

![ROS1 RViz demo: narrow map with planned trajectory and swept body geometry](docs/images/demo.png)

![Trajectory planning through scattered obstacles](docs/images/1.gif)

![Trajectory planning through an obstacle field](docs/images/2.gif)

![Trajectory planning through a grid of openings](docs/images/3.gif)

## Requirements and build

Use Ubuntu 20.04, ROS1 Noetic, and system Python 3.8. Install ROS Noetic before running the following commands. Run all commands from the repository root.

```bash
git clone https://github.com/Chai8226/ArbitraryGemometryPlanner.git
cd ArbitraryGemometryPlanner
sudo apt-get update
sudo apt-get install build-essential cmake curl python3-catkin-tools python3-yaml \
  ros-noetic-roscpp ros-noetic-rospy ros-noetic-roslib ros-noetic-roslaunch \
  ros-noetic-rviz ros-noetic-pcl-conversions ros-noetic-tf2-geometry-msgs \
  ros-noetic-message-generation libeigen3-dev libpcl-dev \
  libccd-dev liboctomap-dev tmux tmuxp
bash mighty_ws/src/mighty/setup.sh -j 4
```

The setup script builds `dynus_interfaces` and `mighty` with system compilers and Python. On first use it downloads and builds FCL 0.7.0 and nlohmann/json 3.10.5; this requires network access. Generated dependencies and build products stay under `mighty_ws/` and are excluded from Git.

## Run

```bash
bash mighty_ws/src/mighty/scripts/noetic_env.sh \
  /usr/bin/python3 mighty_ws/src/mighty/scripts/run_known_map.py \
  --map hole --shape t
```

In RViz, use **2D Nav Goal** to choose a goal. The shipped planner configuration fixes goal height at 1 m (`force_goal_z: true`); edit `mighty_ws/src/mighty/config/params/mighty_node.yaml` to use the requested goal height instead. RViz goal orientation specifies terminal yaw.

| Map | Shape set | Available `--shape` values |
| --- | --- | --- |
| `hole` | `small` | `cuboid`, `cross`, `t`, `v`, `long1`, `long2` |
| `forest`, `narrow`, `block` | `normal` | `cuboid`, `cross`, `t`, `v` |

Run a specified task without RViz:

```bash
bash mighty_ws/src/mighty/scripts/noetic_env.sh \
  /usr/bin/python3 mighty_ws/src/mighty/scripts/run_known_map.py \
  --map narrow --shape t --start 0 0 1 --start-yaw 0 \
  --goal 2.27 -0.11 1 --no-rviz --detached
```

Coordinates are in metres. `--start-yaw` and `--goal-yaw` are in degrees; `--goal-yaw` requires `--goal`. Omitting `--goal-yaw` leaves terminal yaw to the planner. The default initial yaw is zero. The simulator follows the resulting position-and-yaw trajectory; this release runs in `global_once + known_pcd + fake_sim` mode.

The launcher creates the `mighty_known_map` tmux session. Use `tmux attach -t mighty_known_map` to inspect it and `tmux kill-session -t mighty_known_map` to stop it. A pre-existing session is rejected unless explicitly replaced with `--kill-stale`; replacement checks ownership by checkout.

## Configuration and assets

- `mighty_ws/src/mighty/config/mighty.yaml`: map, shape, task, and vehicle selections.
- `mighty_ws/src/mighty/config/params/mighty_node.yaml`: search, optimization, and motion constraints.
- [Asset format and coordinate conventions](docs/assets.md): final map and body files consumed by the runtime.

## License and provenance

The combined release is distributed under [GPL-3.0-only](LICENSE). It builds on [MIT ACL MIGHTY](https://github.com/mit-acl/mighty) and the [Shape-Traj development repository](https://github.com/Chai8226/Shape-Traj). Original copyright notices and applicable third-party terms remain in force; see [third-party notices](THIRD_PARTY_NOTICES.md) and [the upstream BSD license](LICENSES/MIGHTY-BSD-3-Clause.txt).
