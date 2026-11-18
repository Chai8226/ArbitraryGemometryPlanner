#!/usr/bin/env bash
# Run ROS1 commands with system Python/libraries, even from an active Conda shell.
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mighty_workspace="$(cd "$script_dir/../../.." && pwd)"
export PATH=/usr/bin:/bin
unset PYTHONPATH LD_LIBRARY_PATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH
source /opt/ros/noetic/setup.bash
mighty_setup="${MIGHTY_SETUP_BASH:-$mighty_workspace/devel/setup.bash}"
if [[ ! -f "$mighty_setup" ]]; then echo "Build first: bash $script_dir/../setup.sh" >&2; exit 2; fi
source "$mighty_setup"
export MIGHTY_SETUP_BASH="$mighty_setup"
exec "$@"
