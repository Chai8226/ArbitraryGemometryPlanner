#!/usr/bin/env bash
set -eo pipefail
jobs=4
if [[ ${1:-} == -j || ${1:-} == --jobs ]]; then jobs="${2:?missing job count}";
elif [[ $# != 0 ]]; then echo "usage: $0 [-j JOBS]" >&2; exit 2; fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mighty_workspace="$(cd "$script_dir/../.." && pwd)"
# Conda exports compiler and package paths even when python is pinned.
export PATH=/usr/bin:/bin
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH PKG_CONFIG_PATH PYTHONPATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH
source /opt/ros/noetic/setup.bash
fcl_prefix="$mighty_workspace/.deps/noetic"
if [[ ! -f "$fcl_prefix/lib/cmake/fcl/fcl-config.cmake" || ! -f "$fcl_prefix/lib/cmake/nlohmann_json/nlohmann_jsonConfig.cmake" ]]; then
  "$script_dir/scripts/build_noetic_fcl.sh" -j "$jobs"
fi
cd "$mighty_workspace"
catkin config --profile default --extend /opt/ros/noetic --source-space src \
  --build-space build --devel-space devel --install-space install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ "-Dfcl_DIR=$fcl_prefix/lib/cmake/fcl" "-Dnlohmann_json_DIR=$fcl_prefix/lib/cmake/nlohmann_json"
catkin profile set default
catkin build --profile default dynus_interfaces mighty -j "$jobs" -p 1 --no-status
printf 'Build complete. Run: source %s/devel/setup.bash\n' "$mighty_workspace"
