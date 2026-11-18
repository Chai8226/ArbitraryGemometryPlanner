#!/usr/bin/env bash
set -euo pipefail
jobs=4
if [[ ${1:-} == -j ]]; then jobs="${2:?missing job count}"; fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
deps="$(cd "$script_dir/../../.." && pwd)/.deps"
mkdir -p "$deps/downloads"
archive="$deps/downloads/fcl-0.7.0.tar.gz"
if [[ ! -f "$archive" ]]; then
  curl -fL --retry 3 https://codeload.github.com/flexible-collision-library/fcl/tar.gz/refs/tags/0.7.0 -o "$archive.part"
  mv "$archive.part" "$archive"
fi
printf '%s  %s\n' 90409e940b24045987506a6b239424a4222e2daf648c86dd146cbcb692ebdcbc "$archive" | sha256sum --check --status
if [[ ! -d "$deps/fcl-0.7.0" ]]; then tar -xzf "$archive" -C "$deps"; fi
export PATH=/usr/bin:/bin
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH PKG_CONFIG_PATH CMAKE_PREFIX_PATH
cmake -S "$deps/fcl-0.7.0" -B "$deps/fcl-build-system" \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$deps/noetic" \
  -DPKG_CONFIG_EXECUTABLE=/usr/bin/pkg-config -DFCL_BUILD_TESTS=OFF -DFCL_BUILD_EXAMPLES=OFF
cmake --build "$deps/fcl-build-system" -j "$jobs"
cmake --install "$deps/fcl-build-system"

json_archive="$deps/downloads/json-3.10.5.tar.gz"
if [[ ! -f "$json_archive" ]]; then
  curl -fL --retry 3 https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.10.5 -o "$json_archive.part"
  mv "$json_archive.part" "$json_archive"
fi
printf '%s  %s\n' 5daca6ca216495edf89d167f808d1d03c4a4d929cef7da5e10f135ae1540c7e4 "$json_archive" | sha256sum --check --status
if [[ ! -d "$deps/json-3.10.5" ]]; then tar -xzf "$json_archive" -C "$deps"; fi
cmake -S "$deps/json-3.10.5" -B "$deps/json-build" -DJSON_BuildTests=OFF -DCMAKE_INSTALL_PREFIX="$deps/noetic"
cmake --install "$deps/json-build"
