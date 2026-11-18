# Third-party notices

The combined release is distributed under GPL-3.0-only; see [LICENSE](LICENSE). This does not replace the original terms or copyright notices of incorporated components. Retained notices in individual source files continue to apply.

## MIGHTY and development provenance

The planner derives from [MIT ACL MIGHTY](https://github.com/mit-acl/mighty), through [Shape-Traj](https://github.com/Chai8226/Shape-Traj).

The upstream MIGHTY license is BSD-3-Clause, copyright (c) 2024 Kota Kondo. Its complete notice is preserved in [LICENSES/MIGHTY-BSD-3-Clause.txt](LICENSES/MIGHTY-BSD-3-Clause.txt). Several retained source headers additionally state copyright 2025 Kota Kondo, Aerospace Controls Laboratory, with references to the project license. Those authorship notices are preserved.

Retained runtime additions include GPL-3.0-only SPDX notices, including `mighty_main.cpp`, `runtime_guards.hpp`, `polymap/artifact_format.hpp`, and `polymap/visualization.hpp`. The package metadata for the combined planner uses GPL-3.0-only instead of the development package's inconsistent MIT entry.

The two retained PolyMap runtime headers were adapted from the Shape-PolyMap `plan_env` implementation in July 2026, under GPL version 3. The construction implementation is excluded from this release.

The `dynus_interfaces` package declares MIT in its own package metadata, with Kota Kondo as maintainer. That declaration and the [MIT license terms](LICENSES/MIT.txt) are retained separately; the combined repository license does not erase component-specific terms. This inventory is not an independent authorship or license audit of every message definition.

## Small-dimensional linear programming

`mighty_ws/src/mighty/include/mighty/third_party/sdlp.hpp` retains these notices:

```text
Copyright (c) 1990 Michael E. Hohmeyer,
      hohmeyer@icemcfd.com
Permission is granted to modify and re-distribute this code in any manner
as long as this notice is preserved. All standard disclaimers apply.

R. Seidel's algorithm for solving LPs (linear programs.)

Copyright (c) 2021 Zhepei Wang,
      wangzhepei@live.com
1. Bug fix in "move_to_front" function that "prev[m]" is illegally accessed
   while "prev" originally has only m ints. It is fixed by allocating a
   "prev" with m + 1 ints.
2. Add Eigen interface.
3. Resursive template.
Permission is granted to modify and re-distribute this code in any manner
as long as this notice is preserved. All standard disclaimers apply.
```

The source cites Seidel, R. (1991), “Small-dimensional linear programming and convex hulls made easy,” *Discrete & Computational Geometry* 6:423–434, doi:10.1007/BF02574699.

## Downloaded build dependencies

The build script downloads these upstream releases; their sources and compiled binaries are not checked into this repository:

| Component | Version | Upstream license notice |
| --- | --- | --- |
| [Flexible Collision Library](https://github.com/flexible-collision-library/fcl/tree/0.7.0) | 0.7.0 | BSD license; copyright (c) 2008–2014 Willow Garage, Inc.; copyright (c) 2014–2016 Open Source Robotics Foundation |
| [JSON for Modern C++](https://github.com/nlohmann/json/tree/v3.10.5) | 3.10.5 | MIT; copyright (c) 2013–2022 Niels Lohmann |

Refer to the downloaded FCL `LICENSE` and JSON `LICENSE.MIT` files for their full terms. Preserve those notices when redistributing dependency sources or binaries. System dependencies, including ROS, Eigen, PCL, libccd, and OctoMap, retain their respective licenses; this file is not a complete inventory of transitive dependencies.
