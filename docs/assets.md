# Runtime geometry assets

This repository supplies final geometry for runtime loading. It does not include PolyMap construction, decomposition, or raw shape-generation tools.

Paths below are relative to `mighty_ws/src/mighty/`.

| Asset | Location | Purpose |
| --- | --- | --- |
| Obstacle point clouds | `maps/{hole,forest,narrow,block}.pcd` | Known-map point-cloud input and display |
| Obstacle convex geometry | `maps/polymap/<map>.polymap.json` | Collision geometry, optimization geometry, and map surface display |
| Body convex geometry | `shape/{small,normal}/ply/<shape>.polymap.json` | Shape-aware collision checking and trajectory optimization |
| Body surface mesh | `shape/{small,normal}/ply/<shape>.ply` | Body wireframe visualization |
| Quadrotor display model | `meshes/quadrotor/` | Simulator display marker, separate from collision geometry |

The `hole` map selects `small` shapes: `cuboid`, `cross`, `t`, `v`, `long1`, and `long2`. The other three maps select `normal` shapes: `cuboid`, `cross`, `t`, and `v`. These are separate prepared geometries, not runtime scale settings.

## Coordinates and pose

All geometry coordinates and distances are in metres. Map vertices are expressed in the `map` frame and body vertices in the `body` frame. Runtime placement is

```text
world_vertex = position + Rz(yaw) * body_vertex
```

Yaw is a rotation about positive z; positive yaw rotates positive x toward positive y. The planner models translation and yaw, without planning roll or pitch. Internal yaw values use radians; the launcher accepts start and goal yaw in degrees. Do not recenter or rescale a body mesh independently of its JSON artifact.

## PolyMap JSON version 2

The runtime accepts `format: "mighty.polymap"` and `format_version: 2`. The principal fields are:

| Field | Meaning |
| --- | --- |
| `kind` | `"map"` or `"shape"` |
| `name` | Must match the basename before `.polymap.json` |
| `frame_id` | `"map"` for maps, `"body"` for shapes |
| `grid.min`, `grid.max` | Three-element grid bounds; map bounds must be ordered and match runtime configuration |
| `grid.resolution` | Map resolution, checked against configuration |
| `polyhedra` | Nonempty array of convex components |
| `polyhedra[].vertices` | Finite `[x, y, z]` vertices |
| `polyhedra[].faces` | Polygon faces, each with at least three zero-based vertex indices |
| `polyhedra[].halfspaces` | Rows `[a, b, c, d]` describing `a*x + b*y + c*z <= d` |
| `polyhedra[].triangles` | Zero-based triangle vertex indices used for map surface display |
| `summary.polyhedra` | Number of components, checked by map visualization loading |

Each collision component needs at least four vertices and four faces. Halfspace normals must be finite and nonzero; the loader normalizes normals and offsets together. The configured body clearance is applied by the runtime rather than by modifying stored vertices.

Map identity, frame, resolution, bounds, component count, and triangle indices are also validated by map visualization loading. A map JSON name must agree with both its `.polymap.json` basename and the associated PCD basename. The supplied assets and configuration form a matched set.

Some supplied JSON fields describe construction provenance or diagnostics. The runtime ignores construction fields such as `source_pcd`, `pcd_to_frame`, `decomposition`, `input_partition`, `solidify_closed_shell`, `ads`, and `source_voxels`. Their presence does not make the excluded construction pipeline available. This document describes the runtime contract, not a standalone geometry builder or a guarantee that arbitrary replacement geometry is valid.
