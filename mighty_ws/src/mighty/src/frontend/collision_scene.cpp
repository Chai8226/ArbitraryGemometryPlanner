#include "mighty/frontend/collision_scene.hpp"

#include <mighty/polymap/artifact_format.hpp>
#include "mighty/pose_interpolation.hpp"

#include <fcl/fcl.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace mighty::frontend {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
// Search edges are validated before insertion, so the lowest-cost label is
// sufficient for a Pose. Shortcut validation remains deferred to the final
// path simplifier and does not change reachability of the search tree.
constexpr std::size_t kMaxPathLabelsPerPose = 1;

bool finite(double value) { return std::isfinite(value); }

bool finite(const Eigen::Vector3d& value) {
  return value.array().isFinite().all();
}

double wrapAngle(double angle) {
  return std::remainder(angle, kTwoPi);
}

double positiveAngle(double angle) {
  double result = std::fmod(angle, kTwoPi);
  if (result < 0.0) result += kTwoPi;
  return result;
}

double angleDifference(double target, double source) {
  return wrapAngle(target - source);
}

Eigen::Vector3d jsonVector3(const nlohmann::json& value,
                            const std::string& field) {
  if (!value.is_array() || value.size() != 3) {
    throw std::runtime_error(field + " must contain exactly three numbers");
  }
  Eigen::Vector3d result(value[0].get<double>(), value[1].get<double>(),
                         value[2].get<double>());
  if (!finite(result)) {
    throw std::runtime_error(field + " contains a non-finite value");
  }
  return result;
}

std::string expectedArtifactName(const std::string& path) {
  std::string filename = path;
  const auto slash = filename.find_last_of("/\\");
  if (slash != std::string::npos) filename = filename.substr(slash + 1);
  constexpr const char* suffix = ".polymap.json";
  const std::size_t suffix_length = std::char_traits<char>::length(suffix);
  if (filename.size() <= suffix_length ||
      filename.compare(filename.size() - suffix_length, suffix_length,
                       suffix) != 0) {
    throw std::runtime_error(
        "PolyMap artifact filename must end in .polymap.json: " + path);
  }
  return filename.substr(0, filename.size() - suffix_length);
}

struct ConvexComponent {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::shared_ptr<const std::vector<fcl::Vector3d>> fcl_vertices;
  std::shared_ptr<const std::vector<int>> fcl_faces;
  std::shared_ptr<fcl::Convexd> geometry;
  std::vector<Eigen::Vector3d> face_normals;
  std::vector<Eigen::Vector3d> edge_directions;
  Eigen::Vector3d local_min{Eigen::Vector3d::Constant(kInfinity)};
  Eigen::Vector3d local_max{Eigen::Vector3d::Constant(-kInfinity)};

  Eigen::Vector3d supportVertex(const Eigen::Vector3d& direction) const {
    double best = -kInfinity;
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    for (const auto& vertex : *fcl_vertices) {
      const double value = direction.dot(vertex);
      if (value > best) {
        best = value;
        result = vertex;
      }
    }
    return result;
  }
};

void prepareSatAxes(ConvexComponent& component) {
  component.local_min = Eigen::Vector3d::Constant(kInfinity);
  component.local_max = Eigen::Vector3d::Constant(-kInfinity);
  for (const auto& vertex : *component.fcl_vertices) {
    component.local_min = component.local_min.cwiseMin(vertex);
    component.local_max = component.local_max.cwiseMax(vertex);
  }
  component.edge_directions.clear();
  for (std::size_t first = 0; first < component.face_normals.size(); ++first) {
    for (std::size_t second = first + 1;
         second < component.face_normals.size(); ++second) {
      Eigen::Vector3d direction = component.face_normals[first].cross(
          component.face_normals[second]);
      const double norm = direction.norm();
      if (norm <= 1.0e-10) continue;
      direction /= norm;
      bool duplicate = false;
      for (const auto& kept : component.edge_directions) {
        if (std::abs(direction.dot(kept)) > 1.0 - 1.0e-9) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) component.edge_directions.push_back(direction);
    }
  }
}

struct LoadedPolyMap {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::string name;
  std::string frame;
  std::string kind;
  Eigen::Vector3d grid_min{Eigen::Vector3d::Zero()};
  Eigen::Vector3d grid_max{Eigen::Vector3d::Zero()};
  std::vector<ConvexComponent> components;
};

LoadedPolyMap loadPolyMap(const std::string& path,
                          const std::string& expected_kind,
                          double tolerance,
                          double outward_offset = 0.0) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("failed to open PolyMap: " + path);

  const auto keep_runtime_fields =
      [](int, nlohmann::json::parse_event_t event, nlohmann::json& parsed) {
        if (event != nlohmann::json::parse_event_t::key) return true;
        const std::string key = parsed.get<std::string>();
        return key != "source_voxels" && key != "triangles" &&
               key != "ads" &&
               key != "source_pcd" && key != "pcd_to_frame" &&
               key != "decomposition" && key != "input_partition" &&
               key != "solidify_closed_shell";
      };
  const nlohmann::json root =
      nlohmann::json::parse(input, keep_runtime_fields);
  if (root.at("format").get<std::string>() != "mighty.polymap" ||
      root.at("format_version").get<int>() !=
          mighty::polymap::kArtifactFormatVersion) {
    throw std::runtime_error("unsupported PolyMap format: " + path);
  }

  LoadedPolyMap result;
  result.kind = root.at("kind").get<std::string>();
  result.name = root.at("name").get<std::string>();
  result.frame = root.at("frame_id").get<std::string>();
  if (result.kind != expected_kind) {
    throw std::runtime_error("PolyMap kind mismatch for " + path);
  }
  if (result.name != expectedArtifactName(path)) {
    throw std::runtime_error("PolyMap name/filename mismatch for " + path);
  }
  if ((expected_kind == "map" && result.frame != "map") ||
      (expected_kind == "shape" && result.frame != "body")) {
    throw std::runtime_error("PolyMap frame mismatch for " + path);
  }

  if (expected_kind == "map") {
    const auto& grid = root.at("grid");
    result.grid_min = jsonVector3(grid.at("min"), "grid.min");
    result.grid_max = jsonVector3(grid.at("max"), "grid.max");
    if ((result.grid_max.array() <= result.grid_min.array()).any()) {
      throw std::runtime_error("PolyMap grid bounds are empty or inverted");
    }
  }

  const auto& polyhedra = root.at("polyhedra");
  if (!polyhedra.is_array() || polyhedra.empty()) {
    throw std::runtime_error("PolyMap has no convex polyhedra: " + path);
  }
  result.components.reserve(polyhedra.size());
  for (std::size_t component_index = 0;
       component_index < polyhedra.size(); ++component_index) {
    const auto& encoded = polyhedra[component_index];
    const auto& encoded_vertices = encoded.at("vertices");
    const auto& encoded_faces = encoded.at("faces");
    if (!encoded_vertices.is_array() || encoded_vertices.size() < 4 ||
        !encoded_faces.is_array() || encoded_faces.size() < 4) {
      throw std::runtime_error("invalid convex component " +
                               std::to_string(component_index));
    }

    auto vertices = std::make_shared<std::vector<fcl::Vector3d>>();
    vertices->reserve(encoded_vertices.size());
    ConvexComponent component;
    for (const auto& encoded_vertex : encoded_vertices) {
      const Eigen::Vector3d vertex =
          jsonVector3(encoded_vertex, "polyhedra[].vertices");
      vertices->push_back(vertex);
    }

    auto faces = std::make_shared<std::vector<int>>();
    for (const auto& face : encoded_faces) {
      if (!face.is_array() || face.size() < 3) {
        throw std::runtime_error("invalid face in " + path);
      }
      faces->push_back(static_cast<int>(face.size()));
      for (const auto& encoded_index : face) {
        const int vertex_index = encoded_index.get<int>();
        if (vertex_index < 0 ||
            static_cast<std::size_t>(vertex_index) >= vertices->size()) {
          throw std::runtime_error("invalid face index in " + path);
        }
        faces->push_back(vertex_index);
      }
    }

    std::vector<Eigen::Vector3d> halfspace_normals;
    std::vector<double> halfspace_offsets;
    for (const auto& encoded_halfspace : encoded.at("halfspaces")) {
      if (!encoded_halfspace.is_array() ||
          encoded_halfspace.size() != 4) {
        throw std::runtime_error("invalid halfspace in " + path);
      }
      Eigen::Vector4d halfspace(
          encoded_halfspace[0].get<double>(),
          encoded_halfspace[1].get<double>(),
          encoded_halfspace[2].get<double>(),
          encoded_halfspace[3].get<double>());
      if (!halfspace.array().isFinite().all() ||
          halfspace.head<3>().norm() <= tolerance) {
        throw std::runtime_error("invalid halfspace normal in " + path);
      }
      const double normal_norm = halfspace.head<3>().norm();
      halfspace_normals.push_back(halfspace.head<3>() / normal_norm);
      halfspace_offsets.push_back(halfspace[3] / normal_norm +
                                  outward_offset);
    }

    // Rebuild an outward-offset component directly from its H-form.  Runtime
    // Shape inflation uses this path so frontend FCL geometry exactly matches
    // the backend's normalized half-space safety envelope.
    if (outward_offset > 0.0) {
      vertices = std::make_shared<std::vector<fcl::Vector3d>>();
      const double feasibility_tolerance = std::max(1.0e-8, tolerance * 10.0);
      for (std::size_t i = 0; i < halfspace_normals.size(); ++i) {
        for (std::size_t j = i + 1; j < halfspace_normals.size(); ++j) {
          for (std::size_t k = j + 1; k < halfspace_normals.size(); ++k) {
            Eigen::Matrix3d matrix;
            matrix.row(0) = halfspace_normals[i];
            matrix.row(1) = halfspace_normals[j];
            matrix.row(2) = halfspace_normals[k];
            if (std::abs(matrix.determinant()) <= 1.0e-10) continue;
            const Eigen::Vector3d rhs(halfspace_offsets[i],
                                      halfspace_offsets[j],
                                      halfspace_offsets[k]);
            const Eigen::Vector3d candidate = matrix.inverse() * rhs;
            bool feasible = candidate.allFinite();
            for (std::size_t plane = 0;
                 feasible && plane < halfspace_normals.size(); ++plane) {
              feasible = halfspace_normals[plane].dot(candidate) <=
                         halfspace_offsets[plane] + feasibility_tolerance;
            }
            if (!feasible) continue;
            bool duplicate = false;
            for (const auto& vertex : *vertices) {
              if ((vertex - candidate).norm() <= feasibility_tolerance) {
                duplicate = true;
                break;
              }
            }
            if (!duplicate) vertices->push_back(candidate);
          }
        }
      }
      if (vertices->size() < 4) {
        throw std::runtime_error("failed to reconstruct inflated convex component in " +
                                 path);
      }
      faces = std::make_shared<std::vector<int>>();
      int face_count = 0;
      for (std::size_t plane = 0; plane < halfspace_normals.size(); ++plane) {
        std::vector<int> indices;
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (std::size_t vertex = 0; vertex < vertices->size(); ++vertex) {
          if (std::abs(halfspace_normals[plane].dot((*vertices)[vertex]) -
                       halfspace_offsets[plane]) <= feasibility_tolerance) {
            indices.push_back(static_cast<int>(vertex));
            centroid += (*vertices)[vertex];
          }
        }
        if (indices.size() < 3) continue;
        centroid /= static_cast<double>(indices.size());
        const Eigen::Vector3d normal = halfspace_normals[plane];
        const Eigen::Vector3d helper =
            std::abs(normal.z()) < 0.9 ? Eigen::Vector3d::UnitZ()
                                      : Eigen::Vector3d::UnitX();
        const Eigen::Vector3d axis_u = helper.cross(normal).normalized();
        const Eigen::Vector3d axis_v = normal.cross(axis_u);
        std::sort(indices.begin(), indices.end(), [&](int lhs, int rhs_index) {
          const Eigen::Vector3d left = (*vertices)[lhs] - centroid;
          const Eigen::Vector3d right = (*vertices)[rhs_index] - centroid;
          return std::atan2(axis_v.dot(left), axis_u.dot(left)) <
                 std::atan2(axis_v.dot(right), axis_u.dot(right));
        });
        for (std::size_t triangle = 1; triangle + 1 < indices.size();
             ++triangle) {
          faces->push_back(3);
          faces->push_back(indices[0]);
          faces->push_back(indices[triangle]);
          faces->push_back(indices[triangle + 1]);
          ++face_count;
        }
      }
      if (face_count < 4) {
        throw std::runtime_error("inflated convex component has too few faces in " +
                                 path);
      }
      component.geometry = std::make_shared<fcl::Convexd>(
          vertices, face_count, faces, true);
    } else {
      component.geometry = std::make_shared<fcl::Convexd>(
          vertices, static_cast<int>(encoded_faces.size()), faces, true);
    }

    component.fcl_vertices = vertices;
    component.fcl_faces = faces;
    component.face_normals = halfspace_normals;
    prepareSatAxes(component);
    component.geometry->computeLocalAABB();
    result.components.push_back(std::move(component));
  }
  return result;
}

ConvexComponent makeAxisAlignedBox(const Eigen::Vector3d& minimum,
                                   const Eigen::Vector3d& maximum) {
  if (!finite(minimum) || !finite(maximum) ||
      (maximum.array() <= minimum.array()).any()) {
    throw std::invalid_argument("virtual ceiling has invalid bounds");
  }
  auto vertices = std::make_shared<std::vector<fcl::Vector3d>>();
  vertices->reserve(8);
  vertices->emplace_back(minimum.x(), minimum.y(), minimum.z());
  vertices->emplace_back(maximum.x(), minimum.y(), minimum.z());
  vertices->emplace_back(maximum.x(), maximum.y(), minimum.z());
  vertices->emplace_back(minimum.x(), maximum.y(), minimum.z());
  vertices->emplace_back(minimum.x(), minimum.y(), maximum.z());
  vertices->emplace_back(maximum.x(), minimum.y(), maximum.z());
  vertices->emplace_back(maximum.x(), maximum.y(), maximum.z());
  vertices->emplace_back(minimum.x(), maximum.y(), maximum.z());
  auto faces = std::make_shared<std::vector<int>>(
      std::initializer_list<int>{
          4, 0, 3, 2, 1, 4, 4, 5, 6, 7, 4, 0, 1, 5, 4,
          4, 3, 7, 6, 2, 4, 0, 4, 7, 3, 4, 1, 2, 6, 5});
  ConvexComponent result;
  result.fcl_vertices = vertices;
  result.fcl_faces = faces;
  result.geometry = std::make_shared<fcl::Convexd>(
      result.fcl_vertices, 6, result.fcl_faces, true);
  result.face_normals = {Eigen::Vector3d::UnitX(),
                         Eigen::Vector3d::UnitY(),
                         Eigen::Vector3d::UnitZ()};
  prepareSatAxes(result);
  result.geometry->computeLocalAABB();
  return result;
}

bool validConfig(const CollisionConfig& config, std::string* reason) {
  const auto positive = [](double value) {
    return finite(value) && value > 0.0;
  };
  const auto nonnegative = [](double value) {
    return finite(value) && value >= 0.0;
  };
  if (!positive(config.spatial_resolution) ||
      !positive(config.yaw_resolution) ||
      config.yaw_resolution > kPi ||
      !positive(config.max_yaw_delta_per_step) ||
      config.max_yaw_delta_per_step > kPi ||
      !nonnegative(config.yaw_cost_weight) ||
      !nonnegative(config.body_clearance) ||
      !positive(config.collision_sample_max_distance) ||
      !positive(config.collision_sample_max_yaw_step) ||
      config.smoothing_iterations == 0 ||
      !positive(config.smoothing_alpha) || config.smoothing_alpha > 1.0 ||
      !finite(config.virtual_ceiling_height) ||
      !nonnegative(config.numeric_tolerance)) {
    if (reason) {
      *reason = "one or more WholeBodyCollisionScene parameters are invalid";
    }
    return false;
  }
  return true;
}

bool validPose(const PoseState& state) {
  return finite(state.position) && finite(state.yaw);
}

std::uint64_t doubleBits(double value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct EdgeCacheKey {
  std::array<std::uint64_t, 8> values{};

  bool operator==(const EdgeCacheKey& other) const {
    return values == other.values;
  }
};

struct EdgeCacheKeyHash {
  std::size_t operator()(const EdgeCacheKey& key) const {
    std::size_t seed = 0x27d4eb2f165667c5ULL;
    for (const std::uint64_t value : key.values) {
      seed ^= static_cast<std::size_t>(value) + 0x9e3779b97f4a7c15ULL +
              (seed << 6U) + (seed >> 2U);
    }
    return seed;
  }
};

EdgeCacheKey edgeCacheKey(const PoseState& start, const PoseState& finish) {
  return EdgeCacheKey{{doubleBits(start.position.x()),
                       doubleBits(start.position.y()),
                       doubleBits(start.position.z()), doubleBits(start.yaw),
                       doubleBits(finish.position.x()),
                       doubleBits(finish.position.y()),
                       doubleBits(finish.position.z()),
                       doubleBits(finish.yaw)}};
}

struct PoseCacheKey {
  std::array<std::uint64_t, 4> values{};

  bool operator==(const PoseCacheKey& other) const {
    return values == other.values;
  }
};

struct PoseCacheKeyHash {
  std::size_t operator()(const PoseCacheKey& key) const {
    std::size_t seed = 0x165667b19e3779f9ULL;
    for (const std::uint64_t value : key.values) {
      seed ^= static_cast<std::size_t>(value) + 0x9e3779b97f4a7c15ULL +
              (seed << 6U) + (seed >> 2U);
    }
    return seed;
  }
};

PoseCacheKey poseCacheKey(const PoseState& state) {
  return PoseCacheKey{{doubleBits(state.position.x()),
                       doubleBits(state.position.y()),
                       doubleBits(state.position.z()),
                       doubleBits(state.yaw)}};
}

}  // namespace

struct QueryContext::Cache {
  Cache() {
    edge_validation_cache.reserve(65536);
    pose_collision_free_cache.reserve(65536);
  }

  std::unordered_map<EdgeCacheKey, bool, EdgeCacheKeyHash>
      edge_validation_cache;
  std::unordered_map<PoseCacheKey, bool, PoseCacheKeyHash>
      pose_collision_free_cache;
};

class WholeBodyCollisionScene::Impl {
 public:
  Impl(const CollisionConfig& config, const std::string& map_path,
       const std::string& body_path)
      : config_(config),
        environment_(loadPolyMap(map_path, "map",
                                 config.numeric_tolerance)),
        body_(loadPolyMap(body_path, "shape",
                          config.numeric_tolerance,
                          config.body_clearance)) {
    std::string reason;
    if (!validConfig(config_, &reason)) {
      throw std::invalid_argument("invalid WholeBodyCollisionScene config: " + reason);
    }

    Eigen::Vector3d ceiling_min = environment_.grid_min;
    Eigen::Vector3d ceiling_max = environment_.grid_max;
    ceiling_min.z() = config_.virtual_ceiling_height;
    // Extend through the top of the map so a pose cannot pass over the slab.
    ceiling_max.z() =
        std::max(environment_.grid_max.z(), config_.virtual_ceiling_height) +
        CollisionConfig::kVirtualCeilingThickness;
    environment_.components.push_back(
        makeAxisAlignedBox(ceiling_min, ceiling_max));

    info_.map_name = environment_.name;
    info_.map_frame = environment_.frame;
    info_.body_name = body_.name;
    info_.obstacle_components = environment_.components.size();
    info_.body_components = body_.components.size();
    info_.map_min = environment_.grid_min;
    info_.map_max = environment_.grid_max;

    environment_objects_.reserve(environment_.components.size());
    std::vector<fcl::CollisionObjectd*> objects;
    objects.reserve(environment_.components.size());
    for (std::size_t index = 0; index < environment_.components.size();
         ++index) {
      auto object = std::make_unique<fcl::CollisionObjectd>(
          environment_.components[index].geometry);
      object->setUserData(&environment_.components[index]);
      objects.push_back(object.get());
      environment_objects_.push_back(std::move(object));
    }
    environment_manager_.registerObjects(objects);
    environment_manager_.setup();

    body_objects_.reserve(body_.components.size());
    for (auto& component : body_.components) {
      auto object = std::make_unique<fcl::CollisionObjectd>(component.geometry);
      object->setUserData(&component);
      body_objects_.push_back(std::move(object));
    }

    Eigen::Vector3d body_min = Eigen::Vector3d::Constant(kInfinity);
    Eigen::Vector3d body_max = Eigen::Vector3d::Constant(-kInfinity);
    for (const auto& component : body_.components) {
      body_min = body_min.cwiseMin(component.local_min);
      body_max = body_max.cwiseMax(component.local_max);
    }
    body_radius_ = 0.5 * (body_max - body_min).norm() + config_.body_clearance;
  }

  const ModelInfo& modelInfo() const { return info_; }
  double bodyRadius() const { return body_radius_; }
  double yawCostWeight() const { return config_.yaw_cost_weight; }

  bool isPoseCollisionFree(const PoseState& state,
                           QueryContext* outer) const {
    if (outer != nullptr && outer->stopRequested()) return false;
    return poseCollisionFree(state, outer);
  }

  bool isEdgeCollisionFree(const PoseState& start, const PoseState& finish,
                           QueryContext& outer,
                           EdgeValidationMode mode) const {
    if (outer.stopRequested()) return false;
    return edgeCollisionFree(start, finish, outer, mode);
  }

  std::optional<double> nearestCollisionFreeYaw(
      const Eigen::Vector3d& position, double hinted_yaw,
      double reference_yaw) const {
    if (!position.allFinite() || !std::isfinite(hinted_yaw) ||
        !std::isfinite(reference_yaw)) {
      return std::nullopt;
    }

    struct Candidate {
      double yaw;
      double hint_distance;
      double reference_distance;
      double normalized_yaw;
    };
    std::optional<Candidate> best;
    const int steps = static_cast<int>(
        std::ceil(M_PI / config_.yaw_resolution));
    for (int offset = -steps; offset <= steps; ++offset) {
      const double yaw = hinted_yaw + offset * config_.yaw_resolution;
      if (!poseCollisionFree(PoseState{position, yaw}, nullptr)) continue;
      const Candidate candidate{
          yaw,
          std::abs(std::remainder(yaw - hinted_yaw, 2.0 * M_PI)),
          std::abs(std::remainder(yaw - reference_yaw, 2.0 * M_PI)),
          std::remainder(yaw, 2.0 * M_PI)};
      constexpr double tolerance = 1.0e-12;
      const bool better =
          !best ||
          candidate.hint_distance < best->hint_distance - tolerance ||
          (std::abs(candidate.hint_distance - best->hint_distance) <= tolerance &&
           (candidate.reference_distance < best->reference_distance - tolerance ||
            (std::abs(candidate.reference_distance - best->reference_distance) <= tolerance &&
             candidate.normalized_yaw < best->normalized_yaw)));
      if (better) best = candidate;
    }
    return best ? std::optional<double>(best->yaw) : std::nullopt;
  }

  std::vector<PoseState> simplifyPath(const std::vector<PoseState>& raw,
                                      QueryContext& outer) const {
    return simplifyPathImpl(raw, outer);
  }
  std::vector<PoseState> smoothPath(const std::vector<PoseState>& shortcut,
                                    QueryContext& outer,
                                    std::size_t& accepted_updates) const {
    return smoothPathImpl(shortcut, outer, accepted_updates);
  }
  std::vector<PoseState> cleanPosePath(
      const std::vector<PoseState>& input) const;

 private:
  struct CollisionQuery {
    const QueryContext* context{nullptr};
    double tolerance{0.0};
    bool collided{false};
  };

  static bool separatedOnAxis(const ConvexComponent& first,
                              const fcl::Transform3d& first_transform,
                              const ConvexComponent& second,
                              const fcl::Transform3d& second_transform,
                              const Eigen::Vector3d& axis,
                              double tolerance) {
    const double norm = axis.norm();
    if (norm <= 1.0e-10) return false;
    const Eigen::Vector3d unit = axis / norm;
    double first_min = kInfinity, first_max = -kInfinity;
    for (const auto& local : *first.fcl_vertices) {
      const double value = unit.dot(first_transform * local);
      first_min = std::min(first_min, value);
      first_max = std::max(first_max, value);
    }
    double second_min = kInfinity, second_max = -kInfinity;
    for (const auto& local : *second.fcl_vertices) {
      const double value = unit.dot(second_transform * local);
      second_min = std::min(second_min, value);
      second_max = std::max(second_max, value);
    }
    return first_max < second_min - tolerance ||
           second_max < first_min - tolerance;
  }

  static bool satOverlap(const ConvexComponent& first,
                         const fcl::Transform3d& first_transform,
                         const ConvexComponent& second,
                         const fcl::Transform3d& second_transform,
                         double tolerance) {
    for (const auto& normal : first.face_normals) {
      if (separatedOnAxis(first, first_transform, second, second_transform,
                          first_transform.linear() * normal, tolerance)) {
        return false;
      }
    }
    for (const auto& normal : second.face_normals) {
      if (separatedOnAxis(first, first_transform, second, second_transform,
                          second_transform.linear() * normal, tolerance)) {
        return false;
      }
    }
    for (const auto& first_edge : first.edge_directions) {
      const Eigen::Vector3d world_first =
          first_transform.linear() * first_edge;
      for (const auto& second_edge : second.edge_directions) {
        const Eigen::Vector3d axis = world_first.cross(
            second_transform.linear() * second_edge);
        if (separatedOnAxis(first, first_transform, second, second_transform,
                            axis, tolerance)) {
          return false;
        }
      }
    }
    return true;
  }

  static bool sweptAabbOverlap(const ConvexComponent& moving,
                               const Eigen::Matrix3d& rotation,
                               const Eigen::Vector3d& start,
                               const Eigen::Vector3d& finish,
                               const ConvexComponent& fixed,
                               double tolerance) {
    Eigen::Vector3d moving_min = Eigen::Vector3d::Constant(kInfinity);
    Eigen::Vector3d moving_max = Eigen::Vector3d::Constant(-kInfinity);
    for (const auto& vertex : *moving.fcl_vertices) {
      const Eigen::Vector3d rotated = rotation * vertex;
      moving_min = moving_min.cwiseMin(rotated + start);
      moving_min = moving_min.cwiseMin(rotated + finish);
      moving_max = moving_max.cwiseMax(rotated + start);
      moving_max = moving_max.cwiseMax(rotated + finish);
    }
    return (moving_max.array() >=
            (fixed.local_min.array() - tolerance)).all() &&
           (fixed.local_max.array() >=
            (moving_min.array() - tolerance)).all();
  }

  static bool sweptSatOverlap(const ConvexComponent& moving,
                              const Eigen::Matrix3d& rotation,
                              const Eigen::Vector3d& start,
                              const Eigen::Vector3d& finish,
                              const ConvexComponent& fixed,
                              double tolerance) {
    const Eigen::Vector3d displacement = finish - start;
    double first_contact = 0.0;
    double last_contact = 1.0;
    const auto overlapsDuringSweep = [&](const Eigen::Vector3d& axis) {
      const double norm = axis.norm();
      if (norm <= 1.0e-10) return true;
      const Eigen::Vector3d unit = axis / norm;
      double moving_min = kInfinity, moving_max = -kInfinity;
      for (const auto& vertex : *moving.fcl_vertices) {
        const double value = unit.dot(start + rotation * vertex);
        moving_min = std::min(moving_min, value);
        moving_max = std::max(moving_max, value);
      }
      double fixed_min = kInfinity, fixed_max = -kInfinity;
      for (const auto& vertex : *fixed.fcl_vertices) {
        const double value = unit.dot(vertex);
        fixed_min = std::min(fixed_min, value);
        fixed_max = std::max(fixed_max, value);
      }
      const double speed = unit.dot(displacement);
      const double enter_numerator = fixed_min - tolerance - moving_max;
      const double leave_numerator = fixed_max + tolerance - moving_min;
      if (std::abs(speed) <= 1.0e-12) {
        return enter_numerator <= 0.0 && leave_numerator >= 0.0;
      }
      double enter = enter_numerator / speed;
      double leave = leave_numerator / speed;
      if (enter > leave) std::swap(enter, leave);
      first_contact = std::max(first_contact, enter);
      last_contact = std::min(last_contact, leave);
      return first_contact <= last_contact;
    };

    for (const auto& normal : moving.face_normals) {
      if (!overlapsDuringSweep(rotation * normal)) return false;
    }
    for (const auto& normal : fixed.face_normals) {
      if (!overlapsDuringSweep(normal)) return false;
    }
    for (const auto& moving_edge : moving.edge_directions) {
      const Eigen::Vector3d world_edge = rotation * moving_edge;
      for (const auto& fixed_edge : fixed.edge_directions) {
        if (!overlapsDuringSweep(world_edge.cross(fixed_edge))) return false;
      }
    }
    return true;
  }

  static bool collideBroadphase(fcl::CollisionObjectd* first,
                                fcl::CollisionObjectd* second,
                                void* data) {
    auto* query = static_cast<CollisionQuery*>(data);
    if (query->context != nullptr) {
      if (query->context->stopRequested()) {
        query->collided = true;
        return true;
      }
      ++query->context->collision_queries;
      ++query->context->fcl_queries;
    }
    const auto* first_component =
        static_cast<const ConvexComponent*>(first->getUserData());
    const auto* second_component =
        static_cast<const ConvexComponent*>(second->getUserData());
    query->collided = first_component && second_component &&
        satOverlap(*first_component, first->getTransform(),
                   *second_component, second->getTransform(),
                   query->tolerance);
    return query->collided;
  }

  Eigen::Matrix3d yawRotation(double yaw) const {
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    Eigen::Matrix3d result;
    result << cosine, -sine, 0.0,
              sine, cosine, 0.0,
              0.0, 0.0, 1.0;
    return result;
  }

  bool poseCollisionFree(const PoseState& state,
                         QueryContext* context) const;
  bool edgeCollisionFree(const PoseState& start, const PoseState& finish,
                         QueryContext& context,
                         EdgeValidationMode mode) const;
  bool translationSweepCollisionFree(const PoseState& start,
                                     const PoseState& finish,
                                     QueryContext& context) const;
  std::vector<PoseState> simplifyPathImpl(
      const std::vector<PoseState>& raw, QueryContext& context) const;
  std::vector<PoseState> smoothPathImpl(
      const std::vector<PoseState>& shortcut, QueryContext& context,
      std::size_t& accepted_updates) const;

  CollisionConfig config_;
  LoadedPolyMap environment_;
  LoadedPolyMap body_;
  ModelInfo info_;
  struct DiscreteYawCache {
    Eigen::Matrix3d rotation;
    std::vector<Eigen::Vector3d> aabb_min;
    std::vector<Eigen::Vector3d> aabb_max;
  };

  DiscreteYawCache makeYawCache(double yaw) const {
    DiscreteYawCache cache;
    cache.rotation = yawRotation(yaw);
    cache.aabb_min.reserve(body_.components.size());
    cache.aabb_max.reserve(body_.components.size());
    for (const auto& component : body_.components) {
      Eigen::Vector3d local_min = Eigen::Vector3d::Constant(kInfinity);
      Eigen::Vector3d local_max = Eigen::Vector3d::Constant(-kInfinity);
      for (const auto& vertex : *component.fcl_vertices) {
        const Eigen::Vector3d rotated = cache.rotation * vertex;
        local_min = local_min.cwiseMin(rotated);
        local_max = local_max.cwiseMax(rotated);
      }
      cache.aabb_min.push_back(local_min);
      cache.aabb_max.push_back(local_max);
    }
    return cache;
  }

  const DiscreteYawCache* latticeYawCache(double yaw) const {
    const double step = config_.yaw_resolution;
    if (step <= 0.0) return nullptr;
    const int bin = static_cast<int>(std::lround(yaw / step));
    const double lattice = static_cast<double>(bin) * step;
    if (std::abs(std::remainder(positiveAngle(yaw) - positiveAngle(lattice),
                                kTwoPi)) >
        10.0 * config_.numeric_tolerance &&
        std::abs(std::remainder(yaw - lattice, kTwoPi)) >
            10.0 * config_.numeric_tolerance) {
      return nullptr;
    }
    auto found = discrete_yaw_cache_.find(bin);
    if (found == discrete_yaw_cache_.end()) {
      found = discrete_yaw_cache_.emplace(bin, makeYawCache(lattice)).first;
    }
    return &found->second;
  }

  struct SweepQuery {
    const Impl* impl{nullptr};
    QueryContext* context{nullptr};
    Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d start{Eigen::Vector3d::Zero()};
    Eigen::Vector3d finish{Eigen::Vector3d::Zero()};
    bool collided{false};
  };

  static bool sweepBroadphase(fcl::CollisionObjectd* first,
                              fcl::CollisionObjectd* second, void* data) {
    auto* query = static_cast<SweepQuery*>(data);
    if (query->context != nullptr && query->context->stopRequested()) {
      query->collided = true;
      return true;
    }
    const auto* obstacle_component =
        static_cast<const ConvexComponent*>(first ? first->getUserData() : nullptr);
    if (obstacle_component == nullptr && second != nullptr) {
      obstacle_component =
          static_cast<const ConvexComponent*>(second->getUserData());
    }
    if (query->context != nullptr) ++query->context->swept_aabb_candidates;
    if (!query->impl || !obstacle_component) return false;
    for (const auto& body_component : query->impl->body_.components) {
      if (!sweptAabbOverlap(body_component, query->rotation, query->start,
                            query->finish, *obstacle_component,
                            query->impl->config_.numeric_tolerance)) {
        continue;
      }
      if (query->context != nullptr) {
        ++query->context->swept_sat_calls;
        ++query->context->collision_queries;
      }
      if (sweptSatOverlap(body_component, query->rotation, query->start,
                          query->finish, *obstacle_component,
                          query->impl->config_.numeric_tolerance)) {
        query->collided = true;
        return true;
      }
    }
    return false;
  }

  std::vector<std::unique_ptr<fcl::CollisionObjectd>> environment_objects_;
  mutable std::vector<std::unique_ptr<fcl::CollisionObjectd>> body_objects_;
  mutable fcl::DynamicAABBTreeCollisionManagerd environment_manager_;
  mutable std::unordered_map<int, DiscreteYawCache> discrete_yaw_cache_;
  double body_radius_{0.0};
};

bool WholeBodyCollisionScene::Impl::poseCollisionFree(
    const PoseState& state, QueryContext* context) const {
  if (context != nullptr && context->stopRequested()) return false;
  if (!validPose(state)) return false;
  const PoseCacheKey cache_key = poseCacheKey(state);
  if (context != nullptr) {
    const auto cached = context->cache().pose_collision_free_cache.find(cache_key);
    if (cached != context->cache().pose_collision_free_cache.end()) {
      ++context->pose_cache_hits;
      return cached->second;
    }
  }
  const auto cacheResult = [&](bool collision_free) {
    if (context != nullptr) {
      context->cache().pose_collision_free_cache.emplace(cache_key,
                                                         collision_free);
    }
    return collision_free;
  };
  const auto* yaw_cache = latticeYawCache(state.yaw);
  const Eigen::Matrix3d rotation =
      yaw_cache ? yaw_cache->rotation : yawRotation(state.yaw);

  for (const auto& component : body_.components) {
    for (int boundary_index = 0; boundary_index < 6; ++boundary_index) {
      const int axis = boundary_index / 2;
      const bool maximum_plane = (boundary_index % 2) == 1;
      Eigen::Vector3d normal = Eigen::Vector3d::Zero();
      normal[axis] = maximum_plane ? 1.0 : -1.0;
      const double coordinate =
          maximum_plane ? environment_.grid_max[axis]
                        : -environment_.grid_min[axis];
      const Eigen::Vector3d local_vertex =
          component.supportVertex(rotation.transpose() * normal);
      const Eigen::Vector3d body_point =
          state.position + rotation * local_vertex;
      const double clearance = coordinate - normal.dot(body_point);
      if (clearance <= config_.numeric_tolerance) {
        return cacheResult(false);
      }
    }
  }

  fcl::Transform3d body_transform = fcl::Transform3d::Identity();
  body_transform.linear() = rotation;
  body_transform.translation() = state.position;
  CollisionQuery query{context, config_.numeric_tolerance, false};
  for (auto& body_object : body_objects_) {
    body_object->setTransform(body_transform);
    body_object->computeAABB();
    environment_manager_.collide(body_object.get(), &query, collideBroadphase);
    if (query.collided) return cacheResult(false);
  }
  return cacheResult(true);
}

bool WholeBodyCollisionScene::Impl::edgeCollisionFree(
    const PoseState& start, const PoseState& finish, QueryContext& context,
    EdgeValidationMode mode) const {
  if (!validPose(start) || !validPose(finish)) return false;
  auto& edges = context.cache().edge_validation_cache;
  const EdgeCacheKey cache_key = edgeCacheKey(start, finish);
  const auto cached = edges.find(cache_key);
  if (cached != edges.end()) {
    if (!cached->second || mode == EdgeValidationMode::kSearch) {
      return cached->second;
    }
  }
  const auto cacheResult = [&](bool value) {
    if (mode == EdgeValidationMode::kCertified || !value) {
      edges[cache_key] = value;
    } else {
      edges.emplace(cache_key, value);
    }
    return value;
  };
  const double distance = (finish.position - start.position).norm();
  const double yaw_delta = angleDifference(finish.yaw, start.yaw);
  std::size_t intervals = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::ceil(distance /
                       config_.collision_sample_max_distance)));
  intervals = std::max(
      intervals, static_cast<std::size_t>(
                     std::ceil(std::abs(yaw_delta) /
                               config_.collision_sample_max_yaw_step)));
    if (mode == EdgeValidationMode::kCertified) {
      intervals = std::max<std::size_t>(8, intervals);
      intervals = std::min<std::size_t>(256, intervals);
    } else {
      intervals = std::min<std::size_t>(24, intervals);
    }

  for (std::size_t index = 0; index <= intervals; ++index) {
    if (context.stopRequested()) {
      return false;
    }
    const double normalized_time =
        static_cast<double>(index) / static_cast<double>(intervals);
    const double fraction = mighty::quinticPosePhase(normalized_time);
    PoseState sample;
    sample.position =
        start.position + fraction * (finish.position - start.position);
    sample.yaw = start.yaw + fraction * yaw_delta;
    ++context.collision_samples;
    if (!poseCollisionFree(sample, &context)) return cacheResult(false);
  }
  if (mode == EdgeValidationMode::kCertified &&
      std::abs(yaw_delta) <= config_.numeric_tolerance &&
      !translationSweepCollisionFree(start, finish, context)) {
    edges[cache_key] = false;
    return false;
  }
  return cacheResult(true);
}

bool WholeBodyCollisionScene::Impl::translationSweepCollisionFree(
    const PoseState& start, const PoseState& finish,
    QueryContext& context) const {
  const DiscreteYawCache* yaw_cache = latticeYawCache(start.yaw);
  const Eigen::Matrix3d rotation =
      yaw_cache ? yaw_cache->rotation : yawRotation(start.yaw);
  Eigen::Vector3d world_min = Eigen::Vector3d::Constant(kInfinity);
  Eigen::Vector3d world_max = Eigen::Vector3d::Constant(-kInfinity);
  if (yaw_cache) {
    for (std::size_t index = 0; index < yaw_cache->aabb_min.size(); ++index) {
      world_min = world_min.cwiseMin(yaw_cache->aabb_min[index] + start.position);
      world_min = world_min.cwiseMin(yaw_cache->aabb_min[index] + finish.position);
      world_max = world_max.cwiseMax(yaw_cache->aabb_max[index] + start.position);
      world_max = world_max.cwiseMax(yaw_cache->aabb_max[index] + finish.position);
    }
  } else {
    for (const auto& body_component : body_.components) {
      for (const auto& vertex : *body_component.fcl_vertices) {
        const Eigen::Vector3d rotated = rotation * vertex;
        world_min = world_min.cwiseMin(rotated + start.position);
        world_min = world_min.cwiseMin(rotated + finish.position);
        world_max = world_max.cwiseMax(rotated + start.position);
        world_max = world_max.cwiseMax(rotated + finish.position);
      }
    }
  }
  world_min.array() -= config_.numeric_tolerance;
  world_max.array() += config_.numeric_tolerance;
  const Eigen::Vector3d extents =
      (world_max - world_min).cwiseMax(1.0e-6);
  const auto box = std::make_shared<fcl::Boxd>(extents.x(), extents.y(),
                                               extents.z());
  fcl::CollisionObjectd sweep_box(box);
  sweep_box.setTranslation(0.5 * (world_min + world_max));
  sweep_box.computeAABB();
  SweepQuery query{this, &context, rotation, start.position, finish.position,
                   false};
  environment_manager_.collide(&sweep_box, &query, sweepBroadphase);
  return !query.collided;
}

std::vector<PoseState> WholeBodyCollisionScene::Impl::simplifyPathImpl(
    const std::vector<PoseState>& raw, QueryContext& context) const {
  if (raw.empty()) return {};
  if (raw.size() == 1) return raw;

  const std::size_t count = raw.size();
  std::vector<PoseState> shortened;
  shortened.reserve(count);
  shortened.push_back(raw.front());
  std::size_t begin = 0;
  while (begin + 1 < count) {
    std::size_t selected = count;
    for (std::size_t candidate = count - 1; candidate > begin;
         --candidate) {
      if (context.stopRequested()) return {};
      ++context.edge_checks;
      if (edgeCollisionFree(raw[begin], raw[candidate], context,
                            EdgeValidationMode::kCertified)) {
        selected = candidate;
        break;
      }
    }
    if (selected == count) return {};
    shortened.push_back(raw[selected]);
    begin = selected;
  }
  return shortened;
}

std::vector<PoseState> WholeBodyCollisionScene::Impl::cleanPosePath(
    const std::vector<PoseState>& input) const {
  if (input.size() <= 2) return input;
  std::vector<PoseState> cleaned;
  cleaned.reserve(input.size());
  cleaned.push_back(input.front());
  for (std::size_t index = 1; index + 1 < input.size(); ++index) {
    const PoseState& previous = cleaned.back();
    const PoseState& current = input[index];
    const PoseState& next = input[index + 1];
    const double incoming = (current.position - previous.position).norm();
    const double outgoing = (next.position - current.position).norm();
    const double incoming_yaw = std::abs(angleDifference(current.yaw, previous.yaw));
    const double outgoing_yaw = std::abs(angleDifference(next.yaw, current.yaw));
    if ((incoming <= config_.numeric_tolerance &&
         incoming_yaw <= config_.numeric_tolerance) ||
        (outgoing <= config_.numeric_tolerance &&
         outgoing_yaw <= config_.numeric_tolerance)) {
      continue;
    }
    cleaned.push_back(current);
  }
  cleaned.push_back(input.back());
  return cleaned;
}

std::vector<PoseState> WholeBodyCollisionScene::Impl::smoothPathImpl(
    const std::vector<PoseState>& shortcut, QueryContext& context,
    std::size_t& accepted_updates) const {
  accepted_updates = 0;
  if (!config_.smoothing_enabled || shortcut.size() <= 2) return shortcut;

  std::vector<PoseState> smoothed = shortcut;
  for (std::size_t iteration = 0;
       iteration < config_.smoothing_iterations; ++iteration) {
    bool changed = false;
    for (std::size_t index = 1; index + 1 < smoothed.size(); ++index) {
      if (context.stopRequested()) return {};
      const PoseState current = smoothed[index];
      PoseState candidate = current;
      const Eigen::Vector3d midpoint =
          0.5 * (smoothed[index - 1].position + smoothed[index + 1].position);
      candidate.position += config_.smoothing_alpha * (midpoint - current.position);
      const double previous_delta =
          angleDifference(smoothed[index - 1].yaw, current.yaw);
      const double next_delta =
          angleDifference(smoothed[index + 1].yaw, current.yaw);
      candidate.yaw += config_.smoothing_alpha *
                       0.5 * (previous_delta + next_delta);

      if (!poseCollisionFree(candidate, &context) ||
          !edgeCollisionFree(smoothed[index - 1], candidate, context,
                             EdgeValidationMode::kCertified) ||
          !edgeCollisionFree(candidate, smoothed[index + 1], context,
                             EdgeValidationMode::kCertified)) {
        continue;
      }
      smoothed[index] = candidate;
      ++accepted_updates;
      changed = true;
    }
    if (!changed) break;
  }

  for (std::size_t edge = 0; edge + 1 < smoothed.size(); ++edge) {
    if (!edgeCollisionFree(smoothed[edge], smoothed[edge + 1], context,
                           EdgeValidationMode::kCertified)) {
      return {};
    }
  }
  smoothed.front() = shortcut.front();
  smoothed.back() = shortcut.back();
  return smoothed;
}


const char* toString(PlanStatus status) {
  switch (status) {
    case PlanStatus::kSuccess:
      return "success";
    case PlanStatus::kCancelled:
      return "cancelled";
    case PlanStatus::kTimedOut:
      return "timed_out";
    case PlanStatus::kNoPath:
      return "no_path";
    case PlanStatus::kCertificationFailed:
      return "certification_failed";
    case PlanStatus::kInvalidRequest:
      return "invalid_request";
    case PlanStatus::kStartInCollision:
      return "start_in_collision";
    case PlanStatus::kGoalInCollision:
      return "goal_in_collision";
    case PlanStatus::kInternalError:
      return "internal_error";
  }
  return "unknown";
}

QueryContext::QueryContext(
    CancelPredicate cancelled,
    std::optional<std::chrono::steady_clock::time_point> deadline)
    : cancelled_(std::move(cancelled)), deadline_(std::move(deadline)) {}

QueryContext::~QueryContext() = default;
QueryContext::QueryContext(QueryContext&&) noexcept = default;
QueryContext& QueryContext::operator=(QueryContext&&) noexcept = default;

QueryContext::Cache& QueryContext::cache() const {
  if (!cache_) cache_ = std::make_unique<Cache>();
  return *cache_;
}

bool QueryContext::stopRequested() const {
  if (cancelled_ && cancelled_()) {
    cancellation_seen_ = true;
    return true;
  }
  if (deadline_ && std::chrono::steady_clock::now() >= *deadline_) {
    timeout_seen_ = true;
    return true;
  }
  return false;
}

WholeBodyCollisionScene::WholeBodyCollisionScene(
    const CollisionConfig& config, const std::string& map_polymap_path,
    const std::string& body_polymap_path)
    : impl_(std::make_unique<Impl>(config, map_polymap_path, body_polymap_path)) {}

WholeBodyCollisionScene::~WholeBodyCollisionScene() = default;
WholeBodyCollisionScene::WholeBodyCollisionScene(
    WholeBodyCollisionScene&&) noexcept = default;
WholeBodyCollisionScene& WholeBodyCollisionScene::operator=(
    WholeBodyCollisionScene&&) noexcept = default;

const ModelInfo& WholeBodyCollisionScene::modelInfo() const {
  return impl_->modelInfo();
}

double WholeBodyCollisionScene::bodyRadius() const {
  return impl_->bodyRadius();
}

bool WholeBodyCollisionScene::isPoseCollisionFree(
    const PoseState& state, QueryContext* context) const {
  return impl_->isPoseCollisionFree(state, context);
}

bool WholeBodyCollisionScene::isEdgeCollisionFree(
    const PoseState& start, const PoseState& finish, QueryContext& context,
    EdgeValidationMode mode) const {
  return impl_->isEdgeCollisionFree(start, finish, context, mode);
}

bool WholeBodyCollisionScene::isCertifiedEdgeCollisionFree(
    const PoseState& start, const PoseState& finish) const {
  QueryContext context;
  return impl_->isEdgeCollisionFree(start, finish, context,
                                    EdgeValidationMode::kCertified);
}

std::optional<double> WholeBodyCollisionScene::nearestCollisionFreeYaw(
    const Eigen::Vector3d& position, double hinted_yaw,
    double reference_yaw) const {
  return impl_->nearestCollisionFreeYaw(position, hinted_yaw, reference_yaw);
}

std::vector<PoseState> WholeBodyCollisionScene::simplifyPath(
    const std::vector<PoseState>& raw, QueryContext& context) const {
  return impl_->simplifyPath(raw, context);
}

std::vector<PoseState> WholeBodyCollisionScene::cleanPosePath(
    const std::vector<PoseState>& input) const {
  return impl_->cleanPosePath(input);
}

std::vector<PoseState> WholeBodyCollisionScene::smoothPath(
    const std::vector<PoseState>& shortcut, QueryContext& context,
    std::size_t& accepted_updates) const {
  return impl_->smoothPath(shortcut, context, accepted_updates);
}

bool WholeBodyCollisionScene::certifyPath(
    const std::vector<PoseState>& waypoints, QueryContext& context) const {
  if (waypoints.size() < 2) return false;
  for (std::size_t i = 0; i + 1 < waypoints.size(); ++i) {
    ++context.edge_checks;
    if (!isEdgeCollisionFree(waypoints[i], waypoints[i + 1], context,
                             EdgeValidationMode::kCertified)) {
      return false;
    }
  }
  return true;
}

double WholeBodyCollisionScene::pathCost(
    const std::vector<PoseState>& waypoints) const {
  double cost = 0.0;
  for (std::size_t i = 0; i + 1 < waypoints.size(); ++i) {
    cost += (waypoints[i + 1].position - waypoints[i].position).norm();
    cost += impl_->yawCostWeight() *
            std::abs(std::remainder(waypoints[i + 1].yaw - waypoints[i].yaw,
                                    2.0 * 3.14159265358979323846));
  }
  return cost;
}

std::vector<PoseState> WholeBodyCollisionScene::postprocessPath(
    const std::vector<PoseState>& raw, QueryContext& context,
    PlanStatistics* statistics) const {
  auto shortcut = simplifyPath(raw, context);
  if (shortcut.empty()) return {};
  std::size_t updates = 0;
  auto smoothed = smoothPath(shortcut, context, updates);
  std::vector<PoseState> result;
  if (!smoothed.empty()) {
    result = cleanPosePath(smoothed);
    if (statistics) {
      statistics->smoothing_applied = true;
      statistics->smoothing_updates = updates;
      statistics->smoothing_fallback = false;
    }
  } else {
    result = cleanPosePath(shortcut);
    if (statistics) {
      statistics->smoothing_applied = false;
      statistics->smoothing_fallback = true;
      statistics->smoothing_updates = 0;
    }
  }
  if (!certifyPath(result, context)) return {};
  if (statistics) {
    statistics->raw_waypoints = raw.size();
    statistics->final_cost = pathCost(result);
    double length = 0.0;
    double yaw_travel = 0.0;
    double vertical = 0.0;
    double max_alt = 0.0;
    for (std::size_t i = 0; i + 1 < result.size(); ++i) {
      length += (result[i + 1].position - result[i].position).norm();
      yaw_travel += std::abs(std::remainder(
          result[i + 1].yaw - result[i].yaw, 2.0 * 3.14159265358979323846));
      vertical += std::abs(result[i + 1].position.z() - result[i].position.z());
      max_alt = std::max(max_alt, std::abs(result[i + 1].position.z() -
                                           result.front().position.z()));
    }
    statistics->path_length = length;
    statistics->yaw_travel = yaw_travel;
    statistics->vertical_travel = vertical;
    statistics->maximum_altitude_excursion = max_alt;
    statistics->yaw_cost_weight = impl_->yawCostWeight();
    statistics->collision_queries = context.collision_queries;
    statistics->collision_samples = context.collision_samples;
    statistics->edge_checks = context.edge_checks;
  }
  return result;
}

}  // namespace mighty::frontend
