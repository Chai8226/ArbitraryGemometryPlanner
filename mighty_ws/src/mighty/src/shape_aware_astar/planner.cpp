#include "mighty/shape_aware_astar/planner.hpp"
#include "mighty/shape_aware_astar/corridor.hpp"

#include "mighty/frontend/collision_scene.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mighty::shape_aware_astar {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kInfinity = std::numeric_limits<double>::infinity();

bool finite(double value) { return std::isfinite(value); }

bool finite(const Eigen::Vector3d& value) {
  return value.array().isFinite().all();
}

double wrapAngle(double angle) { return std::remainder(angle, kTwoPi); }

double positiveAngle(double angle) {
  double result = std::fmod(angle, kTwoPi);
  if (result < 0.0) result += kTwoPi;
  return result;
}

double angleDifference(double target, double source) {
  return wrapAngle(target - source);
}

bool validConfig(const PlannerConfig& config, std::string* reason) {
  const auto positive = [](double value) {
    return finite(value) && value > 0.0;
  };
  const auto nonnegative = [](double value) {
    return finite(value) && value >= 0.0;
  };
  if (!positive(config.timeout_seconds) || config.max_expansions == 0 ||
      !positive(config.heuristic_weight) ||
      !positive(config.collision.spatial_resolution) ||
      !positive(config.collision.yaw_resolution) ||
      config.collision.yaw_resolution > kPi ||
      !positive(config.collision.max_yaw_delta_per_step) ||
      config.collision.max_yaw_delta_per_step > kPi ||
      config.max_yaw_candidates == 0 ||
      !nonnegative(config.collision.yaw_cost_weight) ||
      !nonnegative(config.collision.body_clearance) ||
      !positive(config.collision.collision_sample_max_distance) ||
      !positive(config.collision.collision_sample_max_yaw_step) ||
      config.collision.smoothing_iterations == 0 ||
      !positive(config.collision.smoothing_alpha) ||
      config.collision.smoothing_alpha > 1.0 ||
      !finite(config.collision.virtual_ceiling_height) ||
      !nonnegative(config.collision.numeric_tolerance)) {
    if (reason) {
      *reason =
          "one or more scalar Shape-aware Weighted A* parameters are invalid";
    }
    return false;
  }
  return true;
}

bool validPose(const PoseState& state) {
  return finite(state.position) && finite(state.yaw);
}

struct GridKey {
  int x{0};
  int y{0};
  int z{0};
  int yaw{0};
  std::uint8_t phase{0};

  bool operator==(const GridKey& other) const {
    return x == other.x && y == other.y && z == other.z && yaw == other.yaw &&
           phase == other.phase;
  }
};

struct GridKeyHash {
  std::size_t operator()(const GridKey& key) const {
    std::size_t seed = 0x85ebca6bU;
    for (const int value : {key.x, key.y, key.z, key.yaw,
                            static_cast<int>(key.phase)}) {
      const std::size_t hash = std::hash<int>{}(value);
      seed ^= hash + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    }
    return seed;
  }
};

struct SpatialKey {
  int x{0};
  int y{0};
  int z{0};
  std::uint8_t phase{0};

  bool operator==(const SpatialKey& other) const {
    return x == other.x && y == other.y && z == other.z &&
           phase == other.phase;
  }
};

struct SpatialKeyHash {
  std::size_t operator()(const SpatialKey& key) const {
    std::size_t seed = 0x27d4eb2fU;
    for (const int value : {key.x, key.y, key.z, static_cast<int>(key.phase)}) {
      const std::size_t hash = std::hash<int>{}(value);
      seed ^= hash + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    }
    return seed;
  }
};

struct SearchScore {
  double cost{0.0};
  double distance{0.0};
  double vertical{0.0};
  double yaw_travel{0.0};
};

bool scoreLess(const SearchScore& left, const SearchScore& right,
               double tolerance) {
  if (left.cost < right.cost - tolerance) return true;
  if (left.cost > right.cost + tolerance) return false;
  if (left.vertical < right.vertical - tolerance) return true;
  if (left.vertical > right.vertical + tolerance) return false;
  if (left.yaw_travel < right.yaw_travel - tolerance) return true;
  if (left.yaw_travel > right.yaw_travel + tolerance) return false;
  return left.distance < right.distance - tolerance;
}

bool scoreEqual(const SearchScore& left, const SearchScore& right,
                double tolerance) {
  return std::abs(left.cost - right.cost) <= tolerance &&
         std::abs(left.distance - right.distance) <= tolerance &&
         std::abs(left.vertical - right.vertical) <= tolerance &&
         std::abs(left.yaw_travel - right.yaw_travel) <= tolerance;
}

struct SearchNode {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GridKey key;
  PoseState pose;
  SearchScore score;
  std::int64_t parent{-1};
  bool edge_validated{true};
};

struct QueueEntry {
  double priority{0.0};
  double heuristic{0.0};
  double vertical{0.0};
  double yaw_travel{0.0};
  std::uint8_t phase{0};
  std::size_t sequence{0};
  std::size_t node{0};
  SearchScore score;
};

struct QueueGreater {
  bool operator()(const QueueEntry& left, const QueueEntry& right) const {
    return std::tie(left.priority, left.heuristic, left.vertical,
                    left.yaw_travel, left.phase, left.sequence) >
           std::tie(right.priority, right.heuristic, right.vertical,
                    right.yaw_travel, right.phase, right.sequence);
  }
};

}  // namespace

class ShapeAwareAstarPlanner::Impl {
 public:
  Impl(const PlannerConfig& config, const std::string& map_path,
       const std::string& body_path)
      : config_(config), scene_(config.collision, map_path, body_path) {
    std::string reason;
    if (!validConfig(config_, &reason)) {
      throw std::invalid_argument(
          "invalid Shape-aware Weighted A* config: " + reason);
    }
    const auto& src = scene_.modelInfo();
    info_.map_name = src.map_name;
    info_.map_frame = src.map_frame;
    info_.body_name = src.body_name;
    info_.obstacle_components = src.obstacle_components;
    info_.body_components = src.body_components;
    info_.map_min = src.map_min;
    info_.map_max = src.map_max;
  }

  const ModelInfo& modelInfo() const { return info_; }

  PlanResult plan(const PlanRequest& request,
                  const CancelPredicate& cancelled) const;

  bool isPoseCollisionFree(const PoseState& state) const {
    return scene_.isPoseCollisionFree(state);
  }

  bool isEdgeCollisionFree(const PoseState& start,
                           const PoseState& finish) const {
    return scene_.isCertifiedEdgeCollisionFree(start, finish);
  }

  std::optional<double> nearestCollisionFreeYaw(
      const Eigen::Vector3d& position, double hinted_yaw,
      double reference_yaw) const {
    return scene_.nearestCollisionFreeYaw(position, hinted_yaw, reference_yaw);
  }

 private:
  PlannerConfig config_;
  mighty::frontend::WholeBodyCollisionScene scene_;
  ModelInfo info_;
};

PlanResult ShapeAwareAstarPlanner::Impl::plan(
    const PlanRequest& original_request, const CancelPredicate& cancelled) const {
  PlanRequest request = original_request;
  PlanResult result;
  const auto started = std::chrono::steady_clock::now();
  const auto outer_deadline =
      started + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(config_.timeout_seconds));
  mighty::frontend::QueryContext context(cancelled, outer_deadline);

  const auto copySceneStats = [&]() {
    result.statistics.collision_queries = context.collision_queries;
    result.statistics.collision_samples = context.collision_samples;
    result.statistics.edge_checks = context.edge_checks;
    result.statistics.cache_hits = context.pose_cache_hits;
    result.statistics.fcl_queries = context.fcl_queries;
    result.statistics.swept_aabb_candidates = context.swept_aabb_candidates;
    result.statistics.swept_sat_calls = context.swept_sat_calls;
  };

  const auto finish = [&](PlanStatus status, const std::string& message) {
    copySceneStats();
    if (status == PlanStatus::kSuccess && context.cancellationSeen()) {
      status = PlanStatus::kCancelled;
      result.waypoints.clear();
    } else if (status == PlanStatus::kSuccess && context.timeoutSeen() &&
               result.waypoints.size() < 2) {
      status = PlanStatus::kTimedOut;
    }
    result.status = status;
    result.message = status == PlanStatus::kCancelled
                         ? "cancelled"
                         : status == PlanStatus::kTimedOut ? "timed out"
                                                           : message;
    result.statistics.elapsed_seconds = std::chrono::duration<double>(
                                            std::chrono::steady_clock::now() -
                                            started)
                                            .count();
    result.statistics.yaw_cost_weight = config_.collision.yaw_cost_weight;
    return result;
  };

  if (!validPose(request.start) || !validPose(request.goal.pose)) {
    return finish(PlanStatus::kInvalidRequest,
                  "start or goal contains a non-finite value");
  }
  if (context.stopRequested()) {
    return finish(context.cancellationSeen() ? PlanStatus::kCancelled
                                             : PlanStatus::kTimedOut,
                  context.cancellationSeen() ? "cancelled" : "timed out");
  }

  const auto poseFree = [&](const PoseState& pose) {
    if (context.stopRequested()) return false;
    ++context.collision_samples;
    return scene_.isPoseCollisionFree(pose, &context);
  };
  const auto edgeFree = [&](const PoseState& start, const PoseState& finish,
                            mighty::frontend::EdgeValidationMode mode) {
    if (context.stopRequested()) return false;
    return scene_.isEdgeCollisionFree(start, finish, context, mode);
  };

  const bool yaw_invariant_body = info_.body_name == "sphere";
  const int uniform_yaw_count = yaw_invariant_body
                                    ? 1
                                    : std::max(4, static_cast<int>(std::ceil(
                                                      kTwoPi / config_.collision.yaw_resolution)));
  const double uniform_yaw_step =
      kTwoPi / static_cast<double>(uniform_yaw_count);
  std::vector<double> yaw_phases;
  yaw_phases.reserve(static_cast<std::size_t>(uniform_yaw_count + 1));
  for (int index = 0; index < uniform_yaw_count; ++index) {
    yaw_phases.push_back(uniform_yaw_step * static_cast<double>(index));
  }
  if (request.goal.yaw_constraint == GoalYawConstraint::kExact) {
    const bool duplicate = std::any_of(
        yaw_phases.begin(), yaw_phases.end(), [&](double phase) {
          return std::abs(angleDifference(request.start.yaw + phase,
                                          request.goal.pose.yaw)) <=
                 config_.collision.numeric_tolerance;
        });
    if (!duplicate) {
      yaw_phases.push_back(
          positiveAngle(request.goal.pose.yaw - request.start.yaw));
    }
  }
  yaw_phases.erase(std::unique(yaw_phases.begin(), yaw_phases.end(),
                               [&](double left, double right) {
                                 return std::abs(left - right) <=
                                        config_.collision.numeric_tolerance;
                               }),
                   yaw_phases.end());
  std::vector<double> yaws;
  yaws.reserve(yaw_phases.size());
  for (double phase : yaw_phases) yaws.push_back(request.start.yaw + phase);

  const auto closestYawIndex = [&](double yaw) {
    int best = 0;
    double difference = kInfinity;
    for (std::size_t index = 0; index < yaws.size(); ++index) {
      const double candidate = std::abs(angleDifference(yaw, yaws[index]));
      if (candidate < difference) {
        difference = candidate;
        best = static_cast<int>(index);
      }
    }
    return best;
  };
  const int start_yaw_index = closestYawIndex(request.start.yaw);

  std::vector<std::vector<int>> yaw_order(yaws.size());
  for (std::size_t from = 0; from < yaws.size(); ++from) {
    std::vector<int> ordered;
    ordered.reserve(yaws.size());
    for (std::size_t to = 0; to < yaws.size(); ++to) {
      const double difference =
          std::abs(angleDifference(yaws[to], yaws[from]));
      if (difference <= config_.collision.max_yaw_delta_per_step +
                            config_.collision.numeric_tolerance) {
        ordered.push_back(static_cast<int>(to));
      }
    }
    std::sort(ordered.begin(), ordered.end(), [&](int left, int right) {
      const auto axisPenalty = [&](int index) {
        const double yaw = yaws[static_cast<std::size_t>(index)];
        double best = std::abs(angleDifference(yaw, 0.0));
        best = std::min(best, std::abs(angleDifference(yaw, kPi / 2.0)));
        best = std::min(best, std::abs(angleDifference(yaw, kPi)));
        best = std::min(best, std::abs(angleDifference(yaw, -kPi / 2.0)));
        return best;
      };
      const double left_difference = std::abs(
          angleDifference(yaws[static_cast<std::size_t>(left)], yaws[from]));
      const double right_difference = std::abs(
          angleDifference(yaws[static_cast<std::size_t>(right)], yaws[from]));
      const bool left_axis =
          axisPenalty(left) <= config_.collision.numeric_tolerance;
      const bool right_axis =
          axisPenalty(right) <= config_.collision.numeric_tolerance;
      if (left_difference <= config_.collision.numeric_tolerance &&
          right_difference > config_.collision.numeric_tolerance) {
        return true;
      }
      if (right_difference <= config_.collision.numeric_tolerance &&
          left_difference > config_.collision.numeric_tolerance) {
        return false;
      }
      if (left_axis != right_axis) return left_axis;
      if (std::abs(left_difference - right_difference) >
          config_.collision.numeric_tolerance) {
        return left_difference < right_difference;
      }
      return left < right;
    });
    yaw_order[from] = std::move(ordered);
  }
  std::vector<std::vector<int>> goal_guided_yaw_order = yaw_order;
  if (request.goal.yaw_constraint == GoalYawConstraint::kExact) {
    for (auto& ordered : goal_guided_yaw_order) {
      if (ordered.size() < 2) continue;
      std::stable_sort(ordered.begin() + 1, ordered.end(),
                       [&](int left, int right) {
                         return std::abs(angleDifference(
                                    yaws[static_cast<std::size_t>(left)],
                                    request.goal.pose.yaw)) <
                                std::abs(angleDifference(
                                    yaws[static_cast<std::size_t>(right)],
                                    request.goal.pose.yaw));
                       });
    }
  }

  std::unordered_map<GridKey, bool, GridKeyHash> node_free_cache;
  const auto nodeFree = [&](const GridKey& key, const PoseState& pose) {
    const auto cached = node_free_cache.find(key);
    if (cached != node_free_cache.end()) {
      ++context.pose_cache_hits;
      return cached->second;
    }
    const bool free = poseFree(pose);
    node_free_cache.emplace(key, free);
    return free;
  };

  const GridKey start_key{0, 0, 0, start_yaw_index, 0};
  if (!nodeFree(start_key, request.start)) {
    if (context.stopRequested()) {
      return finish(context.cancellationSeen() ? PlanStatus::kCancelled
                                               : PlanStatus::kTimedOut,
                    context.cancellationSeen() ? "cancelled" : "timed out");
    }
    return finish(PlanStatus::kStartInCollision,
                  "exact start pose is in collision");
  }
  if (request.goal.yaw_constraint == GoalYawConstraint::kExact &&
      !poseFree(request.goal.pose)) {
    if (context.stopRequested()) {
      return finish(context.cancellationSeen() ? PlanStatus::kCancelled
                                               : PlanStatus::kTimedOut,
                    context.cancellationSeen() ? "cancelled" : "timed out");
    }
    if ((request.goal.pose.position - request.start.position).norm() <=
            config_.collision.numeric_tolerance &&
        std::abs(angleDifference(request.goal.pose.yaw, request.start.yaw)) >
            config_.collision.numeric_tolerance) {
      return finish(PlanStatus::kNoPath,
                    "pure-yaw requests are not supported by the frontend");
    }
    return finish(PlanStatus::kGoalInCollision,
                  "exact goal pose is in collision");
  }

  std::unordered_map<int, bool> goal_yaw_free_cache;
  const auto goalYawFree = [&](int yaw_index) {
    const auto cached = goal_yaw_free_cache.find(yaw_index);
    if (cached != goal_yaw_free_cache.end()) {
      ++context.pose_cache_hits;
      return cached->second;
    }
    const bool free = poseFree(PoseState{
        request.goal.pose.position, yaws[static_cast<std::size_t>(yaw_index)]});
    goal_yaw_free_cache.emplace(yaw_index, free);
    return free;
  };
  if (request.goal.yaw_constraint == GoalYawConstraint::kFree) {
    bool any_goal_yaw_free = false;
    for (std::size_t index = 0; index < yaws.size(); ++index) {
      if (goalYawFree(static_cast<int>(index))) {
        any_goal_yaw_free = true;
        break;
      }
    }
    if (context.stopRequested()) {
      return finish(context.cancellationSeen() ? PlanStatus::kCancelled
                                               : PlanStatus::kTimedOut,
                    context.cancellationSeen() ? "cancelled" : "timed out");
    }
    if (!any_goal_yaw_free) {
      return finish(PlanStatus::kGoalInCollision,
                    "goal position has no collision-free yaw");
    }
  }
  if (request.goal.yaw_constraint == GoalYawConstraint::kExact &&
      (request.goal.pose.position - request.start.position).norm() <=
          config_.collision.numeric_tolerance &&
      std::abs(angleDifference(request.goal.pose.yaw, request.start.yaw)) >
          config_.collision.numeric_tolerance) {
    return finish(PlanStatus::kNoPath,
                  "pure-yaw requests are not supported by the frontend");
  }

  double resolution = config_.collision.spatial_resolution;
  Eigen::Array3i minimum_index = Eigen::Array3i::Zero();
  Eigen::Array3i maximum_index = Eigen::Array3i::Zero();
  const auto inside = [&](const GridKey& key) {
    return key.x >= minimum_index.x() && key.x <= maximum_index.x() &&
           key.y >= minimum_index.y() && key.y <= maximum_index.y() &&
           key.z >= minimum_index.z() && key.z <= maximum_index.z();
  };
  const auto poseFor = [&](const GridKey& key) {
    PoseState pose;
    pose.position = request.start.position +
                    resolution * Eigen::Vector3d(static_cast<double>(key.x),
                                                 static_cast<double>(key.y),
                                                 static_cast<double>(key.z));
    pose.yaw = yaws[static_cast<std::size_t>(key.yaw)];
    return pose;
  };

  std::vector<SearchNode> nodes;
  std::unordered_map<GridKey, SearchScore, GridKeyHash> validated_best;
  std::unordered_map<GridKey, SearchScore, GridKeyHash> pending_best;
  std::unordered_map<GridKey, SearchScore, GridKeyHash> expanded;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueGreater> open;
  std::size_t sequence = 0;
  const auto heuristic = [&](const PoseState& pose) {
    double estimate = (request.goal.pose.position - pose.position).norm();
    if (request.goal.yaw_constraint == GoalYawConstraint::kExact) {
      estimate += config_.collision.yaw_cost_weight *
                  std::abs(angleDifference(request.goal.pose.yaw, pose.yaw));
    }
    return estimate;
  };

  const auto reconstruct = [&](std::size_t node_index) {
    std::vector<PoseState> reverse;
    for (std::int64_t cursor = static_cast<std::int64_t>(node_index);
         cursor >= 0; cursor = nodes[static_cast<std::size_t>(cursor)].parent) {
      reverse.push_back(nodes[static_cast<std::size_t>(cursor)].pose);
    }
    std::reverse(reverse.begin(), reverse.end());
    return reverse;
  };

  const auto finalizePath = [&](const std::vector<PoseState>& raw,
                                std::vector<PoseState> shortened) {
    if (shortened.size() < 2) return false;
    double path_length = 0.0;
    double vertical_travel = 0.0;
    double yaw_travel = 0.0;
    for (std::size_t edge = 0; edge + 1 < shortened.size(); ++edge) {
      const auto& first = shortened[edge];
      const auto& second = shortened[edge + 1];
      path_length += (second.position - first.position).norm();
      vertical_travel += std::abs(second.position.z() - first.position.z());
      const double edge_yaw_travel =
          std::abs(angleDifference(second.yaw, first.yaw));
      yaw_travel += edge_yaw_travel;
    }
    double maximum_z = -kInfinity;
    for (const auto& waypoint : shortened) {
      maximum_z = std::max(maximum_z, waypoint.position.z());
    }
    result.waypoints = std::move(shortened);
    result.statistics.raw_waypoints = raw.size();
    result.statistics.path_length = path_length;
    result.statistics.vertical_travel = vertical_travel;
    result.statistics.yaw_travel = yaw_travel;
    result.statistics.maximum_altitude_excursion = std::max(
        0.0, maximum_z - std::max(request.start.position.z(),
                                  request.goal.pose.position.z()));
    result.statistics.final_cost =
        path_length + config_.collision.yaw_cost_weight * yaw_travel;
    return true;
  };

  const auto certifyAdjacent = [&](const std::vector<PoseState>& path) {
    const auto cert_started = std::chrono::steady_clock::now();
    const std::size_t queries0 = context.collision_queries;
    const std::size_t samples0 = context.collision_samples;
    if (path.size() < 2) return false;
    for (std::size_t index = 0; index + 1 < path.size(); ++index) {
      if (!edgeFree(path[index], path[index + 1],
                    mighty::frontend::EdgeValidationMode::kCertified)) {
        result.statistics.certification.elapsed_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          cert_started)
                .count();
        return false;
      }
    }
    result.statistics.certification.elapsed_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      cert_started)
            .count();
    result.statistics.certification.collision_queries +=
        context.collision_queries - queries0;
    result.statistics.certification.collision_samples +=
        context.collision_samples - samples0;
    return true;
  };

  const auto tryExplicitPath = [&](std::vector<PoseState> raw, bool postprocess) {
    auto certified = scene_.cleanPosePath(raw);
    if (!certifyAdjacent(certified)) return false;
    if (result.statistics.first_certified_solution_seconds <= 0.0) {
      result.statistics.first_certified_solution_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        started)
              .count();
    }
    if (!postprocess) return finalizePath(raw, certified);

    const auto shortcut_started = std::chrono::steady_clock::now();
    const std::size_t q0 = context.collision_queries;
    const std::size_t s0 = context.collision_samples;
    auto shortened = scene_.simplifyPath(certified, context);
    result.statistics.shortcut.elapsed_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      shortcut_started)
            .count();
    result.statistics.shortcut.collision_queries +=
        context.collision_queries - q0;
    result.statistics.shortcut.collision_samples +=
        context.collision_samples - s0;
    if (context.cancellationSeen()) return false;
    std::vector<PoseState> best = certified;
    if (!shortened.empty()) {
      if (certifyAdjacent(shortened)) best = std::move(shortened);
    }

    const auto smooth_started = std::chrono::steady_clock::now();
    const std::size_t q1 = context.collision_queries;
    const std::size_t s1 = context.collision_samples;
    std::size_t smoothing_updates = 0;
    auto smoothed = scene_.smoothPath(best, context, smoothing_updates);
    result.statistics.smooth.elapsed_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      smooth_started)
            .count();
    result.statistics.smooth.collision_queries +=
        context.collision_queries - q1;
    result.statistics.smooth.collision_samples +=
        context.collision_samples - s1;
    result.statistics.smoothing_updates = smoothing_updates;
    if (context.cancellationSeen()) return false;
    if (!smoothed.empty()) {
      if (certifyAdjacent(smoothed)) {
        result.statistics.smoothing_applied = smoothing_updates > 0;
        result.statistics.smoothing_fallback = false;
        return finalizePath(raw, std::move(smoothed));
      }
    }
    result.statistics.smoothing_applied = false;
    result.statistics.smoothing_fallback =
        config_.collision.smoothing_enabled && smoothing_updates > 0;
    return finalizePath(raw, std::move(best));
  };

  const auto tryCandidatePath = [&](std::size_t node_index, bool postprocess) {
    std::vector<PoseState> raw = reconstruct(node_index);
    if (!raw.empty() &&
        (raw.front().position - original_request.start.position).norm() >
            config_.collision.numeric_tolerance) {
      std::reverse(raw.begin(), raw.end());
    }
    return tryExplicitPath(std::move(raw), postprocess);
  };

  struct PoseCandidate {
    GridKey key;
    PoseState pose;
    SearchScore score;
  };

  std::unordered_map<SpatialKey, std::vector<std::int8_t>, SpatialKeyHash>
      yaw_records;
  std::unordered_set<std::uint64_t> corridor_cells;
  bool use_corridor = false;

  const auto localPoseCandidates = [&](const GridKey& position_key,
                                       const Eigen::Vector3d& position,
                                       const PoseState& parent_pose,
                                       int parent_yaw_index,
                                       const SearchScore& parent_score,
                                       double step, std::size_t yaw_limit) {
    const auto& phase_yaw_order = position_key.phase == 2
        ? goal_guided_yaw_order
        : yaw_order;
    const std::vector<int>& ordered_yaw_indices =
        phase_yaw_order[static_cast<std::size_t>(parent_yaw_index)];
    SpatialKey spatial{position_key.x, position_key.y, position_key.z,
                       position_key.phase};
    auto record = yaw_records.find(spatial);
    if (record == yaw_records.end()) {
      record = yaw_records
                   .emplace(spatial, std::vector<std::int8_t>(yaws.size(), -1))
                   .first;
      ++result.statistics.yaw_records_built;
    } else {
      ++result.statistics.yaw_record_hits;
    }

    std::vector<PoseCandidate> candidates;
    candidates.reserve(yaw_limit);
    for (const int yaw_index : ordered_yaw_indices) {
      GridKey key = position_key;
      key.yaw = yaw_index;
      const PoseState pose{position, yaws[static_cast<std::size_t>(yaw_index)]};
      SearchScore score = parent_score;
      const double yaw_travel =
          std::abs(angleDifference(pose.yaw, parent_pose.yaw));
      score.cost += step + config_.collision.yaw_cost_weight * yaw_travel;
      score.distance += step;
      score.vertical =
          parent_score.vertical + std::abs(pose.position.z() - parent_pose.position.z());
      score.yaw_travel = parent_score.yaw_travel + yaw_travel;
      const auto validated = validated_best.find(key);
      if (validated != validated_best.end() &&
          !scoreLess(score, validated->second,
                     config_.collision.numeric_tolerance)) {
        ++result.statistics.dominated_before_collision;
        continue;
      }
      ++result.statistics.generated_edges;
      std::int8_t& yaw_state = record->second[static_cast<std::size_t>(yaw_index)];
      bool free = false;
      if (yaw_state >= 0) {
        free = yaw_state == 1;
      } else {
        free = nodeFree(key, pose);
        yaw_state = free ? 1 : 0;
      }
      if (!free) {
        continue;
      }
      candidates.push_back(PoseCandidate{key, pose, score});
      if (candidates.size() >= yaw_limit) break;
    }
    return candidates;
  };

  const auto relax = [&](std::size_t parent_index, const PoseCandidate& candidate,
                         bool edge_validated) {
    const auto validated = validated_best.find(candidate.key);
    if (validated != validated_best.end() &&
        !scoreLess(candidate.score, validated->second,
                   config_.collision.numeric_tolerance)) {
      return;
    }
    const auto pending = pending_best.find(candidate.key);
    if (pending != pending_best.end() &&
        !scoreLess(candidate.score, pending->second,
                   config_.collision.numeric_tolerance) &&
        !scoreEqual(candidate.score, pending->second,
                    config_.collision.numeric_tolerance)) {
      return;
    }
    pending_best[candidate.key] = candidate.score;
    const double h = heuristic(candidate.pose);
    nodes.push_back(SearchNode{candidate.key, candidate.pose, candidate.score,
                               static_cast<std::int64_t>(parent_index),
                               edge_validated});
    open.push(QueueEntry{candidate.score.cost + config_.heuristic_weight * h, h,
                         candidate.score.vertical, candidate.score.yaw_travel,
                         candidate.key.phase, sequence++, nodes.size() - 1,
                         candidate.score});
  };

  double goal_connection_radius = 0.0;
  const auto tryGoalConnection = [&](const SearchNode& current,
                                     std::size_t current_index,
                                     bool postprocess) {
    const Eigen::Vector3d delta =
        request.goal.pose.position - current.pose.position;
    const double distance = delta.norm();
    if (distance > goal_connection_radius) return false;

    std::vector<PoseState> goal_poses;
    if (request.goal.yaw_constraint == GoalYawConstraint::kExact) {
      const double yaw_delta =
          std::abs(angleDifference(request.goal.pose.yaw, current.pose.yaw));
      if (distance <= config_.collision.numeric_tolerance &&
          yaw_delta > config_.collision.numeric_tolerance) {
        return false;
      }
      if (distance > config_.collision.numeric_tolerance &&
          yaw_delta > config_.collision.max_yaw_delta_per_step +
                           config_.collision.numeric_tolerance) {
        return false;
      }
      ++result.statistics.generated_edges;
      goal_poses.push_back(request.goal.pose);
    } else {
      const std::vector<int>& ordered_yaw_indices =
          yaw_order[static_cast<std::size_t>(current.key.yaw)];
      for (const int yaw_index : ordered_yaw_indices) {
        ++result.statistics.generated_edges;
        if (distance <= config_.collision.numeric_tolerance &&
            std::abs(angleDifference(yaws[static_cast<std::size_t>(yaw_index)],
                                     current.pose.yaw)) >
                config_.collision.numeric_tolerance) {
          continue;
        }
        if (!goalYawFree(yaw_index)) {
          continue;
        }
        goal_poses.push_back(PoseState{
            request.goal.pose.position,
            yaws[static_cast<std::size_t>(yaw_index)]});
        if (goal_poses.size() >= config_.max_yaw_candidates) break;
      }
    }

    for (const PoseState& goal_pose : goal_poses) {
      if (context.stopRequested()) return false;
      const double yaw_travel =
          std::abs(angleDifference(goal_pose.yaw, current.pose.yaw));
      if (distance <= config_.collision.numeric_tolerance &&
          yaw_travel > config_.collision.numeric_tolerance) {
        continue;
      }
      if ((current.pose.position - goal_pose.position).squaredNorm() <=
              config_.collision.numeric_tolerance *
                  config_.collision.numeric_tolerance &&
          yaw_travel <= config_.collision.numeric_tolerance) {
        if (tryCandidatePath(current_index, postprocess)) return true;
        continue;
      }
      SearchScore goal_score = current.score;
      goal_score.cost +=
          distance + config_.collision.yaw_cost_weight * yaw_travel;
      goal_score.distance += distance;
      goal_score.vertical +=
          std::abs(goal_pose.position.z() - current.pose.position.z());
      goal_score.yaw_travel += yaw_travel;
      nodes.push_back(SearchNode{current.key, goal_pose, goal_score,
                                 static_cast<std::int64_t>(current_index),
                                 true});
      if (tryCandidatePath(nodes.size() - 1, postprocess)) return true;
    }
    return false;
  };

  const int nphases = 4;
  PlanResult coarse_keep;
  coarse_keep.status = PlanStatus::kNoPath;
  const auto returnCoarseFallback = [&](const std::string& message) {
    if (!coarse_keep.success() || coarse_keep.waypoints.size() < 2) {
      return finish(context.cancellationSeen() ? PlanStatus::kCancelled
                                               : PlanStatus::kTimedOut,
                    message);
    }
    const auto search_phases = result.statistics.search_phases;
    const auto fine_stats = result.statistics.fine;
    result = coarse_keep;
    for (std::size_t index = 0; index < search_phases.size(); ++index) {
      if (search_phases[index].ran) {
        result.statistics.search_phases[index] = search_phases[index];
      }
    }
    if (fine_stats.ran) result.statistics.fine = fine_stats;
    tryExplicitPath(coarse_keep.waypoints, true);
    return finish(PlanStatus::kSuccess, message);
  };
  std::vector<Eigen::Vector3d> band_centers;
  std::vector<Eigen::Vector3i> directions;

  // If the fine six-neighbour bridge is exhausted, try the same coarse
  // search from a safe terminal yaw. The shared request deadline still applies.
  bool reverse_middle = false;
  for (int phase = 0; phase < nphases; ++phase) {
    if (phase == 3 && reverse_middle) {
      request = original_request;
      goal_yaw_free_cache.clear();
    }
    const bool coarse_phase = phase < 2;
    const bool mid_phase = phase == 2;
    const bool fine_phase = phase == 3;
    if (phase >= 1 && phase <= 2 && !band_centers.empty()) {
      continue;
    }
    const std::uint8_t phase_id = static_cast<std::uint8_t>(phase);
    const std::size_t yaw_limit =
        (phase == 0 || (mid_phase && reverse_middle))
            ? std::size_t{1}
            : config_.max_yaw_candidates;
    // Free-yaw forward bridging uses a finer lattice; repeating phase 1
    // at the same resolution and yaw order cannot discover a new corridor.
    resolution = (coarse_phase ||
                  (mid_phase && request.goal.yaw_constraint ==
                                    GoalYawConstraint::kExact))
                     ? config_.coarse_spatial_resolution
                     : config_.collision.spatial_resolution;
    const auto phase_started = std::chrono::steady_clock::now();
    const std::size_t expansions0 = result.statistics.expansions;
    const std::size_t queries0 = context.collision_queries;
    const std::size_t samples0 = context.collision_samples;
    bool phase_found = false;
    const auto previous_phase_stats =
        result.statistics.search_phases[static_cast<std::size_t>(phase)];
    const auto recordPhase = [&]() {
      PhaseStatistics phase_stats;
      phase_stats.elapsed_seconds = std::chrono::duration<double>(
                                        std::chrono::steady_clock::now() -
                                        phase_started)
                                        .count();
      phase_stats.expansions = result.statistics.expansions - expansions0;
      phase_stats.collision_queries = context.collision_queries - queries0;
      phase_stats.collision_samples = context.collision_samples - samples0;
      phase_stats.corridor_formed = (coarse_phase || mid_phase) && phase_found;
      phase_stats.ran = true;
      if (reverse_middle && phase == 2) {
        const auto& previous = previous_phase_stats;
        phase_stats.elapsed_seconds += previous.elapsed_seconds;
        phase_stats.expansions += previous.expansions;
        phase_stats.collision_queries += previous.collision_queries;
        phase_stats.collision_samples += previous.collision_samples;
      }
      result.statistics.search_phases[static_cast<std::size_t>(phase)] =
          phase_stats;
      if (phase == 0) result.statistics.coarse = phase_stats;
      if (phase == 3) result.statistics.fine = phase_stats;
    };

    minimum_index =
        ((info_.map_min - request.start.position) / resolution).array().ceil().cast<int>();
    maximum_index =
        ((info_.map_max - request.start.position) / resolution).array().floor().cast<int>();

    const double band_radius = scene_.bodyRadius() + resolution;
    use_corridor = fine_phase && !band_centers.empty();
    if (use_corridor) {
      corridor_cells = rasterizeBallUnion(
          band_centers, request.start.position, resolution, band_radius,
          minimum_index, maximum_index, config_.collision.numeric_tolerance);
    } else {
      corridor_cells.clear();
    }
    const bool use_26 = fine_phase;
    goal_connection_radius =
        std::sqrt(use_26 ? 3.0 : 12.0) * resolution +
        config_.collision.numeric_tolerance;

    directions.clear();
    if (use_26) {
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            if (dx == 0 && dy == 0 && dz == 0) continue;
            directions.emplace_back(dx, dy, dz);
          }
        }
      }
    } else {
      directions = {Eigen::Vector3i(1, 0, 0),  Eigen::Vector3i(-1, 0, 0),
                    Eigen::Vector3i(0, 1, 0),  Eigen::Vector3i(0, -1, 0),
                    Eigen::Vector3i(0, 0, 1),  Eigen::Vector3i(0, 0, -1)};
    }

    const Eigen::Vector3d goal_delta =
        request.goal.pose.position - request.start.position;
    const int manhattan =
        std::abs(static_cast<int>(std::lround(goal_delta.x() / resolution))) +
        std::abs(static_cast<int>(std::lround(goal_delta.y() / resolution))) +
        std::abs(static_cast<int>(std::lround(goal_delta.z() / resolution)));
    std::size_t phase_expansion_limit = config_.max_expansions;
    if (phase == 0) {
      phase_expansion_limit = std::min(config_.max_expansions, std::size_t{8000});
    } else if (phase == 1) {
      phase_expansion_limit = std::min(config_.max_expansions, std::size_t{18500});
    } else if (phase == 2) {
      phase_expansion_limit =
          std::min(config_.max_expansions, std::size_t{8000});
    } else if (use_corridor) {
      phase_expansion_limit = std::min(
          config_.max_expansions,
          std::max<std::size_t>(
              8000, 80ull * static_cast<std::size_t>(std::max(1, manhattan))));
    }

    nodes.clear();
    validated_best.clear();
    pending_best.clear();
    expanded.clear();
    open = {};
    sequence = 0;
    yaw_records.clear();
    nodes.reserve(std::min<std::size_t>(config_.max_expansions, 100000));
    GridKey phase_start = start_key;
    phase_start.phase = phase_id;
    phase_start.yaw = closestYawIndex(request.start.yaw);
    nodes.push_back(SearchNode{phase_start, request.start, {}, -1, true});
    validated_best.emplace(phase_start, SearchScore{});
    const double start_h = heuristic(request.start);
    open.push(QueueEntry{config_.heuristic_weight * start_h, start_h, 0.0, 0.0,
                         phase_id, sequence++, 0, {}});

    std::size_t phase_expansions = 0;
    while (!open.empty()) {
      if (context.stopRequested() ||
          (fine_phase && coarse_keep.success() &&
           std::chrono::steady_clock::now() >=
               started + std::chrono::duration_cast<
                             std::chrono::steady_clock::duration>(
                             std::chrono::duration<double>(
                                 0.55 * config_.timeout_seconds)))) {
        if (coarse_phase || mid_phase) break;
        recordPhase();
        if (coarse_keep.success()) {
          return returnCoarseFallback(
              "coarse fallback after fine-layer budget");
        }
        if (result.waypoints.size() >= 2) {
          return finish(PlanStatus::kSuccess,
                        "certified path returned before postprocess timeout");
        }
        return finish(context.cancellationSeen() ? PlanStatus::kCancelled
                                                 : PlanStatus::kTimedOut,
                      context.cancellationSeen() ? "cancelled" : "timed out");
      }
      const QueueEntry queued = open.top();
      open.pop();
      SearchNode current = nodes[queued.node];
      const auto already_validated = validated_best.find(current.key);
      if (already_validated != validated_best.end() &&
          !scoreEqual(already_validated->second, queued.score,
                      config_.collision.numeric_tolerance) &&
          !scoreLess(queued.score, already_validated->second,
                     config_.collision.numeric_tolerance)) {
        continue;
      }
      const auto already_expanded = expanded.find(current.key);
      if (already_expanded != expanded.end() &&
          scoreEqual(already_expanded->second, current.score,
                     config_.collision.numeric_tolerance)) {
        continue;
      }
      if (current.parent >= 0 && !current.edge_validated) {
        const PoseState& parent_pose =
            nodes[static_cast<std::size_t>(current.parent)].pose;
        if (!edgeFree(parent_pose, current.pose,
                      mighty::frontend::EdgeValidationMode::kSearch)) {
          const auto pending = pending_best.find(current.key);
          if (pending != pending_best.end() &&
              scoreEqual(pending->second, queued.score,
                         config_.collision.numeric_tolerance)) {
            pending_best.erase(pending);
          }
          continue;
        }
        current.edge_validated = true;
        nodes[queued.node].edge_validated = true;
      }
      validated_best[current.key] = current.score;
      expanded[current.key] = current.score;
      ++phase_expansions;
      if (++result.statistics.expansions > config_.max_expansions ||
          phase_expansions > phase_expansion_limit) {
        if (coarse_phase || mid_phase) break;
        recordPhase();
        if (coarse_keep.success()) {
          return returnCoarseFallback("coarse fallback after expansion limit");
        }
        if (result.waypoints.size() >= 2) {
          return finish(PlanStatus::kSuccess,
                        "certified path returned at expansion limit");
        }
        return finish(PlanStatus::kNoPath,
                      "geometric search reached max_expansions");
      }

      if (tryGoalConnection(current, queued.node, false)) {
        if (coarse_phase || mid_phase) {
          band_centers.clear();
          constexpr double kBandSkeletonSpacing = 0.4;
          const std::vector<PoseState> skeleton =
              result.waypoints.empty() ? reconstruct(queued.node)
                                       : result.waypoints;
          for (const auto& waypoint : skeleton) {
            if (band_centers.empty()) {
              band_centers.push_back(waypoint.position);
              continue;
            }
            const Eigen::Vector3d previous = band_centers.back();
            const double span = (waypoint.position - previous).norm();
            const int steps = std::max(
                1, static_cast<int>(std::ceil(span / kBandSkeletonSpacing)));
            for (int step = 1; step <= steps; ++step) {
              const double t = static_cast<double>(step) /
                               static_cast<double>(steps);
              band_centers.push_back(previous +
                                     t * (waypoint.position - previous));
            }
          }
          result.statistics.coarse_corridor_formed = true;
          phase_found = true;
          recordPhase();
          coarse_keep = result;
          coarse_keep.status = PlanStatus::kSuccess;
          break;
        }
        tryExplicitPath(result.waypoints, true);
        recordPhase();
        return finish(PlanStatus::kSuccess,
                      "hierarchical Weighted A* with certified shortcutting");
      }

      for (const Eigen::Vector3i& direction : directions) {
        GridKey position_key = current.key;
        position_key.x += direction.x();
        position_key.y += direction.y();
        position_key.z += direction.z();
        position_key.phase = phase_id;
        if (!inside(position_key)) continue;
        const Eigen::Vector3d position = poseFor(position_key).position;
        if (use_corridor) {
          ++result.statistics.membership_queries;
          if (!corridor_cells.count(packGridCell(
                  position_key.x, position_key.y, position_key.z))) {
            continue;
          }
        }
        const double step = resolution * direction.cast<double>().norm();
        const auto candidates = localPoseCandidates(
            position_key, position, current.pose, current.key.yaw,
            current.score, step, yaw_limit);
        if (context.stopRequested()) break;
        for (const PoseCandidate& candidate : candidates) {
          relax(queued.node, candidate, false);
        }
      }
      if (coarse_phase && context.stopRequested()) break;
    }

    recordPhase();
    if ((coarse_phase || mid_phase) && !phase_found) band_centers.clear();
    if (mid_phase && !phase_found && !reverse_middle &&
        original_request.goal.yaw_constraint == GoalYawConstraint::kFree &&
        !context.stopRequested()) {
      for (std::size_t index = 0; index < yaws.size(); ++index) {
        if (!goalYawFree(static_cast<int>(index))) continue;
        request.start =
            PoseState{original_request.goal.pose.position, yaws[index]};
        request.goal.pose = original_request.start;
        request.goal.yaw_constraint = GoalYawConstraint::kExact;
        reverse_middle = true;
        node_free_cache.clear();
        goal_yaw_free_cache.clear();
        --phase;
        break;
      }
    }
  }

  if (coarse_keep.success()) {
    return returnCoarseFallback("coarse fallback after fine-layer exhaustion");
  }
  if (result.waypoints.size() >= 2) {
    return finish(PlanStatus::kSuccess, "certified path after search");
  }
  return finish(PlanStatus::kNoPath,
                "OPEN exhausted without a node-safe Pose path");
}

ShapeAwareAstarPlanner::ShapeAwareAstarPlanner(
    const PlannerConfig& config, const std::string& map_polymap_path,
    const std::string& body_polymap_path)
    : impl_(std::make_unique<Impl>(config, map_polymap_path,
                                   body_polymap_path)) {}

ShapeAwareAstarPlanner::~ShapeAwareAstarPlanner() = default;
ShapeAwareAstarPlanner::ShapeAwareAstarPlanner(
    ShapeAwareAstarPlanner&&) noexcept = default;
ShapeAwareAstarPlanner& ShapeAwareAstarPlanner::operator=(
    ShapeAwareAstarPlanner&&) noexcept = default;

const ModelInfo& ShapeAwareAstarPlanner::modelInfo() const {
  return impl_->modelInfo();
}

PlanResult ShapeAwareAstarPlanner::plan(const PlanRequest& request,
                                        CancelPredicate cancelled) const {
  const auto started = std::chrono::steady_clock::now();
  try {
    PlanResult result = impl_->plan(request, cancelled);
    result.statistics.elapsed_seconds = std::chrono::duration<double>(
                                            std::chrono::steady_clock::now() -
                                            started)
                                            .count();
    return result;
  } catch (const std::exception& error) {
    PlanResult result;
    result.status = PlanStatus::kInternalError;
    result.message = error.what();
    result.statistics.elapsed_seconds = std::chrono::duration<double>(
                                            std::chrono::steady_clock::now() -
                                            started)
                                            .count();
    return result;
  }
}

bool ShapeAwareAstarPlanner::isPoseCollisionFree(const PoseState& state) const {
  return impl_->isPoseCollisionFree(state);
}

bool ShapeAwareAstarPlanner::isEdgeCollisionFree(
    const PoseState& start, const PoseState& finish) const {
  return impl_->isEdgeCollisionFree(start, finish);
}

std::optional<double> ShapeAwareAstarPlanner::nearestCollisionFreeYaw(
    const Eigen::Vector3d& position, double hinted_yaw,
    double reference_yaw) const {
  return impl_->nearestCollisionFreeYaw(position, hinted_yaw, reference_yaw);
}

}  // namespace mighty::shape_aware_astar
