#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace mighty::frontend {

struct PoseState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  double yaw{0.0};
};

enum class GoalYawConstraint : std::uint8_t {
  kFree,
  kExact,
};

struct Goal {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PoseState pose;
  GoalYawConstraint yaw_constraint{GoalYawConstraint::kFree};
};

struct PlanRequest {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PoseState start;
  Goal goal;
};

enum class PlanStatus : std::uint8_t {
  kSuccess,
  kCancelled,
  kTimedOut,
  kNoPath,
  kCertificationFailed,
  kInvalidRequest,
  kStartInCollision,
  kGoalInCollision,
  kInternalError,
};

const char* toString(PlanStatus status);

struct PhaseStatistics {
  double elapsed_seconds{0.0};
  std::size_t expansions{0};
  std::size_t collision_queries{0};
  std::size_t collision_samples{0};
  bool corridor_formed{false};
  bool ran{false};
};

struct PlanStatistics {
  std::size_t expansions{0};
  std::size_t generated_edges{0};
  std::size_t edge_checks{0};
  std::size_t collision_queries{0};
  std::size_t collision_samples{0};
  std::size_t cache_hits{0};
  std::size_t dominated_before_collision{0};
  std::size_t yaw_records_built{0};
  std::size_t yaw_record_hits{0};
  std::size_t membership_queries{0};
  std::size_t fcl_queries{0};
  std::size_t swept_aabb_candidates{0};
  std::size_t swept_sat_calls{0};
  std::size_t smoothing_updates{0};
  std::size_t raw_waypoints{0};
  std::size_t final_waypoints{0};
  double first_certified_solution_seconds{0.0};
  double elapsed_seconds{0.0};
  double path_length{0.0};
  double yaw_travel{0.0};
  double vertical_travel{0.0};
  double maximum_altitude_excursion{0.0};
  double final_cost{0.0};
  double yaw_cost_weight{0.20};
  bool smoothing_applied{false};
  bool smoothing_fallback{false};
  bool local_retry{false};
  bool coarse_corridor_formed{false};
  PhaseStatistics coarse;
  PhaseStatistics fine;
  // Index 0: 0.4 m 1-yaw 6-connect; 1: 0.4 m 5-yaw 6-connect;
  // 2: 0.2 m 5-yaw 6-connect; 3: 0.2 m 26-connect. Skipped layers have ran=false.
  std::array<PhaseStatistics, 4> search_phases{};
  PhaseStatistics shortcut;
  PhaseStatistics smooth;
  PhaseStatistics certification;
  std::string termination_reason;
};

struct PlanResult {
  PlanStatus status{PlanStatus::kInternalError};
  std::string message;
  std::vector<PoseState> waypoints;
  PlanStatistics statistics;

  bool success() const { return status == PlanStatus::kSuccess; }
};

struct ModelInfo {
  std::string map_name;
  std::string map_frame;
  std::string body_name;
  std::size_t obstacle_components{0};
  std::size_t body_components{0};
  Eigen::Vector3d map_min{Eigen::Vector3d::Zero()};
  Eigen::Vector3d map_max{Eigen::Vector3d::Zero()};
};

struct CollisionConfig {
  static constexpr double kVirtualCeilingThickness = 0.05;

  double body_clearance{0.0};
  double collision_sample_max_distance{0.20};
  double collision_sample_max_yaw_step{0.17453292519943295};
  double virtual_ceiling_height{4.5};
  double numeric_tolerance{1e-8};
  double yaw_cost_weight{0.20};
  double spatial_resolution{0.20};
  double max_yaw_delta_per_step{1.5707963267948966};
  double yaw_resolution{0.17453292519943295};
  bool smoothing_enabled{true};
  std::size_t smoothing_iterations{4};
  double smoothing_alpha{0.35};
};

struct FrontendConfig {
  CollisionConfig collision;
  double timeout_seconds{1.0};
  std::size_t max_expansions{250000};
  double heuristic_weight{1.5};
  std::size_t max_yaw_candidates{5};
  double coarse_spatial_resolution{0.40};
};

using CancelPredicate = std::function<bool()>;

}  // namespace mighty::frontend
