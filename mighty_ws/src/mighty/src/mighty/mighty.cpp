/* ----------------------------------------------------------------------------
 * Copyright 2025, Kota Kondo, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Kota Kondo, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include "mighty/mighty.hpp"
#include "mighty/topp_seed.hpp"

#include <cstring>

namespace {
thread_local std::function<void(const char*)> planning_stage_observer_;
}

thread_local const char* planning_stage_limit_ = nullptr;

void MIGHTY::setPlanningStageObserver(std::function<void(const char*)> observer) {
  planning_stage_observer_ = std::move(observer);
}

void MIGHTY::setPlanningStageLimit(const char* stage) {
  planning_stage_limit_ = stage;
}

namespace {
bool limitedTo(const char* stage) {
  return planning_stage_limit_ != nullptr &&
         std::strcmp(planning_stage_limit_, stage) == 0;
}
}  // namespace

#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <stdexcept>

using namespace mighty;

namespace {

constexpr double kInteriorSpeedTolerance = 1.0e-6;
constexpr std::size_t kLongRiskRunMinimumEdges = 12;
constexpr std::size_t kRiskWindowEdges = 4;
constexpr std::size_t kRiskWindowStride = 3;

std::vector<std::vector<bool>> makeOverlappingRiskWindows(
    const std::vector<bool>& risk_segments) {
  std::vector<std::vector<bool>> windows;
  std::size_t run_begin = 0;
  while (run_begin < risk_segments.size()) {
    while (run_begin < risk_segments.size() && !risk_segments[run_begin]) {
      ++run_begin;
    }
    if (run_begin == risk_segments.size()) break;
    std::size_t run_end = run_begin;
    while (run_end < risk_segments.size() && risk_segments[run_end]) {
      ++run_end;
    }
    if (run_end - run_begin >= kLongRiskRunMinimumEdges) {
      for (std::size_t window_begin = run_begin; window_begin < run_end;
           window_begin += kRiskWindowStride) {
        const std::size_t window_end =
            std::min(window_begin + kRiskWindowEdges, run_end);
        std::vector<bool> window(risk_segments.size(), false);
        std::fill(window.begin() + static_cast<std::ptrdiff_t>(window_begin),
                  window.begin() + static_cast<std::ptrdiff_t>(window_end),
                  true);
        windows.push_back(std::move(window));
        if (window_end == run_end) break;
      }
    }
    run_begin = run_end;
  }
  return windows;
}

class ScopedFlagReset {
 public:
  explicit ScopedFlagReset(bool& flag) : flag_(flag) {}
  ~ScopedFlagReset() { flag_ = false; }

  ScopedFlagReset(const ScopedFlagReset&) = delete;
  ScopedFlagReset& operator=(const ScopedFlagReset&) = delete;

 private:
  bool& flag_;
};

TrajectoryQualityMetrics evaluateTrajectoryQuality(
    const std::vector<state>& samples, double minimum_interior_speed) {
  TrajectoryQualityMetrics metrics;
  if (samples.size() < 2) return metrics;

  metrics.available = true;
  metrics.duration_seconds =
      std::max(0.0, samples.back().t - samples.front().t);
  std::vector<double> speeds;
  speeds.reserve(samples.size());
  for (std::size_t index = 0; index < samples.size(); ++index) {
    const double speed = samples[index].vel.norm();
    speeds.push_back(speed);
    metrics.maximum_speed = std::max(metrics.maximum_speed, speed);
    metrics.maximum_acceleration = std::max(
        metrics.maximum_acceleration, samples[index].accel.norm());
    metrics.maximum_jerk = std::max(
        metrics.maximum_jerk, samples[index].jerk.norm());
    if (index > 0) {
      metrics.path_length +=
          (samples[index].pos - samples[index - 1].pos).norm();
    }
  }

  const auto first_moving = std::find_if(
      speeds.begin(), speeds.end(), [minimum_interior_speed](double speed) {
        return speed + kInteriorSpeedTolerance >= minimum_interior_speed;
      });
  const auto last_moving = std::find_if(
      speeds.rbegin(), speeds.rend(), [minimum_interior_speed](double speed) {
        return speed + kInteriorSpeedTolerance >= minimum_interior_speed;
      });
  if (first_moving == speeds.end()) {
    metrics.has_internal_stop = metrics.path_length > 1.0e-6;
    metrics.maximum_subthreshold_duration = metrics.duration_seconds;
    return metrics;
  }

  const std::size_t first = static_cast<std::size_t>(
      std::distance(speeds.begin(), first_moving));
  const std::size_t last = speeds.size() - 1U - static_cast<std::size_t>(
      std::distance(speeds.rbegin(), last_moving));
  metrics.minimum_interior_speed = std::numeric_limits<double>::infinity();
  double speed_sum = 0.0;
  double squared_speed_sum = 0.0;
  std::size_t speed_count = 0;
  std::optional<std::size_t> slow_run_start;
  for (std::size_t index = first; index <= last; ++index) {
    const double speed = speeds[index];
    metrics.minimum_interior_speed =
        std::min(metrics.minimum_interior_speed, speed);
    speed_sum += speed;
    squared_speed_sum += speed * speed;
    ++speed_count;
    if (speed + kInteriorSpeedTolerance < minimum_interior_speed) {
      metrics.has_internal_stop = true;
      if (!slow_run_start) slow_run_start = index;
    } else if (slow_run_start) {
      metrics.maximum_subthreshold_duration = std::max(
          metrics.maximum_subthreshold_duration,
          samples[index].t - samples[*slow_run_start].t);
      slow_run_start.reset();
    }
  }
  if (slow_run_start) {
    metrics.maximum_subthreshold_duration = std::max(
        metrics.maximum_subthreshold_duration,
        samples[last].t - samples[*slow_run_start].t);
  }
  if (speed_count > 0 && speed_sum > 0.0) {
    const double mean = speed_sum / static_cast<double>(speed_count);
    const double variance = std::max(
        0.0, squared_speed_sum / static_cast<double>(speed_count) -
                 mean * mean);
    metrics.speed_coefficient_of_variation = std::sqrt(variance) / mean;
  }
  return metrics;
}

}  // namespace

const frontend::ModelInfo& MIGHTY::shapeModelInfo() const {
  return shape_planner_->modelInfo();
}

// ----------------------------------------------------------------------------

/**
 * @brief Constructor for MIGHTY.
 * @param parameters par: Input configuration parameters.
 */
MIGHTY::MIGHTY(parameters par,
               const frontend::FrontendConfig& frontend_config,
               const std::string& map_polymap_path,
               const std::string& body_polymap_path)
    : par_(par),
      shape_planner_(
          std::make_unique<shape_aware_astar::ShapeAwareAstarPlanner>(
              frontend_config, map_polymap_path, body_polymap_path)) {
  // Set up the planner parameters
  planner_params_.verbose = false;     // enable verbose output
  planner_params_.V_max = par_.v_max;  // max velocity
  planner_params_.A_max = par_.a_max;  // max acceleration
  planner_params_.J_max = par_.j_max;  // max jerk
  planner_params_.time_weight = par_.time_weight;  // weight for time cost
  planner_params_.pos_anchor_weight = par_.pos_anchor_weight;
  if (par_.planning_mode == "global_once") {
    planner_params_.minimum_forward_speed =
        1.25 * par_.global_once_minimum_interior_speed;
    planner_params_.forward_progress_weight =
        par_.time_weight /
        std::pow(par_.global_once_minimum_interior_speed, 2.0);
  }
  planner_params_.jerk_weight = par_.jerk_weight;
  planner_params_.yaw_jerk_weight = par_.yaw_jerk_weight;
  planner_params_.yaw_reference_weight = par_.yaw_reference_weight;
  planner_params_.spatial_progress_yaw_coupling =
      par_.spatial_progress_yaw_coupling;
  planner_params_.w_max = par_.w_max;
  planner_params_.yaw_acc_max = par_.yaw_acc_max;
  planner_params_.dyn_constr_vel_weight = par_.dyn_constr_vel_weight;
  planner_params_.dyn_constr_acc_weight = par_.dyn_constr_acc_weight;
  planner_params_.dyn_constr_jerk_weight = par_.dyn_constr_jerk_weight;
  planner_params_.dyn_constr_bodyrate_weight = par_.dyn_constr_bodyrate_weight;
  planner_params_.dyn_constr_tilt_weight = par_.dyn_constr_tilt_weight;
  planner_params_.dyn_constr_thrust_weight = par_.dyn_constr_thrust_weight;
  planner_params_.dc = par_.dc;  // discretization constant
  planner_params_.spline_degree = par_.spline_degree;
  planner_params_.topp_init = par_.init_mode != "legacy";
  planner_params_.light_activation_ladder = par_.light_activation_ladder;

  // Set up the L-BFGS parameters
  lbfgs_params_.mem_size = 256;
  lbfgs_params_.min_step = 1.0e-32;
  lbfgs_params_.f_dec_coeff = par_.f_dec_coeff;          // allow larger Armijo steps
  lbfgs_params_.cautious_factor = par_.cautious_factor;  // always accept BFGS update
  lbfgs_params_.past = par_.past;
  lbfgs_params_.max_linesearch = par_.max_linesearch;  // fewer backtracking tries
  lbfgs_params_.max_iterations = par_.max_iterations;  // allow more iterations
  lbfgs_params_.g_epsilon = par_.g_epsilon;
  lbfgs_params_.delta = par_.delta;  // stop once f-improvement is minimal
  // Set up unconstrained optimization solver for whole trajectory
  whole_traj_solver_ptr_ = std::make_shared<lbfgs::SolverLBFGS>();
  whole_traj_solver_ptr_->initializeSolver(planner_params_);

  // The first whole-body backend is intentionally scoped to the confirmed
  // global-once / known-PCD / UAV pipeline.
  if (par_.planning_mode == "global_once" && par_.map_mode == "known_pcd") {
    auto scene = whole_body_safety::SafetyScene::load(
        map_polymap_path, body_polymap_path, par_.planner_Co,
        std::min(par_.z_max, frontend_config.collision.virtual_ceiling_height),
        par_.safety_beta_soft);
    safety_scene_ =
        std::make_shared<const whole_body_safety::SafetyScene>(std::move(scene));
    whole_body_safety::ExponentialHingeOptions penalty;
    penalty.beta_soft = par_.safety_beta_soft;
    penalty.acceptance_beta = par_.safety_acceptance_beta;
    penalty.weight = par_.stat_weight;
    penalty.exponential_gain = par_.safety_exponential_gain;
    penalty.hard_weight_multiplier = par_.safety_hard_weight_multiplier;
    whole_body_safety::ProximityOptions warm;
    warm.residual_tolerance = par_.safety_ipm_tolerance;
    warm.initial_barrier = 1.0e-5;
    warm.kkt_regularization = par_.safety_ipm_regularization;
    warm.max_newton_iterations = par_.safety_ipm_warm_iterations;
    whole_body_safety::ProximityOptions cold = warm;
    cold.max_newton_iterations = par_.safety_ipm_cold_iterations;
    whole_body_safety_ = std::make_shared<whole_body_safety::WholeBodySafety>(
        safety_scene_, penalty, warm, cold);
  }

  // Set up basis converter
  BasisConverter basis_converter;
  A_rest_pos_basis_ = basis_converter.getArestMinvo();  // Use Minvo basis
  A_rest_pos_basis_inverse_ = A_rest_pos_basis_.inverse();

  // Parameters
  v_max_3d_ = Eigen::Vector3d(par_.v_max, par_.v_max, par_.v_max);
  v_max_ = par_.v_max;
  a_max_3d_ = Eigen::Vector3d(par_.a_max, par_.a_max, par_.a_max);
  j_max_3d_ = Eigen::Vector3d(par_.j_max, par_.j_max, par_.j_max);

  // Initialize the state
  changeDroneStatus(DroneStatus::GOAL_REACHED);

  max_dist_vertexes_ = par_.max_dist_vertexes;
}

// ----------------------------------------------------------------------------

void MIGHTY::resetData() {
  final_g_ = 0.0;
  global_planning_time_ = 0.0;
  initial_guess_computation_time_ = 0.0;
  trajectory_optimization_time_ = 0.0;
  trajectory_route_ = TrajectoryRoute::NONE;
  {
    std::lock_guard<std::mutex> lock(mtx_plan_);
    executable_traj_solver_.reset();
    trajectory_execution_epoch_ = std::numeric_limits<double>::quiet_NaN();
  }
  middle_stage_time_ms_ = 0.0;
  seed_stage_time_ms_ = refinement_stage_time_ms_ = -1.0;
  back_end_preparation_time_ms_ = 0.0;
  back_end_optimization_time_ms_ = 0.0;
  back_end_retime_time_ms_ = 0.0;
  trajectory_export_time_ms_ = 0.0;
  plan_commit_time_ms_ = 0.0;
  derivative_activation_scales_.clear();
  optimization_status_ = 0;
  refinement_timed_out_ = false;
  risk_segment_count_ = 0;
  risk_segment_edges_ = 0;
  risk_window_count_ = 0;
  trajectory_segment_count_ = 0;
  trajectory_yaw_variation_ = -1.0;

  middle_goal_setpoints_.clear();
  goal_setpoints_.clear();
  pwp_to_share_.clear();
  cps_.clear();
}

// ----------------------------------------------------------------------------

void MIGHTY::retrieveData(double& final_g, double& global_planning_time,
                          double& initial_guess_computation_time,
                          double& trajectory_optimization_time) {
  final_g = final_g_;
  global_planning_time = global_planning_time_;
  initial_guess_computation_time = initial_guess_computation_time_;
  trajectory_optimization_time = trajectory_optimization_time_;
}

// ----------------------------------------------------------------------------

void MIGHTY::retrieveGoalSetpoints(std::vector<state>& goal_setpoints) {
  goal_setpoints = goal_setpoints_;
}

// ----------------------------------------------------------------------------

void MIGHTY::retrieveMiddleGoalSetpoints(std::vector<state>& goal_setpoints) {
  goal_setpoints = middle_goal_setpoints_;
}

// ----------------------------------------------------------------------------

void MIGHTY::retrieveCPs(std::vector<Eigen::Matrix<double, 3, 6>>& cps) { cps = cps_; }

// ----------------------------------------------------------------------------

GlobalPlanResult MIGHTY::planGlobalOnce(const state& terminal_goal, double request_time) {
  std::lock_guard<std::mutex> transaction_lock(global_once_plan_mutex_);
  const auto planning_started = std::chrono::steady_clock::now();
  global_once_hard_deadline_ = planning_started +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(par_.global_once_hard_timeout_seconds));
  global_once_refinement_deadline_ = planning_started +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(
              par_.global_once_refinement_timeout_seconds));
  global_once_deadline_active_ = true;
  const ScopedFlagReset deadline_guard(global_once_deadline_active_);
  const std::uint64_t generation = global_once_generation_.load();
  resetData();
  active_shape_path_.clear();
  active_goal_yaw_exact_ = terminal_goal.use_tracking_yaw;

  if (par_.planning_mode != "global_once" || !state_initialized_ ||
      !shape_planner_) {
    return {GlobalPlanStage::NOT_READY, {},
            "global_once planner or known map is not ready"};
  }
  const auto frontend_started = std::chrono::steady_clock::now();
  const auto finish_frontend = [&](GlobalPlanResult result) {
    result.frontend_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - frontend_started).count();
    global_once_deadline_active_ = false;
    return result;
  };

  // Capture the measured position at request time. Global-once intentionally
  // starts and ends at rest, independently of the velocity estimate.
  state captured_start;
  getState(captured_start);
  captured_start.vel.setZero();
  captured_start.accel.setZero();
  captured_start.jerk.setZero();
  captured_start.setTimeStamp(request_time);

  state exact_goal = terminal_goal;
  exact_goal.vel.setZero();
  exact_goal.accel.setZero();
  exact_goal.jerk.setZero();
  if (terminal_goal.use_tracking_yaw) {
    const auto selected_yaw = shape_planner_->nearestCollisionFreeYaw(
        terminal_goal.pos, terminal_goal.yaw, captured_start.yaw);
    if (!selected_yaw) {
      frontend::PlanResult shape_result;
      shape_result.status = frontend::PlanStatus::kGoalInCollision;
      shape_result.message = "goal has no collision-free yaw near the hint";
      return finish_frontend(
          {GlobalPlanStage::SHAPE_ASTAR_FAILED, std::move(shape_result),
           "goal has no collision-free yaw near the hint"});
    }
    exact_goal.yaw = captured_start.yaw + std::remainder(
        *selected_yaw - captured_start.yaw, 2.0 * M_PI);
  } else {
    exact_goal.yaw = captured_start.yaw;
  }
  exact_goal.dyaw = 0.0;
  exact_goal.use_tracking_yaw = true;

  frontend::PlanRequest shape_request;
  shape_request.start = {captured_start.pos, captured_start.yaw};
  shape_request.goal.pose = {exact_goal.pos, exact_goal.yaw};
  shape_request.goal.yaw_constraint =
      terminal_goal.use_tracking_yaw
          ? frontend::GoalYawConstraint::kExact
          : frontend::GoalYawConstraint::kFree;

  frontend::PlanResult shape_result;
  if ((captured_start.pos - exact_goal.pos).norm() <= par_.goal_radius) {
    const bool pose_safe = shape_planner_->isPoseCollisionFree(shape_request.start) &&
                           shape_planner_->isPoseCollisionFree(shape_request.goal.pose);
    const bool edge_safe = !terminal_goal.use_tracking_yaw ||
                           shape_planner_->isEdgeCollisionFree(
                               shape_request.start, shape_request.goal.pose);
    if (!pose_safe || !edge_safe) {
      shape_result.status = frontend::PlanStatus::kGoalInCollision;
      shape_result.message = "yaw-only Pose transition is not shape-safe";
      const std::string message = shape_result.message;
      return finish_frontend(
          {GlobalPlanStage::SHAPE_ASTAR_FAILED, std::move(shape_result),
           message});
    }
    shape_result.status = frontend::PlanStatus::kSuccess;
    shape_result.message = "shape-safe yaw-only transition";
    shape_result.waypoints = {shape_request.start, shape_request.goal.pose};
  } else {
    shape_result = shape_planner_->plan(
        shape_request, [this, generation]() {
          return global_once_generation_.load() != generation ||
                 (global_once_deadline_active_ &&
                  std::chrono::steady_clock::now() >=
                      global_once_hard_deadline_);
        });
    if (!shape_result.success()) {
      const std::string message = shape_result.message;
      return finish_frontend(
          {GlobalPlanStage::SHAPE_ASTAR_FAILED, std::move(shape_result),
           message});
    }
  }
  global_planning_time_ = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - frontend_started).count();
  if (planning_stage_observer_) planning_stage_observer_("idle");
  if (!terminal_goal.use_tracking_yaw && !shape_result.waypoints.empty()) {
    exact_goal.yaw = captured_start.yaw + std::remainder(
        shape_result.waypoints.back().yaw - captured_start.yaw, 2.0 * M_PI);
    exact_goal.use_tracking_yaw = true;
  }
  if (limitedTo("frontend")) {
    return finish_frontend(
        {GlobalPlanStage::SUCCESS, std::move(shape_result), {}});
  }

  return executeCertifiedShapePath(
      captured_start, exact_goal, std::move(shape_result), request_time);
}

GlobalPlanResult MIGHTY::planCertifiedShapePath(
    const std::vector<frontend::PoseState>& shape_path,
    double request_time) {
  std::lock_guard<std::mutex> transaction_lock(global_once_plan_mutex_);
  const auto planning_started = std::chrono::steady_clock::now();
  global_once_hard_deadline_ = planning_started +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(par_.global_once_hard_timeout_seconds));
  global_once_refinement_deadline_ = planning_started +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(
              par_.global_once_refinement_timeout_seconds));
  global_once_deadline_active_ = true;
  const ScopedFlagReset deadline_guard(global_once_deadline_active_);
  resetData();
  active_shape_path_.clear();
  active_goal_yaw_exact_ = true;

  if (par_.planning_mode != "global_once" || !state_initialized_ ||
      !shape_planner_) {
    return {GlobalPlanStage::NOT_READY, {},
            "global_once planner or known map is not ready"};
  }
  if (shape_path.size() < 2) {
    return {GlobalPlanStage::SHAPE_ASTAR_FAILED, {},
            "certified Shape path has fewer than two waypoints"};
  }

  state captured_start;
  getState(captured_start);
  captured_start.vel.setZero();
  captured_start.accel.setZero();
  captured_start.jerk.setZero();
  captured_start.setTimeStamp(request_time);
  constexpr double kEndpointTolerance = 1.0e-6;
  if ((shape_path.front().position - captured_start.pos).norm() >
      kEndpointTolerance) {
    return {GlobalPlanStage::SHAPE_ASTAR_FAILED, {},
            "certified Shape path start does not match the captured state"};
  }
  for (std::size_t index = 0; index < shape_path.size(); ++index) {
    const auto& waypoint = shape_path[index];
    if (!waypoint.position.allFinite() || !std::isfinite(waypoint.yaw)) {
      return {GlobalPlanStage::SHAPE_ASTAR_FAILED, {},
              "certified Shape path contains a non-finite waypoint"};
    }
    if (index > 0 &&
        (waypoint.position - shape_path[index - 1].position).norm() <=
            kEndpointTolerance) {
      return {GlobalPlanStage::SHAPE_ASTAR_FAILED, {},
              "certified Shape path contains a zero-length edge"};
    }
  }

  state exact_goal;
  exact_goal.pos = shape_path.back().position;
  exact_goal.yaw = captured_start.yaw + std::remainder(
      shape_path.back().yaw - captured_start.yaw, 2.0 * M_PI);
  exact_goal.dyaw = 0.0;
  exact_goal.use_tracking_yaw = true;

  frontend::PlanResult shape_result;
  shape_result.status = frontend::PlanStatus::kSuccess;
  shape_result.message = "replayed collision-certified Shape path";
  shape_result.waypoints = shape_path;
  for (std::size_t index = 1; index < shape_path.size(); ++index) {
    shape_result.statistics.path_length +=
        (shape_path[index].position - shape_path[index - 1].position).norm();
    const double yaw_travel = std::abs(std::remainder(
        shape_path[index].yaw - shape_path[index - 1].yaw, 2.0 * M_PI));
    shape_result.statistics.yaw_travel += yaw_travel;
  }
  shape_result.statistics.raw_waypoints = shape_path.size();
  shape_result.statistics.final_cost = shape_result.statistics.path_length;

  return executeCertifiedShapePath(
      captured_start, exact_goal, std::move(shape_result), request_time);
}

GlobalPlanResult MIGHTY::executeCertifiedShapePath(
    state captured_start, state exact_goal,
    frontend::PlanResult shape_result, double request_time) {
  const auto backend_started = std::chrono::steady_clock::now();
  active_shape_path_ = shape_result.waypoints;

  const auto finish = [&](GlobalPlanResult result) {
    result.frontend_time_ms = global_planning_time_;
    result.middle_time_ms = middle_stage_time_ms_;
    result.seed_stage_time_ms = seed_stage_time_ms_;
    result.refinement_stage_time_ms = refinement_stage_time_ms_;
    result.route = trajectory_route_;
    result.derivative_activation_scales = derivative_activation_scales_;
    result.optimization_status = optimization_status_;
    result.refinement_timed_out = refinement_timed_out_;
    if (result.stage == GlobalPlanStage::ALREADY_REACHED) {
      result.outcome = GlobalPlanOutcome::NO_ACTION;
    } else if (result.stage == GlobalPlanStage::SUCCESS) {
      result.quality = evaluateTrajectoryQuality(
          goal_setpoints_, par_.global_once_minimum_interior_speed);
      result.outcome = result.quality.has_internal_stop
                           ? GlobalPlanOutcome::SAFE_DEGRADED
                           : GlobalPlanOutcome::NOMINAL;
    } else {
      result.outcome = GlobalPlanOutcome::FAILED;
    }
    result.back_end_preparation_time_ms = back_end_preparation_time_ms_;
    result.back_end_optimization_time_ms = back_end_optimization_time_ms_;
    result.back_end_retime_time_ms = back_end_retime_time_ms_;
    result.trajectory_export_time_ms = trajectory_export_time_ms_;
    result.plan_commit_time_ms = plan_commit_time_ms_;
    result.risk_segment_count = risk_segment_count_;
    result.risk_segment_edges = risk_segment_edges_;
    result.risk_window_count = risk_window_count_;
    result.trajectory_segment_count = trajectory_segment_count_;
    result.trajectory_yaw_variation = trajectory_yaw_variation_;
    result.backend_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - backend_started).count();
    const double attributed =
        result.back_end_preparation_time_ms + result.back_end_optimization_time_ms +
        result.back_end_retime_time_ms +
        result.trajectory_export_time_ms + result.plan_commit_time_ms;
    result.backend_unattributed_time_ms =
        std::max(0.0, result.backend_time_ms - attributed);
    global_once_deadline_active_ = false;
    return result;
  };

  setA(captured_start);
  setA_time(request_time);
  setGterm(exact_goal);
  setG(exact_goal);
  terminal_goal_initialized_ = true;
  global_once_committed_ = false;
  changeDroneStatus(DroneStatus::GOAL_REACHED);

  // Hold the captured start throughout the synchronous planning operation.
  {
    std::lock_guard<std::mutex> lock(mtx_plan_);
    plan_.clear();
    plan_.push_back(captured_start);
  }

  if ((captured_start.pos - exact_goal.pos).norm() <= par_.goal_radius) {
    const double yaw_error = std::abs(
        std::remainder(exact_goal.yaw - captured_start.yaw, 2.0 * M_PI));
    if (yaw_error <= par_.yaw_goal_tolerance) {
      return finish(
          {GlobalPlanStage::ALREADY_REACHED, std::move(shape_result), {}});
    }
    if (!buildYawOnlyTrajectory(captured_start, exact_goal)) {
      return finish({GlobalPlanStage::OPTIMIZATION_FAILED,
                     std::move(shape_result),
                     "failed to build yaw-only trajectory"});
    }
    middle_goal_setpoints_ = goal_setpoints_;
    middle_stage_time_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - backend_started).count();
    const auto commit_started = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mtx_plan_);
      plan_.assign(goal_setpoints_.begin(), goal_setpoints_.end());
    }
    plan_commit_time_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - commit_started).count();
    global_once_committed_ = true;
    replanning_failure_count_ = 0;
    changeDroneStatus(DroneStatus::TRAVELING);
    return finish({GlobalPlanStage::SUCCESS, std::move(shape_result), {}});
  }

  vec_Vecf<3> global_path;
  const auto certified_shape_path = active_shape_path_;
  if (!generateGlobalPath(global_path, request_time, 0.0)) {
    return finish({GlobalPlanStage::SHAPE_ASTAR_FAILED,
                   std::move(shape_result),
                   "Shape A* returned fewer than two waypoints"});
  }

  GlobalPlanStage failure_stage = GlobalPlanStage::OPTIMIZATION_FAILED;
  std::string failure_message;
  bool trajectory_ready =
      planTrajectory(global_path, &failure_stage, &failure_message);
  if (!trajectory_ready &&
      failure_stage == GlobalPlanStage::OPTIMIZATION_FAILED &&
      active_shape_path_.size() > certified_shape_path.size() &&
      (!global_once_deadline_active_ ||
       std::chrono::steady_clock::now() < global_once_hard_deadline_)) {
    active_shape_path_ = certified_shape_path;
    if (generateGlobalPath(global_path, request_time, 0.0, false)) {
      trajectory_ready =
          planTrajectory(global_path, &failure_stage, &failure_message);
    }
  }
  if (!trajectory_ready) {
    return finish({failure_stage, std::move(shape_result),
                   std::move(failure_message)});
  }

  if (goal_setpoints_.empty()) {
    return finish({GlobalPlanStage::OPTIMIZATION_FAILED,
                   std::move(shape_result),
                   "optimizer produced no executable setpoints"});
  }

  // Commit atomically: execution never observes a partially assembled plan.
  const auto commit_started = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lock(mtx_plan_);
    plan_.assign(goal_setpoints_.begin(), goal_setpoints_.end());
  }
  plan_commit_time_ms_ = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - commit_started).count();
  global_once_committed_ = true;
  replanning_failure_count_ = 0;
  changeDroneStatus(DroneStatus::TRAVELING);
  return finish({GlobalPlanStage::SUCCESS, std::move(shape_result), {}});
}

void MIGHTY::cancelPendingGlobalPlan() {
  global_once_generation_.fetch_add(1);
  state hold;
  getState(hold);
  hold.vel.setZero();
  hold.accel.setZero();
  hold.jerk.setZero();
  hold.dyaw = 0.0;
  {
    std::lock_guard<std::mutex> lock(mtx_plan_);
    executable_traj_solver_.reset();
    trajectory_execution_epoch_ = std::numeric_limits<double>::quiet_NaN();
    plan_.clear();
    plan_.push_back(hold);
  }
  global_once_committed_ = false;
}

// ----------------------------------------------------------------------------

bool MIGHTY::generateGlobalPath(vec_Vecf<3>& global_path, double current_time,
                                double last_replaning_computation_time,
                                bool allow_subdivision) {
  (void)current_time;
  (void)last_replaning_computation_time;
  if (active_shape_path_.size() < 2) return false;
  std::vector<mighty::frontend::PoseState> pose_knots;
  pose_knots.push_back(active_shape_path_.front());
  auto subdivision_context = whole_body_safety_
      ? std::optional<whole_body_safety::SafetyQueryContext>(
            whole_body_safety_->createContext())
      : std::nullopt;
  const auto edge_meets_backend_clearance = [&](const auto& start,
                                                 const auto& finish) {
    if (!whole_body_safety_ || !subdivision_context) return true;
    const double distance = (finish.position - start.position).norm();
    const double yaw_delta = std::remainder(finish.yaw - start.yaw,
                                            2.0 * M_PI);
    const int intervals = whole_body_safety::SafetySamplingPolicy{
        par_.safety_position_sample_step, par_.safety_yaw_sample_step, 8, 256}
                              .intervals(distance, std::abs(yaw_delta));
    for (int sample = 0; sample <= intervals; ++sample) {
      const double fraction = static_cast<double>(sample) / intervals;
      const auto safety = whole_body_safety_->evaluate(
          start.position + fraction * (finish.position - start.position),
          start.yaw + fraction * yaw_delta, *subdivision_context, true);
      if (!safety.success ||
          safety.statistics.minimum_beta < par_.safety_acceptance_beta) {
        return false;
      }
    }
    return true;
  };
  bool path_meets_backend_clearance = true;
  for (std::size_t edge = 0; edge + 1 < active_shape_path_.size(); ++edge) {
    if (!edge_meets_backend_clearance(active_shape_path_[edge],
                                      active_shape_path_[edge + 1])) {
      path_meets_backend_clearance = false;
      break;
    }
  }
  for (std::size_t edge = 0; edge + 1 < active_shape_path_.size(); ++edge) {
    const auto& start = active_shape_path_[edge];
    const auto& finish = active_shape_path_[edge + 1];
    const double distance = (finish.position - start.position).norm();
    const int subdivisions = allow_subdivision && max_dist_vertexes_ > 0.0 &&
                                     path_meets_backend_clearance
        ? std::max(1, static_cast<int>(std::ceil(distance / max_dist_vertexes_)))
        : 1;
    const double yaw_delta = std::remainder(finish.yaw - start.yaw,
                                            2.0 * M_PI);
    for (int part = 1; part <= subdivisions; ++part) {
      const double fraction = static_cast<double>(part) / subdivisions;
      mighty::frontend::PoseState knot;
      knot.position = start.position + fraction *
          (finish.position - start.position);
      knot.yaw = start.yaw + fraction * yaw_delta;
      pose_knots.push_back(knot);
    }
  }
  active_shape_path_ = std::move(pose_knots);
  global_path.clear();
  global_path.reserve(active_shape_path_.size());
  for (const auto& pose : active_shape_path_) {
    global_path.push_back(pose.position);
  }
  {
    std::lock_guard<std::mutex> lock(mtx_global_path_);
    global_path_ = global_path;
  }
  final_g_ = 0.0;
  for (std::size_t index = 1; index < active_shape_path_.size(); ++index) {
    final_g_ += (active_shape_path_[index].position -
                 active_shape_path_[index - 1].position).norm();
  }
  return true;
}

// ----------------------------------------------------------------------------

bool MIGHTY::planTrajectory(vec_Vecf<3>& global_path,
                                 GlobalPlanStage* failure_stage,
                                 std::string* failure_message) {
  // Accumulate retries and early failures, while excluding trajectory export.
  struct StageTimer {
    double* total;
    const std::function<void(const char*)>& observer;
    std::chrono::steady_clock::time_point started{std::chrono::steady_clock::now()};
    void stop() {
      if (!total) return;
      *total = std::max(0.0, *total) + std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count();
      total = nullptr;
      if (observer) observer("idle");
    }
    void next(double& value) { stop(); total = &value; started = std::chrono::steady_clock::now(); }
    ~StageTimer() { stop(); }
  } stage_timer{&seed_stage_time_ms_, planning_stage_observer_};
  if (planning_stage_observer_) planning_stage_observer_("middle");
  const auto fail = [&](GlobalPlanStage stage, std::string message) {
    if (failure_stage) *failure_stage = stage;
    if (failure_message) *failure_message = std::move(message);
    return false;
  };
  const auto middle_stage_started = std::chrono::steady_clock::now();
  const auto record_middle_stage_time = [&]() {
    middle_stage_time_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - middle_stage_started).count();
  };
  // Get local_A, local_G and A_time
  state local_A, local_G;
  double A_time;
  getA(local_A);
  getG(local_G);
  getA_time(A_time);

  // If the global path's size is < 2 after trimming, we cannot proceed
  if (global_path.empty() || global_path.size() < 2) {
    // std::cout << "Global path's size is < 2 after trimming" << std::endl;
    replanning_failure_count_++;
    return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                "global path has fewer than two waypoints");
  }

  // Static obstacle avoidance is exclusively provided by whole-body beta*.
  if (!whole_body_safety_) {
    return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                "whole-body safety scene is unavailable");
  }
  whole_body_safety_context_.reset();

  const auto export_trajectory = [&](
      const std::shared_ptr<lbfgs::SolverLBFGS>& solver) {
    const auto started = std::chrono::steady_clock::now();
    solver->getGoalSetpoints(goal_setpoints_, true);
    solver->getPieceWisePol(pwp_to_share_);
    solver->getControlPoints(cps_);
    trajectory_export_time_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
  };
  const auto classify_risk_segments = [&]
      (const std::shared_ptr<lbfgs::SolverLBFGS>& solver) {
    std::vector<bool> high_risk;
    if (!solver || !whole_body_safety_) return high_risk;
    const auto& times = solver->getSegmentTimes();
    high_risk.assign(times.size(), false);
    auto context = whole_body_safety_->createContext();
    const whole_body_safety::SafetySamplingPolicy sampling_policy{
        par_.safety_position_sample_step, par_.safety_yaw_sample_step, 8, 256};
    for (std::size_t segment = 0; segment < times.size(); ++segment) {
      double length = 0.0;
      double yaw_travel = 0.0;
      Eigen::Vector3d previous_position = solver->evalStateDeriv(
          static_cast<int>(segment), 0.0).pos;
      double previous_yaw = solver->evalYawState(
          static_cast<int>(segment), 0.0).yaw;
      for (int probe = 1; probe <= 8; ++probe) {
        const double tau = static_cast<double>(probe) / 8.0;
        const Eigen::Vector3d position = solver->evalStateDeriv(
            static_cast<int>(segment), tau).pos;
        const double yaw = solver->evalYawState(
            static_cast<int>(segment), tau).yaw;
        length += (position - previous_position).norm();
        yaw_travel += std::abs(yaw - previous_yaw);
        previous_position = position;
        previous_yaw = yaw;
      }
      const int samples = sampling_policy.intervals(length, yaw_travel);
      for (int sample = 0; sample <= samples; ++sample) {
        const double tau = static_cast<double>(sample) / samples;
        const auto safety = whole_body_safety_->evaluate(
            solver->evalStateDeriv(static_cast<int>(segment), tau).pos,
            solver->evalYawState(static_cast<int>(segment), tau).yaw,
            context, false);
        if (!safety.success ||
            safety.statistics.minimum_beta < whole_body_safety_->betaSoft()) {
          high_risk[segment] = true;
          break;
        }
      }
    }
    if (whole_body_safety_context_) {
      whole_body_safety_context_->cumulative_statistics.merge(
          context.cumulative_statistics);
    }
    std::vector<bool> expanded = high_risk;
    for (std::size_t segment = 0; segment < high_risk.size(); ++segment) {
      if (!high_risk[segment]) continue;
      if (segment > 0) expanded[segment - 1] = true;
      if (segment + 1 < expanded.size()) expanded[segment + 1] = true;
    }
    return expanded;
  };

  whole_body_safety_context_ =
      std::make_shared<whole_body_safety::SafetyQueryContext>(
          whole_body_safety_->createContext());
  whole_traj_solver_ptr_->setWholeBodySafety(
      whole_body_safety_, whole_body_safety_context_,
      {par_.safety_position_sample_step, par_.safety_yaw_sample_step, 8, 256});

  double unused_optimization_ms = 0.0;
  if (!prepareOrOptimizeTrajectory(
          local_A, A_time, global_path, back_end_preparation_time_ms_,
          unused_optimization_ms, whole_traj_solver_ptr_, lbfgs_params_, false,
          nullptr)) {
    replanning_failure_count_++;
    return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                "warm-start preparation failed");
  }
  if (limitedTo("middle")) {
    record_middle_stage_time();
    state seed;
    getA(seed);
    goal_setpoints_ = {seed};
    middle_goal_setpoints_ = goal_setpoints_;
    return true;
  }
  stage_timer.next(refinement_stage_time_ms_);
  if (planning_stage_observer_) planning_stage_observer_("backend");
  initial_guess_computation_time_ = back_end_preparation_time_ms_;
  derivative_activation_scales_ =
      whole_traj_solver_ptr_->getDerivativeActivationScales();

  const auto risk_segments = classify_risk_segments(whole_traj_solver_ptr_);
  trajectory_segment_count_ = risk_segments.size();
  risk_segment_edges_ = static_cast<std::size_t>(std::count(
      risk_segments.begin(), risk_segments.end(), true));
  for (std::size_t segment = 0; segment < risk_segments.size(); ++segment) {
    if (risk_segments[segment] &&
        (segment == 0 || !risk_segments[segment - 1])) {
      ++risk_segment_count_;
    }
  }
  const auto risk_windows = makeOverlappingRiskWindows(risk_segments);
  risk_window_count_ = risk_windows.size();
  if (!whole_traj_solver_ptr_->setRiskLocalRefinementSegments(
          par_.risk_local_refinement ? risk_segments
                                     : std::vector<bool>{})) {
    return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                "failed to configure risk-local refinement segments");
  }

  whole_traj_solver_ptr_->setOptimizationDeadline(
      global_once_deadline_active_
          ? std::optional<std::chrono::steady_clock::time_point>(
                global_once_refinement_deadline_)
          : std::nullopt);
  Eigen::VectorXd optimized;
  double optimized_cost = std::numeric_limits<double>::infinity();
  lbfgs_params_.mem_size = 256;
  const auto optimization_started = std::chrono::steady_clock::now();
  Eigen::VectorXd optimization_seed = zopt_;

  // A long, monolithic collision-active block has poorly scaled curvature:
  // a line-search step that helps one narrow passage can be rejected by a
  // distant passage.  Use overlapping block-coordinate sweeps to construct a
  // better seed, then retain the mandatory joint solve below.  Every sweep
  // uses the same objective and hard deadline; only its active variables and
  // collision samples are local.
  if (par_.risk_local_refinement && par_.risk_window_refinement &&
      !risk_windows.empty()) {
    lbfgs::lbfgs_parameter_t window_params = lbfgs_params_;
    window_params.mem_size = 32;
    // Four local windows are sufficient for a twelve-edge block.  Once the
    // active run grows beyond that, the extra shared knot needs a deeper
    // local solve or the subsequent joint line search inherits its poor
    // curvature.  Scale work with active geometry, not wall-clock time.
    window_params.max_iterations = risk_segment_edges_ > 12 ? 16 : 8;
    window_params.delta = std::max(window_params.delta, 5.0e-3);
    for (const auto& window : risk_windows) {
      if (!whole_traj_solver_ptr_->setRiskLocalRefinementSegments(window)) {
        return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                    "failed to configure risk-window refinement segments");
      }
      Eigen::VectorXd window_optimized;
      double window_cost = std::numeric_limits<double>::infinity();
      const int window_status = whole_traj_solver_ptr_->optimize(
          optimization_seed, window_optimized, window_cost, window_params);
      if (!window_optimized.allFinite() ||
          lbfgs::isRejectedObjective(window_cost)) {
        optimization_status_ = window_status;
        return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                    "risk-window optimization produced no finite candidate");
      }
      whole_traj_solver_ptr_->reconstructPVATCPopt(window_optimized);
      optimization_seed = std::move(window_optimized);
    }
    if (!whole_traj_solver_ptr_->setRiskLocalRefinementSegments(
            risk_segments)) {
      return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                  "failed to restore joint risk refinement segments");
    }
  }
  bool refine_again = false;
  do {
    optimization_status_ = whole_traj_solver_ptr_->optimize(
        optimization_seed, optimized, optimized_cost, lbfgs_params_);
    if (!lbfgs::isSuccessfulTermination(optimization_status_) ||
        !optimized.allFinite() || lbfgs::isRejectedObjective(optimized_cost)) {
      break;
    }
    whole_traj_solver_ptr_->reconstructPVATCPopt(optimized);
    optimization_seed = optimized;
    refine_again = whole_traj_solver_ptr_->refineSafetySamplingForCurrentTrajectory();
    refine_again = whole_traj_solver_ptr_->refineYawVariationPenaltyForCurrentTrajectory() ||
                   refine_again;
  } while (refine_again);
  refinement_timed_out_ = whole_traj_solver_ptr_->optimizationTimedOut();
  back_end_optimization_time_ms_ =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - optimization_started).count();
  if (!lbfgs::isSuccessfulTermination(optimization_status_) ||
      !optimized.allFinite() || lbfgs::isRejectedObjective(optimized_cost)) {
    replanning_failure_count_++;
    return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                "optimizer did not converge to a finite candidate");
  }
  whole_traj_solver_ptr_->reconstructPVATCPopt(optimized);
  const auto retime_started = std::chrono::steady_clock::now();
  whole_traj_solver_ptr_->retimeForDynamics();
  back_end_retime_time_ms_ = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - retime_started).count();
  if (!whole_traj_solver_ptr_->getCurrentDecisionVariables(zopt_)) {
    replanning_failure_count_++;
    return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                "failed to snapshot retimed trajectory");
  }

  if (!whole_traj_solver_ptr_->yawDynamicsFeasible()) {
    return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                "refined trajectory violates yaw dynamics");
  }
  if (!whole_traj_solver_ptr_->uavDynamicsFeasible()) {
    return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                "refined trajectory violates UAV dynamics");
  }

  const auto& segment_times = whole_traj_solver_ptr_->getSegmentTimes();
  if (segment_times.empty()) {
    return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                "refined trajectory has no segments");
  }
  auto validation_context = whole_body_safety_->createContext();
  const whole_body_safety::SafetySamplingPolicy validation_sampling{
      par_.safety_position_sample_step, par_.safety_yaw_sample_step, 8, 256};
  for (std::size_t segment = 0; segment < segment_times.size(); ++segment) {
    double path_length = 0.0;
    double yaw_travel = 0.0;
    Eigen::Vector3d previous_position = whole_traj_solver_ptr_->evalStateDeriv(
        static_cast<int>(segment), 0.0).pos;
    double previous_yaw = whole_traj_solver_ptr_->evalYawState(
        static_cast<int>(segment), 0.0).yaw;
    for (int probe = 1; probe <= 8; ++probe) {
      const double tau = static_cast<double>(probe) / 8.0;
      const Eigen::Vector3d position = whole_traj_solver_ptr_->evalStateDeriv(
          static_cast<int>(segment), tau).pos;
      const double yaw = whole_traj_solver_ptr_->evalYawState(
          static_cast<int>(segment), tau).yaw;
      path_length += (position - previous_position).norm();
      yaw_travel += std::abs(yaw - previous_yaw);
      previous_position = position;
      previous_yaw = yaw;
    }
    const int intervals = validation_sampling.intervals(path_length, yaw_travel);
    for (int sample = 0; sample <= intervals; ++sample) {
      const double tau = static_cast<double>(sample) / intervals;
      const auto safety = whole_body_safety_->evaluate(
          whole_traj_solver_ptr_->evalStateDeriv(
              static_cast<int>(segment), tau).pos,
          whole_traj_solver_ptr_->evalYawState(
              static_cast<int>(segment), tau).yaw,
          validation_context, true);
      if (!safety.success ||
          safety.statistics.minimum_beta < par_.safety_acceptance_beta) {
        return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                    "refined trajectory failed exact whole-body validation "
                    "at segment " + std::to_string(segment) + ", sample " +
                    std::to_string(sample) + ", beta=" +
                    std::to_string(safety.statistics.minimum_beta));
      }
    }
  }
  if (whole_body_safety_context_) {
    whole_body_safety_context_->cumulative_statistics.merge(
        validation_context.cumulative_statistics);
  }

  const auto reference_yaws = shapeYawInitialGuess(segment_times.size() + 1);
  if (reference_yaws.size() != segment_times.size() + 1) {
    return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                "refined trajectory has no yaw reference");
  }
  double reference_yaw_variation = 0.0;
  for (std::size_t index = 1; index < reference_yaws.size(); ++index) {
    reference_yaw_variation +=
        std::abs(reference_yaws[index] - reference_yaws[index - 1]);
  }
  double trajectory_yaw_variation = 0.0;
  bool have_previous_yaw = false;
  double previous_raw_yaw = 0.0;
  double previous_unwrapped_yaw = 0.0;
  for (std::size_t segment = 0; segment < segment_times.size(); ++segment) {
    for (int sample = segment == 0 ? 0 : 1;
         sample <= kYawValidationIntervals; ++sample) {
      const double tau =
          static_cast<double>(sample) / kYawValidationIntervals;
      const double raw_yaw = whole_traj_solver_ptr_->evalYawState(
          static_cast<int>(segment), tau).yaw;
      if (!std::isfinite(raw_yaw)) {
        return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                    "refined trajectory has non-finite yaw");
      }
      if (have_previous_yaw) {
        const double unwrapped_yaw = previous_unwrapped_yaw +
            std::remainder(raw_yaw - previous_raw_yaw, 2.0 * M_PI);
        trajectory_yaw_variation +=
            std::abs(unwrapped_yaw - previous_unwrapped_yaw);
        previous_unwrapped_yaw = unwrapped_yaw;
      } else {
        previous_unwrapped_yaw = raw_yaw;
        have_previous_yaw = true;
      }
      previous_raw_yaw = raw_yaw;
    }
  }
  trajectory_yaw_variation_ = trajectory_yaw_variation;
  if (trajectory_yaw_variation >
      reference_yaw_variation + kAdditionalYawVariationBudget + 1.0e-9) {
    double endpoint_yaw_variation = 0.0;
    double endpoint_yaw = whole_traj_solver_ptr_->evalYawState(0, 0.0).yaw;
    for (std::size_t segment = 0; segment < segment_times.size(); ++segment) {
      const double next_yaw = whole_traj_solver_ptr_->evalYawState(
          static_cast<int>(segment), 1.0).yaw;
      endpoint_yaw_variation +=
          std::abs(std::remainder(next_yaw - endpoint_yaw, 2.0 * M_PI));
      endpoint_yaw = next_yaw;
    }
    return fail(GlobalPlanStage::OPTIMIZATION_FAILED,
                "refined trajectory violates yaw variation budget: actual=" +
                std::to_string(trajectory_yaw_variation) + ", limit=" +
                std::to_string(reference_yaw_variation +
                               kAdditionalYawVariationBudget) + ", endpoints=" +
                std::to_string(endpoint_yaw_variation));
  }

  fopt_ = optimized_cost;
  trajectory_route_ = TrajectoryRoute::WHOLE_BODY_REFINED;
  stage_timer.stop();
  whole_traj_solver_ptr_->getGoalSetpoints(middle_goal_setpoints_, true);
  record_middle_stage_time();
  trajectory_optimization_time_ = back_end_optimization_time_ms_;
  export_trajectory(whole_traj_solver_ptr_);
  {
    std::lock_guard<std::mutex> lock(mtx_plan_);
    executable_traj_solver_ = whole_traj_solver_ptr_;
    trajectory_execution_epoch_ = std::numeric_limits<double>::quiet_NaN();
  }

  return true;
}

// ----------------------------------------------------------------------------

std::vector<double> MIGHTY::shapeYawInitialGuess(
    std::size_t knot_count) const {
  if (active_shape_path_.empty() || knot_count < 2) return {};
  std::vector<double> cumulative(active_shape_path_.size(), 0.0);
  std::vector<double> unwrapped(active_shape_path_.size(), 0.0);
  unwrapped.front() = active_shape_path_.front().yaw;
  for (std::size_t index = 1; index < active_shape_path_.size(); ++index) {
    cumulative[index] = cumulative[index - 1] +
        (active_shape_path_[index].position -
         active_shape_path_[index - 1].position).norm();
    unwrapped[index] = unwrapped[index - 1] + std::remainder(
        active_shape_path_[index].yaw - active_shape_path_[index - 1].yaw,
        2.0 * M_PI);
  }
  if (knot_count == active_shape_path_.size()) return unwrapped;
  const double total = cumulative.back();
  std::vector<double> result(knot_count, unwrapped.front());
  if (total <= 1e-12) {
    for (std::size_t knot = 0; knot < knot_count; ++knot) {
      const double fraction = static_cast<double>(knot) /
                              static_cast<double>(knot_count - 1);
      result[knot] = unwrapped.front() +
                     fraction * (unwrapped.back() - unwrapped.front());
    }
    return result;
  }
  std::size_t edge = 0;
  for (std::size_t knot = 0; knot < knot_count; ++knot) {
    const double target = total * static_cast<double>(knot) /
                          static_cast<double>(knot_count - 1);
    while (edge + 1 < cumulative.size() &&
           cumulative[edge + 1] < target) {
      ++edge;
    }
    if (edge + 1 >= cumulative.size()) {
      result[knot] = unwrapped.back();
      continue;
    }
    const double length = cumulative[edge + 1] - cumulative[edge];
    const double fraction = length <= 1e-12
                                ? 0.0
                                : (target - cumulative[edge]) / length;
    result[knot] = unwrapped[edge] +
                   fraction * (unwrapped[edge + 1] - unwrapped[edge]);
  }
  return result;
}

bool MIGHTY::buildYawOnlyTrajectory(const state& start, const state& goal) {
  const double delta = std::remainder(goal.yaw - start.yaw, 2.0 * M_PI);
  constexpr double kPeakRate = 1.875;
  constexpr double kPeakAccel = 5.773502691896258;
  const double total_time = 1.01 * std::max(
      {2.0 * par_.dc, kPeakRate * std::abs(delta) / par_.w_max,
       std::sqrt(kPeakAccel * std::abs(delta) / par_.yaw_acc_max)});
  const int sample_count =
      std::max(2, static_cast<int>(std::ceil(total_time / par_.dc)) + 1);
  goal_setpoints_.resize(sample_count);
  for (int i = 0; i < sample_count; ++i) {
    const double relative_time =
        std::min(static_cast<double>(i) * par_.dc, total_time);
    const double tau = relative_time / total_time;
    const double tau2 = tau * tau;
    const double tau3 = tau2 * tau;
    const double tau4 = tau3 * tau;
    const double tau5 = tau4 * tau;
    const double phase = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5;
    const double phase_rate =
        (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / total_time;
    state setpoint = start;
    setpoint.t = start.t + relative_time;
    setpoint.vel.setZero();
    setpoint.accel.setZero();
    setpoint.jerk.setZero();
    setpoint.yaw = start.yaw + delta * phase;
    setpoint.dyaw = delta * phase_rate;
    setpoint.use_tracking_yaw = true;
    goal_setpoints_[i] = setpoint;
  }
  goal_setpoints_.front().dyaw = 0.0;
  goal_setpoints_.back().dyaw = 0.0;

  pwp_to_share_.clear();
  pwp_to_share_.times = {start.t, start.t + total_time};
  Eigen::Matrix<double, 6, 1> cx = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> cy = cx;
  Eigen::Matrix<double, 6, 1> cz = cx;
  cx[5] = start.pos.x();
  cy[5] = start.pos.y();
  cz[5] = start.pos.z();
  pwp_to_share_.coeff_x.push_back(cx);
  pwp_to_share_.coeff_y.push_back(cy);
  pwp_to_share_.coeff_z.push_back(cz);
  cps_.resize(1);
  cps_.front().colwise() = start.pos;
  return true;
}

// ----------------------------------------------------------------------------

void MIGHTY::getPiecewiseQuinticPol(PieceWiseQuinticPol& pwp) { pwp = pwp_to_share_; }

// ----------------------------------------------------------------------------

bool MIGHTY::prepareOrOptimizeTrajectory(
                                     const state& local_A, double A_time, vec_Vec3f& global_path,
                                     double& initial_guess_computation_time,
                                     double& trajectory_optimization_time,
                                     std::shared_ptr<lbfgs::SolverLBFGS>& whole_traj_solver_ptr,
                                     const lbfgs::lbfgs_parameter_t& optimization_parameters,
                                     bool run_optimization,
                                     const Eigen::VectorXd* warm_start) {
  const auto preparation_started = std::chrono::steady_clock::now();
  if (par_.debug_verbose) std::cout << "Preparing solver for replan" << std::endl;

  // Get local_G
  state local_G;
  getG(local_G);

  state local_E;
  Vec3f mean_point;

  if (drone_status_ == DroneStatus::GOAL_REACHED || drone_status_ == DroneStatus::GOAL_SEEN) {
    local_E = local_G;
  } else {
    local_E.pos = global_path.back();
  }

  double seed_assembly_time_ms = 0.0;
  whole_traj_solver_ptr->prepareSolverForReplan(
      A_time, global_path, local_A, local_E,
      shapeYawInitialGuess(global_path.size()), active_goal_yaw_exact_,
      seed_assembly_time_ms);

  // Skip the solver's pushed path â€? use the original frontend path instead
  // whole_traj_solver_ptr->getGlobalPath(global_path);

  {
    std::lock_guard<std::mutex> lock(mtx_global_path_);
    global_path_ = global_path;  // Update the global path
  }

  // update local_E
  local_E.pos = global_path.back();
  {
    std::lock_guard<std::mutex> lock(mtx_E_);
    E_ = local_E;  // Update the local_E
  }

  if (par_.debug_verbose) std::cout << "Solver prepared" << std::endl;

  // Get initial guesses
  auto list_z0 = whole_traj_solver_ptr->getInitialGuesses();
  if (warm_start && !list_z0.empty() &&
      warm_start->size() == list_z0.front().size() && warm_start->allFinite()) {
    list_z0.front() = *warm_start;
  }
  auto list_initial_guess_wps = whole_traj_solver_ptr->getInitialGuessWaypoints();
  initial_guess_computation_time =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - preparation_started).count();

  if (list_z0.empty() || list_initial_guess_wps.empty()) return false;
  if (!run_optimization) {
    zopt_ = list_z0.front();
    fopt_ = 0.0;
    trajectory_optimization_time = 0.0;
    return zopt_.allFinite();
  }

  if (par_.debug_verbose) std::cout << "Initial guesses size: " << list_z0.size() << std::endl;

  // Update L-BFGS parameters.
  // lbfgs_params_.mem_size = static_cast<int>(list_z0[0].size());
  lbfgs_params_.mem_size = 256;

  const auto t_start = std::chrono::high_resolution_clock::now();
  const int status = whole_traj_solver_ptr->optimize(
      list_z0.front(), zopt_, fopt_, optimization_parameters);
  const auto t_end = std::chrono::high_resolution_clock::now();
  trajectory_optimization_time =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

  if (par_.debug_verbose)
    std::cout << "Optimization status: " << status << ", fopt: " << fopt_
              << ", computation time: " << trajectory_optimization_time << " ms" << std::endl;

  // If no solution is found, return.
  if (status < 0 || lbfgs::isRejectedObjective(fopt_)) {
    // do the same output in red with printf
    printf("\033[1;31mLocal Optimization Failed with status: %d, fopt: %.2f\033[0m\n", status,
           fopt_);
    return false;
  }

  // If the optimization succeeded, we can update the goal setpoints in green
  // printf("\033[1;32mLocal Optimization Succeeded with status: %d, fopt: %.2f\033[0m\n", status,
  // fopt_);
  return true;
}

// ----------------------------------------------------------------------------

/**
 * @brief Gets the terminal goal state.
 * @param state &G_term: Output terminal goal state.
 */
void MIGHTY::getGterm(state& G_term) {
  std::lock_guard<std::mutex> lock(mtx_G_term_);
  G_term = G_term_;
}

// ----------------------------------------------------------------------------

/**
 * @brief Sets the terminal goal state.
 * @param state G_term: Terminal goal state to set.
 */
void MIGHTY::setGterm(const state& G_term) {
  std::lock_guard<std::mutex> lock(mtx_G_term_);
  G_term_ = G_term;
}

// ----------------------------------------------------------------------------

/**
 * @brief Gets the subgoal.
 * @param state &G: Output subgoal.
 */
void MIGHTY::getG(state& G) {
  std::lock_guard<std::mutex> lock(mtx_G_);
  G = G_;
}

// ----------------------------------------------------------------------------

/**
 * @brief Gets point E
 * @param state &G: Output point E
 */
void MIGHTY::getE(state& E) {
  std::lock_guard<std::mutex> lock(mtx_E_);
  E = E_;
}

// ----------------------------------------------------------------------------

/**
 * @brief Sets the subgoal.
 * @param state G: Subgoal to set.
 */
void MIGHTY::setG(const state& G) {
  std::lock_guard<std::mutex> lock(mtx_G_);
  G_ = G;
}

// ----------------------------------------------------------------------------

/**
 * @brief Gets A (starting point for global planning).
 * @param state &G: Output A.
 */
void MIGHTY::getA(state& A) {
  std::lock_guard<std::mutex> lock(mtx_A_);
  A = A_;
}

// ----------------------------------------------------------------------------

/**
 * @brief Sets A (starting point for global planning).
 * @param state &G: Input A.
 */
void MIGHTY::setA(const state& A) {
  std::lock_guard<std::mutex> lock(mtx_A_);
  A_ = A;
}

// ----------------------------------------------------------------------------

void MIGHTY::getA_time(double& A_time) {
  std::lock_guard<std::mutex> lock(mtx_A_time_);
  A_time = A_time_;
}

// ----------------------------------------------------------------------------

/**
 * @brief Sets A (starting point for global planning)'s time
 * @param state &G: Input A time
 */
void MIGHTY::setA_time(double A_time) {
  std::lock_guard<std::mutex> lock(mtx_A_time_);
  A_time_ = A_time;
}

// ----------------------------------------------------------------------------

/**
 * @brief Gets the current state.
 * @param state &state: Output current state.
 */
void MIGHTY::getState(state& state) {
  std::lock_guard<std::mutex> lock(mtx_state_);
  state = state_;
}

// ----------------------------------------------------------------------------

/**
 * @brief Gets the last plan state
 * @param state &state: Output last plan state
 */
void MIGHTY::getLastPlanState(state& state) {
  std::lock_guard<std::mutex> lock(mtx_plan_);
  state = plan_.back();
}

// ----------------------------------------------------------------------------

void MIGHTY::updateState(state data) {
  {
    std::lock_guard<std::mutex> lock(mtx_state_);
    state_ = data;
  }

  if (state_initialized_ == false || drone_status_ == DroneStatus::YAWING) {
    // create temporary state
    state tmp;
    tmp.pos = data.pos;
    tmp.yaw = data.yaw;
    previous_yaw_ = data.yaw;

    // Push the state to the plan
    {
      std::lock_guard<std::mutex> lock(mtx_plan_);
      plan_.clear();
      plan_.push_back(tmp);
    }

    // Update Point A
    setA(tmp);

    // Update Point G
    setG(tmp);

    // Update the flag
    state_initialized_ = true;
  }
}

// ----------------------------------------------------------------------------

/**
 * @brief Retrieves the next goal (setpoint) from the plan.
 * @param state &next_goal: Output next goal state.
 * @return bool
 */
bool MIGHTY::getNextGoal(double now, state& next_goal) {
  // Check if the planner is initialized
  if (!checkReadyToReplan()) {
    return false;
  }

  next_goal.setZero();

  std::deque<state> local_plan;
  std::shared_ptr<lbfgs::SolverLBFGS> executable_solver;
  double elapsed_time = 0.0;
  bool analytic_execution = false;
  {
    std::lock_guard<std::mutex> lock(mtx_plan_);
    analytic_execution =
        par_.planning_mode == "global_once" && executable_traj_solver_;
    if (analytic_execution) {
      if (!std::isfinite(trajectory_execution_epoch_)) {
        trajectory_execution_epoch_ = now;
      }
      elapsed_time = std::max(0.0, now - trajectory_execution_epoch_);
      executable_solver = executable_traj_solver_;
      local_plan = plan_;
    }
  }
  if (analytic_execution) {
    if (!executable_solver->evaluateAtElapsedTime(elapsed_time, next_goal)) {
      return false;
    }
  } else {
    std::lock_guard<std::mutex> lock(mtx_plan_);
    local_plan = plan_;
    if (local_plan.empty()) return false;
    if (plan_.size() > 1) {
      plan_.pop_front();
    }
    next_goal = local_plan.front();
  }

  const bool use_planned_yaw = par_.planning_mode == "global_once";
  if (use_planned_yaw) {
    previous_yaw_ = next_goal.yaw;
  } else if (!(drone_status_ == DroneStatus::GOAL_REACHED)) {
    // Get the desired yaw
    // If the planner keeps failing, just keep spinning
    if (replanning_failure_count_ > par_.yaw_spinning_threshold) {
      next_goal.yaw = previous_yaw_ + par_.yaw_spinning_dyaw * par_.dc;
      next_goal.dyaw = par_.yaw_spinning_dyaw;
      previous_yaw_ = next_goal.yaw;
    } else {
      // If the local_plan is small just use the previous yaw with no dyaw
      if (local_plan.size() < 5) {
        next_goal.yaw = previous_yaw_;
        next_goal.dyaw = 0.0;
      } else {
        getDesiredYaw(next_goal);
      }
    }

    next_goal.dyaw = std::clamp(next_goal.dyaw, -par_.w_max, par_.w_max);
  } else {
    next_goal.yaw = previous_yaw_;
    next_goal.dyaw = 0.0;
  }

  return true;
}

// ----------------------------------------------------------------------------

/**
 * @brief Computes the desired yaw for the next goal.
 * @param state &next_goal: Next goal state to update with desired yaw.
 */
void MIGHTY::getDesiredYaw(state& next_goal) {
  double diff = 0.0;
  double desired_yaw = 0.0;

  // Get state
  state local_state;
  getState(local_state);

  // Get G_term
  state G_term;
  {
    std::lock_guard<std::mutex> lock(mtx_G_term_);
    G_term = G_term_;
  }

  switch (drone_status_) {
    case DroneStatus::YAWING:
      desired_yaw = atan2(G_term.pos[1] - next_goal.pos[1], G_term.pos[0] - next_goal.pos[0]);
      diff = desired_yaw - local_state.yaw;
      // std::cout << "diff1= " << diff << std::endl;
      break;
    case DroneStatus::TRAVELING:
    case DroneStatus::GOAL_SEEN:
      desired_yaw =
          atan2(next_goal.pos[1] - local_state.pos.y(), next_goal.pos[0] - local_state.pos.x());
      diff = desired_yaw - local_state.yaw;
      next_goal.yaw = desired_yaw;
      break;
    case DroneStatus::GOAL_REACHED:
      next_goal.dyaw = 0.0;
      next_goal.yaw = previous_yaw_;
      return;
  }

  mighty_utils::angle_wrap(diff);
  if (fabs(diff) < 0.04 && drone_status_ == DroneStatus::YAWING) {
    changeDroneStatus(DroneStatus::TRAVELING);
  }

  yaw(diff, next_goal);
}

// ----------------------------------------------------------------------------

void MIGHTY::yaw(double diff, state& next_goal) {
  diff = std::clamp(diff, -par_.dc * par_.w_max, par_.dc * par_.w_max);
  dyaw_filtered_ = (1 - par_.alpha_filter_dyaw) * (copysign(1, diff) * par_.w_max) +
                   par_.alpha_filter_dyaw * dyaw_filtered_;
  next_goal.dyaw = dyaw_filtered_;
  next_goal.yaw = previous_yaw_ + dyaw_filtered_ * par_.dc;
  previous_yaw_ = next_goal.yaw;
}

// ----------------------------------------------------------------------------

/**
 * @brief Changes the drone's status (YAWING, TRAVELING, GOAL_SEEN, GOAL_REACHED).
 * @param int new_status: New status value.
 */
void MIGHTY::changeDroneStatus(int new_status) {
  const int old_status = drone_status_.load();
  if (new_status == old_status) return;

  std::cout << "Changing DroneStatus from ";

  switch (old_status) {
    case DroneStatus::YAWING:
      std::cout << "status_=YAWING";
      break;
    case DroneStatus::TRAVELING:
      std::cout << "status_=TRAVELING";
      break;
    case DroneStatus::GOAL_SEEN:
      std::cout << "status_=GOAL_SEEN";
      break;
    case DroneStatus::GOAL_REACHED:
      std::cout << "status_=GOAL_REACHED";
      break;
  }

  std::cout << " to ";

  switch (new_status) {
    case DroneStatus::YAWING:
      std::cout << "status_=YAWING";
      break;
    case DroneStatus::TRAVELING:
      std::cout << "status_=TRAVELING";
      break;
    case DroneStatus::GOAL_SEEN:
      std::cout << "status_=GOAL_SEEN";
      break;
    case DroneStatus::GOAL_REACHED:
      std::cout << "status_=GOAL_REACHED";
      break;
  }

  std::cout << std::endl;

  drone_status_.store(new_status);
}

// ----------------------------------------------------------------------------

int MIGHTY::getDroneStatus() const { return drone_status_.load(); }

// ----------------------------------------------------------------------------

/**
 * @brief Checks if all necessary components are initialized.
 * @return bool
 */
bool MIGHTY::checkReadyToReplan() {
  bool is_ready = state_initialized_ && terminal_goal_initialized_ && shape_planner_;

  return is_ready;
}

// ----------------------------------------------------------------------------

bool MIGHTY::goalReachedCheck(double now) {
  if (par_.planning_mode == "global_once") {
    if (!global_once_committed_) return false;

    state current, terminal;
    getState(current);
    getGterm(terminal);

    bool execution_finished = false;
    {
      std::lock_guard<std::mutex> lock(mtx_plan_);
      if (executable_traj_solver_) {
        if (std::isfinite(trajectory_execution_epoch_)) {
          const auto& segment_times = executable_traj_solver_->getSegmentTimes();
          const double trajectory_duration = std::accumulate(
              segment_times.begin(), segment_times.end(), 0.0);
          execution_finished =
              now - trajectory_execution_epoch_ >= trajectory_duration;
          // A free-yaw backend may choose a different certified terminal yaw
          // than the frontend seed. Check the executed endpoint, retaining the
          // same yaw tolerance used for explicitly constrained goals.
          if (!active_goal_yaw_exact_) {
            state endpoint;
            if (!executable_traj_solver_->evaluateAtElapsedTime(
                    trajectory_duration, endpoint)) return false;
            terminal.yaw = endpoint.yaw;
          }
        }
      } else {
        execution_finished = plan_.size() <= 1;
        if (!active_goal_yaw_exact_ && !plan_.empty()) {
          terminal.yaw = plan_.back().yaw;
        }
      }
    }
    const bool position_reached =
        (current.pos - terminal.pos).norm() <= par_.goal_radius;
    const bool yaw_reached =
        std::abs(std::remainder(current.yaw - terminal.yaw,
                                2.0 * M_PI)) <= par_.yaw_goal_tolerance;
    if (execution_finished && position_reached && yaw_reached) {
      global_once_committed_ = false;
      changeDroneStatus(DroneStatus::GOAL_REACHED);
      return true;
    }
    return false;
  }

  if (checkReadyToReplan() && drone_status_ == DroneStatus::GOAL_REACHED) {
    return true;
  }
  return false;
}
