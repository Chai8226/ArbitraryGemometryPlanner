/* ----------------------------------------------------------------------------
 * Copyright 2025, Kota Kondo, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Kota Kondo, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/StdVector>


#include <mighty/lbfgs_solver.hpp>
#include <mighty/utils.hpp>
#include <mighty/yaw_trajectory.hpp>
#include <mighty/shape_aware_astar/planner.hpp>

#include "mighty/mighty_type.hpp"

enum { MAP = 0, UNKNOWN_MAP = 1 };
enum { RETURN_LAST_VERTEX = 0, RETURN_INTERSECTION = 1 };

using Vec3 = Eigen::Vector3d;
using Vec3f = Eigen::Matrix<double, 3, 1>;
using MatXd = Eigen::MatrixXd;
using VecXd = Eigen::VectorXd;

// ------------------------------------------
// 2) The aligned‐allocator for local blocks:
// ------------------------------------------
// your PlaneBlock is still:
using PlaneBlock = std::pair<Eigen::Matrix<double, Eigen::Dynamic, 3>, Eigen::VectorXd>;

// replace your old ConstraintBlocks with:

// 1) an aligned‐allocator for a vector of PlaneBlock
using AlignedPlaneBlockVec = std::vector<PlaneBlock, Eigen::aligned_allocator<PlaneBlock>>;

// 2) then your top‐level blocks is a vector of those
using ConstraintBlocks =
    std::vector<AlignedPlaneBlockVec, Eigen::aligned_allocator<AlignedPlaneBlockVec>>;

enum DroneStatus { YAWING = 0, TRAVELING = 1, GOAL_SEEN = 2, GOAL_REACHED = 3 };

enum class GlobalPlanStage {
  SUCCESS,
  ALREADY_REACHED,
  NOT_READY,
  SHAPE_ASTAR_FAILED,
  OPTIMIZATION_FAILED
};

enum class TrajectoryRoute {
  WHOLE_BODY_REFINED,
  NONE,
};

enum class GlobalPlanOutcome {
  NOMINAL,
  SAFE_DEGRADED,
  FAILED,
  NO_ACTION,
};

struct TrajectoryQualityMetrics {
  bool available{false};
  bool has_internal_stop{false};
  double minimum_interior_speed{0.0};
  double maximum_subthreshold_duration{0.0};
  double speed_coefficient_of_variation{0.0};
  double duration_seconds{0.0};
  double path_length{0.0};
  double maximum_speed{0.0};
  double maximum_acceleration{0.0};
  double maximum_jerk{0.0};
};

struct GlobalPlanResult {
  GlobalPlanResult() = default;
  GlobalPlanResult(GlobalPlanStage plan_stage,
                   mighty::frontend::PlanResult plan_shape_path,
                   std::string plan_message)
      : stage(plan_stage),
        shape_path(std::move(plan_shape_path)),
        message(std::move(plan_message)) {}

  GlobalPlanStage stage{GlobalPlanStage::NOT_READY};
  GlobalPlanOutcome outcome{GlobalPlanOutcome::FAILED};
  mighty::frontend::PlanResult shape_path;
  std::string message;
  TrajectoryRoute route{TrajectoryRoute::NONE};
  TrajectoryQualityMetrics quality;
  std::vector<double> derivative_activation_scales;
  int optimization_status{0};
  bool refinement_timed_out{false};
  std::size_t risk_segment_count{0};
  std::size_t risk_segment_edges{0};
  std::size_t risk_window_count{0};
  std::size_t trajectory_segment_count{0};
  double frontend_time_ms{0.0};
  double middle_time_ms{0.0};
  double backend_time_ms{0.0};
  double back_end_preparation_time_ms{0.0};
  double back_end_optimization_time_ms{0.0};
  double back_end_retime_time_ms{0.0};
  double trajectory_export_time_ms{0.0};
  double plan_commit_time_ms{0.0};
  double backend_unattributed_time_ms{0.0};
  double trajectory_yaw_variation{-1.0};

  bool success() const {
    return stage == GlobalPlanStage::SUCCESS || stage == GlobalPlanStage::ALREADY_REACHED;
  }

  bool has_trajectory() const { return stage == GlobalPlanStage::SUCCESS; }
};

/** @brief Core trajectory planner using Hermite spline-based optimization.
 *
 *  Implements the MIGHTY planning algorithm including global path search,
 *  whole-body trajectory optimization via L-BFGS,
 *  and a state machine (YAWING -> TRAVELING -> GOAL_SEEN -> GOAL_REACHED).
 */
class MIGHTY {
 public:
  /** @brief Construct the planner with the given parameter set.
   *  @param par Configuration parameters for dynamics, map, and planner.
   */
  MIGHTY(parameters par, const mighty::frontend::FrontendConfig& frontend_config,
         const std::string& map_polymap_path,
         const std::string& body_polymap_path);

  /** Execute one whole-map frontend -> whole-body L-BFGS planning pipeline. */
  GlobalPlanResult planGlobalOnce(const state& terminal_goal, double request_time);
  /** Execute a previously collision-certified frontend path without replanning. */
  GlobalPlanResult planCertifiedShapePath(
      const std::vector<mighty::frontend::PoseState>& shape_path,
      double request_time);
  void cancelPendingGlobalPlan();

  const mighty::frontend::ModelInfo& shapeModelInfo() const;

  /** @brief Get the terminal goal state (thread-safe).
   *  @param G_term Output terminal goal.
   */
  void getGterm(state& G_term);

  /** @brief Set the terminal goal state (thread-safe).
   *  @param G_term New terminal goal.
   */
  void setGterm(const state& G_term);

  /** @brief Get the projected in-map goal G (thread-safe).
   *  @param G Output projected goal.
   */
  void getG(state& G);

  /** @brief Get the trajectory endpoint E (thread-safe).
   *  @param E Output endpoint state.
   */
  void getE(state& E);

  /** @brief Set the projected in-map goal G (thread-safe).
   *  @param G New projected goal.
   */
  void setG(const state& G);

  /** @brief Get the planning start state A (thread-safe).
   *  @param A Output start state.
   */
  void getA(state& A);

  /** @brief Set the planning start state A (thread-safe).
   *  @param A New start state.
   */
  void setA(const state& A);

  /** @brief Get the planning start time (thread-safe).
   *  @param A_time Output start time.
   */
  void getA_time(double& A_time);

  /** @brief Set the planning start time (thread-safe).
   *  @param A_time New start time.
   */
  void setA_time(double A_time);

  /** @brief Get the current robot state (thread-safe).
   *  @param state Output current state.
   */
  void getState(state& state);

  /** @brief Get the last state from the current plan.
   *  @param state Output last planned state.
   */
  void getLastPlanState(state& state);

  /** @brief Update the current robot state (thread-safe).
   *  @param data New state data.
   */
  void updateState(state data);

  /** @brief Sample the next goal setpoint from the current plan.
   *  @param next_goal Output goal state.
   *  @return True if a valid goal was obtained.
   */
  bool getNextGoal(double now, state& next_goal);

  /** @brief Check if the planner is ready to replan.
   *  @return True if state and terminal goal have been initialized.
   */
  bool checkReadyToReplan();

  /** @brief Change the drone state machine status.
   *  @param new_status One of DroneStatus enum values.
   */
  void changeDroneStatus(int new_status);

  /** @brief Return the current DroneStatus value for diagnostics. */
  int getDroneStatus() const;

  /** @brief Compute the desired yaw for the next goal based on velocity direction.
   *  @param next_goal Goal state whose yaw field is updated.
   */
  void getDesiredYaw(state& next_goal);

  /** @brief Command a yaw rotation by the given angular difference.
   *  @param diff Yaw angle difference in radians.
   *  @param next_goal Output goal state with updated yaw.
   */
  void yaw(double diff, state& next_goal);

  /** @brief Check if the robot has reached the terminal goal.
   *  @param now Current wall/simulation time in seconds.
   *  @return True if the goal is reached.
   */
  bool goalReachedCheck(double now);

  /** @brief Prepare a certified smooth warm start and optionally run L-BFGS.
   *  @param local_A Start state for the trajectory.
   *  @param A_time Start timestamp.
   *  @param global_path Global waypoints guiding the trajectory.
   *  @param preparation_time Output time for complete solver and seed preparation.
   *  @param trajectory_optimization_time Output time for trajectory optimization.
   *  @param whole_traj_solver_ptr L-BFGS solver instance.
   *  @return True if a feasible trajectory was generated.
   */
  bool prepareOrOptimizeTrajectory(
      const state& local_A, double A_time, vec_Vec3f& global_path,
      double& preparation_time, double& trajectory_optimization_time,
      std::shared_ptr<lbfgs::SolverLBFGS>& whole_traj_solver_ptr,
      const lbfgs::lbfgs_parameter_t& optimization_parameters,
      bool run_optimization, const Eigen::VectorXd* warm_start = nullptr);

  /** @brief Reset all internal timing and cost data to zero. */
  void resetData();

  /** @brief Retrieve computation time breakdowns from the last replan.
   *  @param final_g Final path cost.
   *  @param global_planning_time Total global planning time.
   *  @param initial_guess_computation_time Initial guess computation time.
   *  @param trajectory_optimization_time Whole-trajectory optimization time.
   *  @param safety_check_time Safety check time.
   *  @param safe_paths_time Safe paths computation time.
   */
  void retrieveData(double& final_g, double& global_planning_time,
                    double& initial_guess_computation_time,
                    double& trajectory_optimization_time);

  /** @brief Retrieve goal setpoints from the last trajectory optimization.
   *  @param goal_setpoints Output vector of goal states.
   */
  void retrieveGoalSetpoints(std::vector<state>& goal_setpoints);

  /** @brief Retrieve the certified warm-start trajectory before backend refinement.
   *  @param goal_setpoints Output vector of middle-stage goal states.
   */
  void retrieveMiddleGoalSetpoints(std::vector<state>& goal_setpoints);

  mighty::whole_body_safety::SafetyStatistics wholeBodySafetyStatistics() const {
    return whole_body_safety_context_
               ? whole_body_safety_context_->cumulative_statistics
               : mighty::whole_body_safety::SafetyStatistics{};
  }
  /** @brief Retrieve control points from the last trajectory optimization.
   *  @param cps Output vector of 3x6 control point matrices per segment.
   */
  void retrieveCPs(std::vector<Eigen::Matrix<double, 3, 6>>& cps);

  /** @brief Generate a global path from the current state to the goal.
   *  @param global_path Output global waypoint path.
   *  @param current_time Current wall/sim time.
   *  @param last_replaning_computation_time Previous replan computation time.
   *  @return True if a valid global path was found.
   */
  bool generateGlobalPath(vec_Vecf<3>& global_path, double current_time,
                          double last_replaning_computation_time,
                          bool allow_subdivision = true);

  /** @brief Optimize a whole trajectory along the given global path.
   *  @param global_path Global waypoints guiding trajectory optimization.
   *  @return True if trajectory planning succeeded.
   */
  bool planTrajectory(vec_Vecf<3>& global_path,
                           GlobalPlanStage* failure_stage = nullptr,
                           std::string* failure_message = nullptr);

  std::vector<double> shapeYawInitialGuess(std::size_t knot_count) const;

  bool buildYawOnlyTrajectory(const state& start, const state& goal);

  /** @brief Get the piecewise quintic polynomial representation of the current trajectory.
   *  @param pwp Output piecewise quintic polynomial.
   */
  void getPiecewiseQuinticPol(PieceWiseQuinticPol& pwp);

 private:
  // Parameters
  parameters par_;          // Parameters of the planner
  std::unique_ptr<mighty::shape_aware_astar::ShapeAwareAstarPlanner>
      shape_planner_;
  std::shared_ptr<const mighty::whole_body_safety::SafetyScene> safety_scene_;
  std::shared_ptr<mighty::whole_body_safety::WholeBodySafety> whole_body_safety_;
  std::shared_ptr<mighty::whole_body_safety::SafetyQueryContext>
      whole_body_safety_context_;
  TrajectoryRoute trajectory_route_{TrajectoryRoute::NONE};
  double middle_stage_time_ms_{0.0};
  double back_end_preparation_time_ms_{0.0};
  double back_end_optimization_time_ms_{0.0};
  double back_end_retime_time_ms_{0.0};
  double trajectory_export_time_ms_{0.0};
  double plan_commit_time_ms_{0.0};
  std::size_t risk_segment_count_{0};
  std::size_t risk_segment_edges_{0};
  std::size_t risk_window_count_{0};
  std::size_t trajectory_segment_count_{0};
  double trajectory_yaw_variation_{-1.0};
  std::vector<mighty::frontend::PoseState> active_shape_path_;
  std::mutex global_once_plan_mutex_;
  std::atomic<std::uint64_t> global_once_generation_{0};
  bool active_goal_yaw_exact_{true};
  std::chrono::steady_clock::time_point global_once_hard_deadline_{};
  std::chrono::steady_clock::time_point global_once_refinement_deadline_{};
  bool global_once_deadline_active_{false};
  std::vector<double> derivative_activation_scales_;
  int optimization_status_{0};
  bool refinement_timed_out_{false};

  GlobalPlanResult executeCertifiedShapePath(
      state captured_start, state exact_goal,
      mighty::frontend::PlanResult shape_result,
      double request_time);
  std::shared_ptr<lbfgs::SolverLBFGS>
      whole_traj_solver_ptr_;                    // L-BFGS solver pointer for the whole trajectory
  std::shared_ptr<lbfgs::SolverLBFGS> executable_traj_solver_;
  double trajectory_execution_epoch_{std::numeric_limits<double>::quiet_NaN()};
  Eigen::Vector3d v_max_3d_;                     // Maximum velocity
  Eigen::Vector3d a_max_3d_;                     // Maximum acceleration
  Eigen::Vector3d j_max_3d_;                     // Maximum jerk
  double v_max_;                                 // Maximum speed
  double max_dist_vertexes_{0.0};                // Maximum path-knot spacing
  lbfgs::planner_params_t planner_params_;       // Planner parameters
  lbfgs::lbfgs_parameter_t lbfgs_params_;        // L-BFGS parameters

  // Flags
  bool state_initialized_ = false;          // State initialized
  bool terminal_goal_initialized_ = false;  // Terminal goal initialized
  std::atomic<bool> global_once_committed_{false};

  // Data
  double final_g_ = 0.0;
  double global_planning_time_ = 0.0;
  double initial_guess_computation_time_ = 0.0;
  double trajectory_optimization_time_ = 0.0;
  std::vector<state> middle_goal_setpoints_;
  std::vector<state> goal_setpoints_;
  std::vector<Eigen::Matrix<double, 3, 6>> cps_;

  // Basis
  Eigen::Matrix<double, 4, 4> A_rest_pos_basis_;
  Eigen::Matrix<double, 4, 4> A_rest_pos_basis_inverse_;

  // Replanning-related variables
  state state_;                                     // State for the drone
  state G_;                                         // This goal is always inside of the map
  state A_;                                         // Starting point of the drone
  double A_time_;                                   // Time of the starting point
  state E_;                                         // The goal point of actual trajectory
  state G_term_;                                    // Terminal goal
  std::deque<state> plan_;                          // Plan for the drone
  double previous_yaw_ = 0.0;                       // Previous yaw
  double prev_dyaw_ = 0.0;                          // Previous dyaw
  double dyaw_filtered_ = 0.0;                      // Filtered dyaw
  PieceWiseQuinticPol pwp_to_share_;                // Piecewise polynomial to share

  // Drone status
  std::atomic<int> drone_status_{
      DroneStatus::GOAL_REACHED};  // YAWING, TRAVELING, GOAL_SEEN, or GOAL_REACHED

  // Mutex
  std::mutex mtx_plan_;                  // Mutex for the plan_
  std::mutex mtx_state_;                 // Mutex for the state_
  std::mutex mtx_G_;                     // Mutex for the G_
  std::mutex mtx_A_;                     // Mutex for the A_
  std::mutex mtx_A_time_;                // Mutex for the A_time_
  std::mutex mtx_G_term_;                // Mutex for the G_term_
  std::mutex mtx_E_;                     // Mutex for the E_
  std::mutex mtx_global_path_;           // Mutex for the global_path_
  // Counter for replanning failure
  int replanning_failure_count_ = 0;

  // Global path
  vec_Vecf<3> global_path_;

  // Store data
  Eigen::VectorXd zopt_;
  double fopt_;
};
