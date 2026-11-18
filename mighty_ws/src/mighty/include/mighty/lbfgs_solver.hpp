/* ----------------------------------------------------------------------------
 * Copyright 2025, Kota Kondo, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Kota Kondo, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>  // for std::log, std::pow
#include <numeric>
#include <optional>
#include <set>
#include <vector>

#include <Eigen/Dense>

#include <mighty/lbfgs.hpp>
#include <mighty/whole_body_safety.hpp>
#include <mighty/lbfgs_solver_utils.hpp>
#include <mighty/yaw_trajectory.hpp>

// ROS
#include <mighty/mighty_type.hpp>

namespace lbfgs {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat3xN = std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>>;
using MatXd = Eigen::MatrixXd;
using VecXd = Eigen::VectorXd;

struct planner_params_t {
  bool verbose = false;      // verbosity level
  double V_max = 2.0;        // max velocity
  double A_max = 10.0;       // max acceleration
  double J_max = 100.0;      // max jerk
  double time_weight = 1.0;
  double pos_anchor_weight = 1.0;
  double reference_weight = 0.0;
  double minimum_forward_speed = 0.0;
  double forward_progress_weight = 0.0;
  double jerk_weight = 10.0;
  double yaw_jerk_weight = 0.1;
  double yaw_reference_weight = 0.0;
  bool spatial_progress_yaw_coupling = true;
  double w_max = 1.0;
  double yaw_acc_max = 2.0;
  double dyn_constr_bodyrate_weight = 10.0;  // weight for body rate constraints
  double dyn_constr_tilt_weight = 10.0;      // weight for tilt constraints
  double dyn_constr_thrust_weight = 10.0;    // weight for
  double dyn_constr_vel_weight = 10.0;
  double dyn_constr_acc_weight = 10.0;
  double dyn_constr_jerk_weight = 0.0;
  double dc = 0.01;               // discretization constant
  int integral_resolution = 30;   // resolution for the integral in the optimization
  double hinge_mu = 1e-2;         // hinge mu for the optimization
  double omega_max = 1.0;         // max body rate in rad/s
  double tilt_max_rad = 0.6;      // max tilt in radians (e.g., 35° in rad)
  double f_min = 0.0;             // min thrust in N
  double f_max = 20.0;            // max thrust in N
  double mass = 1.0;              // mass in kg
  double g = 9.81;                // gravity in m/s^2
  int spline_degree = 5;          // Hermite spline degree: 3 (cubic) or 5 (quintic)
  bool topp_init = true;
  bool light_activation_ladder = true;
};

/** @brief L-BFGS-based whole-trajectory optimizer for Hermite quintic splines.
 *
 *  Optimizes piecewise quintic Hermite polynomial trajectories subject to
 *  dynamic constraints (velocity, acceleration, jerk, bodyrate, tilt, thrust)
 *  and costs (obstacle avoidance, smoothness, time).
 */
class SolverLBFGS {
 public:
  /** @brief Default constructor. */
  SolverLBFGS();

  /** @brief Destructor. */
  ~SolverLBFGS();

  // -----------------------------------------------------------------------------

  // ------------------------------
  // Functions
  // ------------------------------

  /** @brief Get the global path used by the solver.
   *  @param global_path Output global waypoint path.
   */
  void getGlobalPath(vec_Vecf<3>& global_path);

  // -----------------------------------------------------------------------------
  /** @brief Evaluate the objective and gradient in a single fused pass.
   *  @param z Decision variable vector.
   *  @param grad Output gradient vector.
   *  @return Scalar objective value.
   */
  double evaluateObjectiveAndGradientFused(const Eigen::VectorXd& z, Eigen::VectorXd& grad);

  // -----------------------------------------------------------------------------

  /**
   * @brief [Debug] Progress callback for L-BFGS, called at each iteration.
   *
   * @param instance  Pointer to the SolverLBFGS instance (unused).
   * @param x         Current decision variables.
   * @param g         Current gradient.
   * @param fx        Current objective value.
   * @param step      Step size (unused).
   * @param k         Iteration number.
   * @param ls        Line search status (unused).
   * @return          0 to continue, non-zero to stop.
   */
  static int progressCallback(void* instance, const Eigen::VectorXd& z, const Eigen::VectorXd& g,
                              const double f, const double step, const int k, const int ls);

  // -----------------------------------------------------------------------------

  /**
   * @brief Optimize the trajectory using L-BFGS.
   *
   * @param z0      Initial guess for decision variables (size K).
   * @param z_opt   Output optimized decision variables (size K).
   * @param f_opt   Output final objective value.
   * @param param   L-BFGS parameters.
   * @return       Status code: 0 on success, negative on error.
   */
  int optimize(const Eigen::VectorXd& z0, Eigen::VectorXd& z_opt, double& f_opt,
               const lbfgs::lbfgs_parameter_t& param) const;

  void setOptimizationDeadline(
      std::optional<std::chrono::steady_clock::time_point> deadline) {
    optimization_deadline_ = deadline;
    optimization_timed_out_ = false;
  }

  bool optimizationTimedOut() const { return optimization_timed_out_; }

  // -----------------------------------------------------------------------------

  /**
   * @brief Reconstruct knot states and Hermite control points from decision vector z.
   *
   * @param z    Decision vector of length K_cp_ + K_sig_:
   *             [p₀,v₀,a₀, �?, p_M,v_M,a_M, σ₀, �?, σ_{M-1}]
   * @param P    (output) Knot positions, size M_+1
   * @param V    (output) Knot velocities, size M_+1
   * @param A    (output) Knot accelerations, size M_+1
   * @param CP   (output) Hermite control points, size M_×6
   * @param T    (output) Segment durations, size M_
   */
  void reconstruct(const VecXd& z, std::vector<Vec3>& P, std::vector<Vec3>& V, std::vector<Vec3>& A,
                   std::vector<std::vector<Vec3>>& CP, std::vector<double>& T) const;

  // -----------------------------------------------------------------------------

  /**
   * @brief Pack knot states and durations into the decision vector z.
   *
   * Inverse of reconstruct(...). Endpoint states are boundary conditions, so
   * only interior position, velocity, acceleration, yaw derivatives, and all
   * segment durations are written to the decision vector.
   *
   * @param P   Knot positions, length M_+1
   * @param V   Knot velocities, length M_+1
   * @param A   Knot accelerations, length M_+1
   * @param yaw_knots Yaw, yaw rate, and yaw acceleration, length M_+1
   * @param T   Segment durations, length M_
   * @param z   Output decision vector
   */
  void packDecisionVariables(const std::vector<Vec3>& P, const std::vector<Vec3>& V,
                             const std::vector<Vec3>& A,
                             const std::vector<mighty::YawKnot>& yaw_knots,
                             const std::vector<double>& T,
                             VecXd& z) const;

  // -----------------------------------------------------------------------------

  void setWholeBodySafety(
      std::shared_ptr<mighty::whole_body_safety::WholeBodySafety> safety,
      std::shared_ptr<mighty::whole_body_safety::SafetyQueryContext> context,
      mighty::whole_body_safety::SafetySamplingPolicy sampling_policy = {}) {
    whole_body_safety_ = std::move(safety);
    whole_body_safety_context_ = std::move(context);
    safety_sampling_policy_ = sampling_policy;
  }

  /**
   * Restrict refinement to the supplied contiguous path segments.  Knot
   * variables remain free only when both adjacent segments are active, which
   * keeps the TOPP seed's position, velocity, acceleration, and yaw state at
   * every active/inactive join unchanged.  An empty vector restores whole-path
   * refinement.
   */
  bool setRiskLocalRefinementSegments(const std::vector<bool>& segments);

  // Grow the fixed safety quadrature from the currently reconstructed
  // trajectory. Returns true when another optimization pass is required.
  bool refineSafetySamplingForCurrentTrajectory();
  bool refineYawVariationPenaltyForCurrentTrajectory();

  bool hasRiskLocalRefinementVariables() const;

  // -----------------------------------------------------------------------------
  // ROS-related functions
  // -----------------------------------------------------------------------------

  /**
   * @brief Sanity check function to verify the solver's setup.
   */
  void sanityCheck() const;

  /**
   * @brief initialize the solver with default parameters.
   */
  void initializeSolver(const planner_params_t& params);

  /** @brief Prepare the solver for a new replan iteration.
   *  @param t0 Start time of the trajectory.
   *  @param global_wps Global waypoints.
   *  @param initial_state Start state (position, velocity, acceleration).
   *  @param goal_state Goal state.
   *  @param initial_yaws Reference yaw at each global waypoint.
   *  @param fix_goal_yaw Whether the terminal yaw is fixed exactly.
   *  @param initial_guess_computation_time Output time for initial guess computation.
   */
  void prepareSolverForReplan(double t0, const vec_Vec3f& global_wps,
                              const state& initial_state, const state& goal_state,
                              const std::vector<double>& initial_yaws,
                              bool fix_goal_yaw,
                              double& initial_guess_computation_time);

  /** @brief Compute the six quintic Hermite control points for one segment.
   *  @param P0 Start position.
   *  @param V0 Start velocity.
   *  @param A0 Start acceleration.
   *  @param P1 End position.
   *  @param V1 End velocity.
   *  @param A1 End acceleration.
   *  @param T Segment duration.
   *  @return Array of 6 control points.
   */
  static std::vector<Vec3> computeQuinticCP(const Vec3& P0, const Vec3& V0, const Vec3& A0,
                                              const Vec3& P1, const Vec3& V1, const Vec3& A1,
                                              double T);

  /** @brief Solve for minimum-jerk interior velocities and accelerations in closed form.
   *  @param wps Waypoints (size M+1).
   *  @param T Segment durations (size M).
   *  @param v0 Initial velocity.
   *  @param a0 Initial acceleration.
   *  @param vf Final velocity.
   *  @param af Final acceleration.
   *  @param V Output velocities at each waypoint (size M+1).
   *  @param A Output accelerations at each waypoint (size M+1).
   */
  static void solveMinJerkVelAcc(const std::vector<Vec3>& wps,  // size M+1
                                 const std::vector<double>& T,  // size M
                                 const Vec3& v0, const Vec3& a0, const Vec3& vf, const Vec3& af,
                                 std::vector<Vec3>& V,   // out size M+1
                                 std::vector<Vec3>& A);  // out size M+1

  /** @brief Elementwise softplus activation on a Vec3.
   *  @param v Input vector.
   *  @return Softplus applied element-wise: log(1 + exp(x)).
   */
  static Vec3 softplus_vec(const Vec3& v) {
    return v.unaryExpr([](double x) { return std::log1p(std::exp(x)); });
  }

  /** @brief Scalar softplus activation: log(1 + exp(x)).
   *  @param x Input value.
   *  @return Softplus result.
   */
  static double softplus(double x) { return std::log1p(std::exp(x)); }

  /** @brief Compute the 4x4 stiffness matrix K_r for a segment of duration T.
   *  @param T Segment duration.
   *  @return 4x4 stiffness matrix.
   */
  static inline Eigen::Matrix4d K_r(double T);

  /** @brief Compute the 4-element RHS vector k_r for a segment.
   *  @param T Segment duration.
   *  @param P0 Start position (1D).
   *  @param P1 End position (1D).
   *  @return Array of 4 RHS values.
   */
  static inline std::array<double, 4> k_r(double T, double P0, double P1);

  /**
   * Assemble the block‐tridiagonal Hessian H and RHS b for the closed-form min-jerk.
   *
   * Inputs:
   *   wps  �? M+1 waypoints (Vec3)
   *   T    �? M segment times
   *   v0,a0,vf,af �? endpoint velocity/accel
   *
   * Outputs:
   *   H (2N×2N), b (2N×3), with N = M�?1
   */
  static void assemble_H_b(const std::vector<Vec3>& wps, const std::vector<double>& T,
                           const Vec3& v0, const Vec3& a0, const Vec3& vf, const Vec3& af,
                           Eigen::MatrixXd& H, Eigen::MatrixXd& b);

  /** @brief Get the goal setpoints computed during the last optimization.
   *  @param goal_setpoints Output vector of goal states at knot times.
   */
  void getGoalSetpoints(std::vector<state>& goal_setpoints,
                        bool include_boundary_samples = false);

  /** @brief Return the optimized segment durations. */
  const std::vector<double>& getSegmentTimes() const { return T_opt_; }

  /** @brief Evaluate the cached analytic trajectory at elapsed time. */
  bool evaluateAtElapsedTime(double elapsed_time, state& sample) const;

  /** @brief Per-knot fraction of minimum-jerk derivatives admitted safely. */
  const std::vector<double>& getDerivativeActivationScales() const {
    return derivative_activation_scales_;
  }

  /** Return the coupled translation/yaw seed paired with the primary seed. */
  bool getCoupledDerivativeSeed(
      Eigen::VectorXd& decision,
      std::vector<double>& activation_scales);

  /** @brief Pack the currently reconstructed trajectory after any retiming. */
  bool getCurrentDecisionVariables(Eigen::VectorXd& decision) const;

  /** Build a conservative decision vector with every interior knot at rest. */
  bool makeRestToRestDecisionVariables(Eigen::VectorXd& decision) const;

  /** @brief Uniformly slow the cached optimized trajectory without changing its geometry.
   *
   *  Segment durations are multiplied by scale while knot velocities and
   *  accelerations are divided by scale and scale squared respectively.
   *  Cached Bernstein control points are rebuilt to keep every exported
   *  representation consistent.
   */
  bool retimeForDynamics(double margin = 1.01);

  /** @brief Get the piecewise quintic polynomial representation of the optimized trajectory.
   *  @param pwp Output piecewise quintic polynomial.
   */
  void getPieceWisePol(PieceWiseQuinticPol& pwp);

  /** @brief Get the control points from the last optimized trajectory.
   *  @param cps Output vector of 3x6 control point matrices per segment.
   */
  void getControlPoints(std::vector<Eigen::Matrix<double, 3, 6>>& cps);

  /** @brief Reconstruct and cache optimal positions, velocities, accelerations, times, and CPs.
   *  @param z Optimized decision variable vector.
   */
  void reconstructPVATCPopt(const Eigen::VectorXd& z);

  /** @brief Evaluate position, velocity, acceleration, and jerk at a point on the trajectory.
   *  @param segment Segment index.
   *  @param tau Normalized time within the segment [0, 1].
   *  @return StateDeriv containing pos, vel, acc, jerk.
   */
  StateDeriv evalStateDeriv(int segment, double tau) const;

  /** @brief Evaluate yaw and its derivatives at normalized segment phase. */
  mighty::YawSample evalYawState(int segment, double tau) const;

  /** @brief Check exact yaw-rate and yaw-acceleration extrema. */
  bool yawDynamicsFeasible(double tolerance = 1.0e-9) const;

  /** @brief Validate sampled UAV translational and flatness-derived limits. */
  bool uavDynamicsFeasible(double relative_tolerance = 1.0e-3) const;

  /** @brief Get the position along one axis at a point on the trajectory.
   *  @param segment Segment index.
   *  @param tau Normalized time within the segment [0, 1].
   *  @param axis Axis index (0=x, 1=y, 2=z).
   *  @return Position value.
   */
  inline double getPosDouble(int segment, double tau, int axis) const;

  /** @brief Get the velocity along one axis at a point on the trajectory.
   *  @param segment Segment index.
   *  @param tau Normalized time within the segment [0, 1].
   *  @param axis Axis index (0=x, 1=y, 2=z).
   *  @return Velocity value.
   */
  inline double getVelDouble(int segment, double tau, int axis) const;

  /** @brief Get the acceleration along one axis at a point on the trajectory.
   *  @param segment Segment index.
   *  @param tau Normalized time within the segment [0, 1].
   *  @param axis Axis index (0=x, 1=y, 2=z).
   *  @return Acceleration value.
   */
  inline double getAccelDouble(int segment, double tau, int axis) const;

  void reconstructYaw(const Eigen::VectorXd& z,
                      const std::vector<double>& segment_times,
                      std::vector<mighty::YawKnot>& knots,
                      std::vector<std::array<double, 6>>& control_points) const;

  /** @brief Get the jerk along one axis at a point on the trajectory.
   *  @param segment Segment index.
   *  @param tau Normalized time within the segment [0, 1].
   *  @param axis Axis index (0=x, 1=y, 2=z).
   *  @return Jerk value.
   */
  inline double getJerkDouble(int segment, double tau, int axis) const;

  // -----------------------------------------------------------------------------

  // -------------------------------
  // Getters
  // -------------------------------

  // -----------------------------------------------------------------------------

  /**
   * @brief Get the number of decision variables (free CPs + slack times).
   * @return Number of decision variables K = 3F + M.
   */
  int getNumDecisionVariables() const { return K_; }

  // -----------------------------------------------------------------------------

  // -----------------------------------------------------------------------------

  /**
   * @brief Get the number of segments in the trajectory.
   * @return Number of segments M = global_wps_.size() - 1.
   */
  int getNumSegments() const { return static_cast<int>(global_wps_.size()) - 1; }

  // -----------------------------------------------------------------------------

  /**
   * @brief Get list_z0, the initial guesses for decision variables.
   * @return List of initial guesses for decision variables (size N).
   */
  const std::vector<Eigen::VectorXd>& getInitialGuesses() const { return list_z0_; }

  // -----------------------------------------------------------------------------

  /**
   * @brief Get the initial guess waypoints.
   * @return List of initial guess waypoints (size N).
   */
  const std::vector<std::vector<Vec3>>& getInitialGuessWaypoints() const {
    return initial_guess_wps_;
  }

  // -----------------------------------------------------------------------------

  /** @brief Numerically stable central-difference directional derivative.
   *  @return  (f(z + eps*d) - f(z - eps*d)) / (2*eps), with eps scaled by ||d||.
   */
  double centralDiff(const VecXd& z, const VecXd& d, double eps_base);

  /** @brief Verify analytic gradient against finite differences in random directions.
   *  @return  Worst observed relative error across all sampled directions.
   */
  double checkGradDirectional(const VecXd& z0, int num_dirs, double eps, unsigned seed);

  /** @brief Verify analytic gradient coordinate-by-coordinate against finite differences.
   *  @return  Worst observed relative error across all sampled coordinates.
   */
  double checkGradCoordinates(const VecXd& z0, int max_coords, double eps, unsigned seed);

  // -----------------------------------------------------------------------------

 private:
  bool adjacentSegmentsWholeBodySafe(
      const std::vector<Vec3>& positions,
      const std::vector<Vec3>& velocities,
      const std::vector<Vec3>& accelerations,
      const std::vector<mighty::YawKnot>& yaw_knots,
      const std::vector<double>& durations, int knot);
  /**
   * @brief Static wrapper for evaluateObjectiveAndGradient.
   * @return The objective value at x; also fills g.
   */
  static double evalObjGradCallback(void* instance, const Eigen::VectorXd& x, Eigen::VectorXd& g);
  bool optimizationDeadlineExpired() const;

  // -----------------------------------------------------------------------------

  /// Find which segment index s contains time t_i
  int findSegment(double ti, const std::vector<double>& T) const;

  /// Evaluate Bernstein basis (degree=5) and derivative at tau
  void evalBernstein5(double tau, double B[6], double dB[6]) const;

  /**
   * @brief Evaluate one trajectory sample u_i and its segment/tau.
   *
   * @param CP           full control–point array [M][6]
   * @param T            segment durations (size M)
   * @param sample_i     index i in [0..num_samples-1]
   * @param num_samples total number of samples
   * @param[out] seg    segment index containing sample i
   * @param[out] tau    normalized time in [0,1] within that segment
   * @param[out] t0     absolute start time of that segment
   * @return            u_i position in ℝ�?
   */
  Eigen::Vector3d evalSample(const std::vector<std::array<Eigen::Vector3d, 6>>& CP,
                             const std::vector<double>& T, int sample_i, int num_samples, int& seg,
                             double& tau, double& t0) const;

  /// Evaluate obstacle position at time t
  Eigen::Vector3d evalObs(const Eigen::Matrix<double, 6, 1>& cx,
                          const Eigen::Matrix<double, 6, 1>& cy,
                          const Eigen::Matrix<double, 6, 1>& cz, double t) const;

  /**
   * Build a small Jacobian ∂u_i/∂z for one sample:
   *   - only the free‐CP columns in segment s (3 per cp)
   *   - plus the two slack columns σ_s and σ_{s+1}
   */
  // in lbfgs_solver.hpp, under your class SolverLBFGS
  void buildSparseDuDzSample(
      const Eigen::VectorXd& z, const std::vector<std::array<Eigen::Vector3d, 6>>& CP,
      const std::vector<double>& T, int sample_i, int num_samples, int seg, double tau,
      const std::vector<std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>>>& dPdz,
      const Eigen::MatrixXd& dTdz, std::vector<int>& cols, Eigen::MatrixXd& Jsmall) const;

  /**
   * Build a small Jacobian ∂k_i/∂z for one sample:
   *   - only the M slack columns (but will be the same α factor for each)
   */
  void buildSparseDkDzSample(const Eigen::VectorXd& z, const std::vector<double>& T, int sample_i,
                             int num_samples, const Eigen::Matrix<double, 6, 1>& cx,
                             const Eigen::Matrix<double, 6, 1>& cy,
                             const Eigen::Matrix<double, 6, 1>& cz, std::vector<int>& cols,
                             Eigen::MatrixXd& Ksmall) const;

  void buildSparseDkDzSample(const Eigen::VectorXd& z, const std::vector<double>& T, int sample_i,
                             int num_samples, const PieceWisePol& pw, std::vector<int>& cols,
                             Eigen::MatrixXd& Ksmall) const;

  // -----------------------------------------------------------------------------
  // Helper functions for ROS
  // -----------------------------------------------------------------------------

  inline int nCk5(int n, int k) {
    // Pascal’s triangle up to row 5
    static constexpr int C[6][6] = {{1, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0}, {1, 2, 1, 0, 0, 0},
                                    {1, 3, 3, 1, 0, 0}, {1, 4, 6, 4, 1, 0}, {1, 5, 10, 10, 5, 1}};
    return C[n][k];
  }

 protected:
  // ------------------------------
  // Parameters
  // ------------------------------

  // Parameters
  int M_;                   // Number of segments (this now is global_wps_.size() - 1)
  int K_;                   // Number of decision variables (CPs + slack times)
  int K_cp_;                // Number of non-time decision variables
  int K_pos_;               // Position PVA decision variables
  int K_yaw_;               // Yaw knot decision variables
  int K_sig_;               // Number of slack times (K_sig_ = M_)
  bool verbose_{false};     // Verbosity flag

  // Spline degree and derived constants
  int degree_{5};              // 3 or 5
  int num_cp_{6};              // degree + 1: control points per segment
  int num_derivs_{2};          // 1 for cubic (P,V), 2 for quintic (P,V,A)
  int dof_per_knot_{9};        // 3*(1+num_derivs): 6 or 9
  double vel_scale_{5.0};      // degree
  double acc_scale_{20.0};     // degree*(degree-1)
  double jrk_scale_{60.0};     // degree*(degree-1)*(degree-2)
  double jrk_cost_coeff_{3600.0}; // closed-form jerk integral coefficient

  std::shared_ptr<mighty::whole_body_safety::WholeBodySafety>
      whole_body_safety_;
  std::shared_ptr<mighty::whole_body_safety::SafetyQueryContext>
      whole_body_safety_context_;
  mighty::whole_body_safety::SafetySamplingPolicy safety_sampling_policy_;
  std::vector<int> optimization_safety_intervals_;
  std::vector<bool> risk_local_segments_;
  std::vector<unsigned char> refinement_variable_mask_;

  // Ego-agent boundary conditions
  Vec3 x0_;
  Vec3 v0_;
  Vec3 a0_;
  Vec3 xf_;
  Vec3 vf_;
  Vec3 af_;
  std::vector<double> initial_yaws_;
  double yaw0_{0.0};
  double yawf_{0.0};
  bool fix_goal_yaw_{true};

  // Global waypoints
  std::vector<Vec3> global_wps_;
  std::vector<double> derivative_activation_scales_;
  Eigen::VectorXd coupled_derivative_seed_;
  std::vector<double> coupled_derivative_activation_scales_;
  bool coupled_derivative_seed_pending_{false};
  std::vector<Vec3> derivative_seed_positions_;
  std::vector<Vec3> derivative_seed_target_velocities_;
  std::vector<Vec3> derivative_seed_target_accelerations_;
  std::vector<mighty::YawKnot> derivative_seed_target_yaw_knots_;
  std::vector<double> derivative_seed_durations_;
  std::optional<std::chrono::steady_clock::time_point> optimization_deadline_;
  mutable bool optimization_timed_out_{false};

  // Dynamic constraints
  double V_max_;
  double A_max_;
  double J_max_;

  // sampling & smoothing
  int integral_resolution_ = 30;
  double hinge_mu_ = 1e-2;

  // vehicle/dynamics limits and constants
  double omega_max_ = 6.0;                     // rad/s
  double tilt_max_rad_ = M_PI * 35.0 / 180.0;  // 35 deg
  double f_min_ = 0.0;                         // N
  double f_max_ = 20.0;                        // N
  double mass_ = 1.0;                          // kg
  double g_ = 9.81;                            // m/s^2

  // Position anchoring

  // Optimization weights and settings
  double time_weight_;
  double pos_anchor_weight_;
  double reference_weight_;
  double minimum_forward_speed_{0.0};
  double forward_progress_weight_{0.0};
  double jerk_weight_;
  double yaw_jerk_weight_{0.1};
  double yaw_reference_weight_{0.0};
  bool spatial_progress_yaw_coupling_{true};
  double yaw_variation_penalty_weight_{0.0};
  double w_max_{1.0};
  double yaw_acc_max_{2.0};
  double dyn_constr_vel_weight_;
  double dyn_constr_acc_weight_;
  double dyn_constr_jerk_weight_;
  double dyn_constr_bodyrate_weight_ = 1.0;
  double dyn_constr_tilt_weight_ = 1.0;
  double dyn_constr_thrust_weight_ = 1.0;
  // Position anchoring
  std::vector<Vec3> P_anchor_;

  // Constants
  // Optimization variables
  mutable std::vector<double> t_abs_;

  // Lists for initial guesses
  std::vector<std::vector<Vec3>> initial_guess_wps_;
  std::vector<Eigen::VectorXd> list_z0_;

  // Small value
  double eps_ = 1e-6;

  // ROS parameters
  std::vector<state> goal_setpoints_;  // vector of goal setpoints
  double dc_;                          // small value
  std::vector<std::vector<Vec3>> CP_opt_;
  std::vector<double> T_opt_;
  std::vector<Vec3> P_opt_, V_opt_, A_opt_;
  std::vector<mighty::YawKnot> yaw_knots_opt_;
  std::vector<std::array<double, 6>> yaw_cp_opt_;
  double t0_{0.0};  // start time of the trajectory
  bool topp_init_{true};
  bool light_activation_ladder_{true};

};  // class SolverLBFGS

}  // namespace lbfgs
