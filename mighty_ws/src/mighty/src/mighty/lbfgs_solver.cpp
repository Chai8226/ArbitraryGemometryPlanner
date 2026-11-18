/* ----------------------------------------------------------------------------
 * Copyright 2025, Kota Kondo, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Kota Kondo, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

#include <unsupported/Eigen/Polynomials>

#include <mighty/lbfgs_solver.hpp>
#include <mighty/pose_interpolation.hpp>
#include <mighty/topp_seed.hpp>

using namespace lbfgs;

// H-polyhedron: rows are [n_x n_y n_z d] meaning n^T x <= d
// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

// GCOPTER τ→T and dT/dτ (exactly matches Python)
inline double tau_to_T(double tau) {
  if (tau >= 0.0) return 0.5 * tau * tau + tau + 1.0;
  const double den = 0.5 * tau * tau - tau + 1.0;
  return 1.0 / den;
}
inline double dT_dtau(double tau) {
  if (tau >= 0.0) return tau + 1.0;
  const double den = 0.5 * tau * tau - tau + 1.0;
  return (1.0 - tau) / (den * den);
}
inline double T_to_tau(double T) {
  if (T >= 1.0) {
    double x = std::max(2.0 * T - 1.0, 0.0);
    return std::sqrt(x) - 1.0;
  }
  double x = std::max(2.0 / T - 1.0, 0.0);
  return 1.0 - std::sqrt(x);
}

// Build per-knot T̄ = [ T0, 0.5(T0+T1), ..., 0.5(T_{M-2}+T_{M-1}), T_{M-1} ]
inline void build_Tbar(const std::vector<double>& T, std::vector<double>& Tbar) {
  const int M = (int)T.size();
  Tbar.resize(M + 1);
  Tbar[0] = T[0];
  for (int i = 1; i < M; ++i) Tbar[i] = 0.5 * (T[i - 1] + T[i]);
  Tbar[M] = T[M - 1];
}

// Distribute gTbar[k] (knot derivatives) onto segment derivatives gTextra[s]:
//   dTbar_0/dT0 = 1
//   dTbar_i/dT_{i-1} = 1/2, dTbar_i/dT_i = 1/2  for i=1..M-1
//   dTbar_M/dT_{M-1} = 1
inline void distribute_gTbar_to_segments(const std::vector<double>& gTbar,
                                         std::vector<double>& gTextra) {
  const int M = (int)gTextra.size();
  // safety: expect gTbar.size() == M+1
  if ((int)gTbar.size() != M + 1) return;
  // clear
  std::fill(gTextra.begin(), gTextra.end(), 0.0);
  // endpoints
  gTextra[0] += gTbar[0];
  gTextra[M - 1] += gTbar[M];
  // interiors
  for (int i = 1; i < M; ++i) {
    const double v = 0.5 * gTbar[i];
    gTextra[i - 1] += v;  // from Tbar_i wrt T_{i-1}
    gTextra[i] += v;      // from Tbar_i wrt T_i
  }
}

// --- GCOPTER cubic hinge (C²) and its derivative ---
// ψ(x; μ) = 0                         , x �? 0
//        = (μ - x/2) (x/μ)^3          , 0 < x < μ
//        = x - μ/2                    , x �? μ
inline double smoothed_l1(double x, double mu) {
  if (x <= 0.0) return 0.0;
  if (x >= mu) return x - 0.5 * mu;
  const double inv_mu = 1.0 / mu;
  const double xd = x * inv_mu;
  const double xd3 = xd * xd * xd;
  return (mu - 0.5 * x) * xd3;
}

// dψ/dx = (x^2 / μ^3) (3μ �? 2x)
inline double smoothed_l1_prime(double x, double mu) {
  if (x <= 0.0) return 0.0;
  if (x >= mu) return 1.0;
  const double inv_mu = 1.0 / mu;
  const double x2 = x * x;
  return (x2 * (3.0 * mu - 2.0 * x)) * (inv_mu * inv_mu * inv_mu);
}

inline void bernstein5(double u, double B[6]) {
  const double um = 1.0 - u;
  const double u2 = u * u, u3 = u2 * u, u4 = u3 * u, u5 = u4 * u;
  const double m2 = um * um, m3 = m2 * um, m4 = m3 * um, m5 = m4 * um;
  B[0] = m5;
  B[1] = 5.0 * u * m4;
  B[2] = 10.0 * u2 * m3;
  B[3] = 10.0 * u3 * m2;
  B[4] = 5.0 * u4 * um;
  B[5] = u5;
}

inline void bernstein4(double u, double b[5]) {
  const double um = 1.0 - u;
  const double u2 = u * u, u3 = u2 * u, u4 = u3 * u;
  const double m2 = um * um, m3 = m2 * um, m4 = m3 * um;
  b[0] = m4;
  b[1] = 4.0 * u * m3;
  b[2] = 6.0 * u2 * m2;
  b[3] = 4.0 * u3 * um;
  b[4] = u4;
}

// Cubic Bernstein basis B^3_k(s), k=0..3
inline void bernstein3(double s, double B[4]) {
  // (optional) clamp for numeric safety
  if (s < 0.0) s = 0.0;
  if (s > 1.0) s = 1.0;

  const double t = 1.0 - s;
  const double t2 = t * t;
  const double s2 = s * s;

  B[0] = t * t2;        // (1 - s)^3
  B[1] = 3.0 * s * t2;  // 3 s (1 - s)^2
  B[2] = 3.0 * s2 * t;  // 3 s^2 (1 - s)
  B[3] = s * s2;        // s^3
}

// Quadratic Bernstein basis B^2_k(s), k=0..2
inline void bernstein2(double s, double B[3]) {
  // (optional) clamp for numeric safety
  if (s < 0.0) s = 0.0;
  if (s > 1.0) s = 1.0;

  const double t = 1.0 - s;

  B[0] = t * t;        // (1 - s)^2
  B[1] = 2.0 * s * t;  // 2 s (1 - s)
  B[2] = s * s;        // s^2
}

// Linear Bernstein basis B^1_k(s), k=0..1
inline void bernstein1(double s, double B[2]) {
  B[0] = 1.0 - s;
  B[1] = s;
}

inline int zcp_offset_for_knot(int i, int M, int dof_per_knot = 9) {
  return (i <= 0 || i >= M) ? -1 : dof_per_knot * (i - 1);
}

inline void accumulate_cp_to_pvaT(int s, const std::vector<Vec3>& gCPs,
                                  const std::vector<Vec3>& V, const std::vector<Vec3>& A,
                                  const std::vector<double>& T, std::vector<Vec3>& gP,
                                  std::vector<Vec3>& gV, std::vector<Vec3>& gA,
                                  std::vector<double>& gT_cp, int degree = 5) {
  const double Ts = T[s];

  if (degree == 3) {
    // Cubic: 4 CPs, gradient to (P, V) only
    gP[s]     += gCPs[0] + gCPs[1];
    gV[s]     += (Ts / 3.0) * gCPs[1];
    gP[s + 1] += gCPs[2] + gCPs[3];
    gV[s + 1] += (-Ts / 3.0) * gCPs[2];

    // �?/∂T via CP(T) dependence
    gT_cp[s] += gCPs[1].dot(V[s] / 3.0);
    gT_cp[s] += gCPs[2].dot((-1.0 / 3.0) * V[s + 1]);
  } else {
    // Quintic: 6 CPs, gradient to (P, V, A)
    const double Ts2 = Ts * Ts;

    gP[s] += gCPs[0] + gCPs[1] + gCPs[2];
    gV[s] += (Ts / 5.0) * gCPs[1] + (2.0 * Ts / 5.0) * gCPs[2];
    gA[s] += (Ts2 / 20.0) * gCPs[2];

    gP[s + 1] += gCPs[3] + gCPs[4] + gCPs[5];
    gV[s + 1] += (-2.0 * Ts / 5.0) * gCPs[3] + (-Ts / 5.0) * gCPs[4];
    gA[s + 1] += (Ts2 / 20.0) * gCPs[3];

    gT_cp[s] += gCPs[1].dot(V[s] / 5.0);
    gT_cp[s] += gCPs[2].dot((2.0 / 5.0) * V[s] + (Ts / 10.0) * A[s]);
    gT_cp[s] += gCPs[3].dot((-2.0 / 5.0) * V[s + 1] + (Ts / 10.0) * A[s + 1]);
    gT_cp[s] += gCPs[4].dot((-1.0 / 5.0) * V[s + 1]);
  }
}

inline std::array<double, 6> yaw_control_points(
    const mighty::YawKnot& start, const mighty::YawKnot& goal, double duration) {
  const double duration2 = duration * duration;
  return {start.yaw,
          start.yaw + duration * start.rate / 5.0,
          start.yaw + 2.0 * duration * start.rate / 5.0 +
              duration2 * start.accel / 20.0,
          goal.yaw - 2.0 * duration * goal.rate / 5.0 +
              duration2 * goal.accel / 20.0,
          goal.yaw - duration * goal.rate / 5.0,
          goal.yaw};
}

inline void accumulate_yaw_cp_gradient(
    int segment, const std::array<double, 6>& gradient_cp,
    const std::vector<mighty::YawKnot>& knots,
    const std::vector<double>& durations, std::vector<double>& gradient_yaw,
    std::vector<double>& gradient_rate, std::vector<double>& gradient_accel,
    std::vector<double>& gradient_time) {
  const double duration = durations[segment];
  const double duration2 = duration * duration;
  gradient_yaw[segment] +=
      gradient_cp[0] + gradient_cp[1] + gradient_cp[2];
  gradient_rate[segment] += duration * gradient_cp[1] / 5.0 +
                            2.0 * duration * gradient_cp[2] / 5.0;
  gradient_accel[segment] += duration2 * gradient_cp[2] / 20.0;

  gradient_yaw[segment + 1] +=
      gradient_cp[3] + gradient_cp[4] + gradient_cp[5];
  gradient_rate[segment + 1] +=
      -2.0 * duration * gradient_cp[3] / 5.0 -
      duration * gradient_cp[4] / 5.0;
  gradient_accel[segment + 1] += duration2 * gradient_cp[3] / 20.0;

  gradient_time[segment] +=
      gradient_cp[1] * knots[segment].rate / 5.0;
  gradient_time[segment] +=
      gradient_cp[2] * (2.0 * knots[segment].rate / 5.0 +
                        duration * knots[segment].accel / 10.0);
  gradient_time[segment] +=
      gradient_cp[3] * (-2.0 * knots[segment + 1].rate / 5.0 +
                        duration * knots[segment + 1].accel / 10.0);
  gradient_time[segment] -=
      gradient_cp[4] * knots[segment + 1].rate / 5.0;
}

std::vector<double> real_roots_in_unit_interval(
    std::vector<double> coefficients) {
  while (coefficients.size() > 1 &&
         std::abs(coefficients.back()) <= 1.0e-12) {
    coefficients.pop_back();
  }
  if (coefficients.size() <= 1) return {};
  Eigen::VectorXd polynomial(coefficients.size());
  for (std::size_t i = 0; i < coefficients.size(); ++i) {
    polynomial[static_cast<Eigen::Index>(i)] = coefficients[i];
  }
  Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
  solver.compute(polynomial);
  std::vector<double> roots;
  for (Eigen::Index i = 0; i < solver.roots().size(); ++i) {
    const auto& root = solver.roots()[i];
    if (std::abs(root.imag()) <= 1.0e-9 && root.real() > 0.0 &&
        root.real() < 1.0) {
      roots.push_back(root.real());
    }
  }
  return roots;
}

// ---- end: helpers ported from mighty_solver.cpp ----

// -----------------------------------------------------------------------------

SolverLBFGS::SolverLBFGS() {
  // Constructor implementation
}

// -----------------------------------------------------------------------------

SolverLBFGS::~SolverLBFGS() {
  // Destructor implementation
  // Clean up resources if necessary
}

// -----------------------------------------------------------------------------

void SolverLBFGS::initializeSolver(const planner_params_t& params) {
  // Initialize the solver with parameters that won't change throughout the mission
  verbose_ = params.verbose;                      // Verbosity level
  V_max_ = params.V_max;                          // Max velocity
  A_max_ = params.A_max;                          // Max acceleration
  J_max_ = params.J_max;                          // Max jerk
  time_weight_ = params.time_weight;              // Weight for time in the objective
  pos_anchor_weight_ = params.pos_anchor_weight;  // Weight for position anchors in the objective
  reference_weight_ = params.reference_weight;
  minimum_forward_speed_ = params.minimum_forward_speed;
  forward_progress_weight_ = params.forward_progress_weight;
  jerk_weight_ = params.jerk_weight;              // Weight for jerk in the objective
  yaw_jerk_weight_ = params.yaw_jerk_weight;
  yaw_reference_weight_ = params.yaw_reference_weight;
  spatial_progress_yaw_coupling_ = params.spatial_progress_yaw_coupling;
  w_max_ = params.w_max;
  yaw_acc_max_ = params.yaw_acc_max;
  dyn_constr_vel_weight_ = params.dyn_constr_vel_weight;  // Weight for dynamic velocity constraints
  dyn_constr_acc_weight_ =
      params.dyn_constr_acc_weight;  // Weight for dynamic acceleration constraints
  dyn_constr_jerk_weight_ = params.dyn_constr_jerk_weight;  // Weight for dynamic jerk constraints
  dyn_constr_bodyrate_weight_ = params.dyn_constr_bodyrate_weight;
  dyn_constr_tilt_weight_ = params.dyn_constr_tilt_weight;
  dyn_constr_thrust_weight_ = params.dyn_constr_thrust_weight;
  dc_ = params.dc;    // Discretization constant
  integral_resolution_ = params.integral_resolution;  // e.g., 30
  hinge_mu_ = params.hinge_mu;                        // e.g., 1e-2
  omega_max_ = params.omega_max;                      // e.g., 6.0
  tilt_max_rad_ = params.tilt_max_rad;                // e.g., 35° in rad
  f_min_ = params.f_min;
  f_max_ = params.f_max;
  mass_ = params.mass;
  g_ = params.g;
  topp_init_ = params.topp_init;
  light_activation_ladder_ = params.light_activation_ladder;
  // Spline degree and derived constants
  degree_ = params.spline_degree;
  assert(degree_ == 3 || degree_ == 5);
  num_cp_ = degree_ + 1;                                         // 4 or 6
  num_derivs_ = (degree_ == 3) ? 1 : 2;                         // cubic: P,V; quintic: P,V,A
  dof_per_knot_ = 3 * (1 + num_derivs_);                        // 6 or 9
  vel_scale_ = static_cast<double>(degree_);                     // 3 or 5
  acc_scale_ = static_cast<double>(degree_ * (degree_ - 1));    // 6 or 20
  jrk_scale_ = static_cast<double>(degree_ * (degree_ - 1) * (degree_ - 2)); // 6 or 60
  jrk_cost_coeff_ = (degree_ == 3) ? 36.0 : 3600.0;
}

// -----------------------------------------------------------------------------

// Numerically stable central-difference directional derivative
double SolverLBFGS::centralDiff(const VecXd& z, const VecXd& d, double eps_base) {
  // Scale step per IEEE advice
  double eps = eps_base / std::max(1.0, d.norm());
  VecXd zp = z + eps * d;
  VecXd zm = z - eps * d;
  // return (evaluateObjective(zp) - evaluateObjective(zm)) / (2.0 * eps);
  double fp, fm;
  VecXd g(z.size());
  fp = evaluateObjectiveAndGradientFused(zp, g);
  fm = evaluateObjectiveAndGradientFused(zm, g);
  return (fp - fm) / (2.0 * eps);
}

double SolverLBFGS::checkGradDirectional(const VecXd& z0, int num_dirs, double eps, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> N(0.0, 1.0);

  VecXd g(z0.size());
  // (void)evaluateObjectiveAndGradient(z0, g);
  (void)evaluateObjectiveAndGradientFused(z0, g);

  double max_rel_err = 0.0;
  for (int k = 0; k < num_dirs; ++k) {
    VecXd d(z0.size());
    for (int i = 0; i < d.size(); ++i) {
      d[i] = refinement_variable_mask_.empty() ||
                     refinement_variable_mask_[static_cast<std::size_t>(i)]
                 ? N(rng)
                 : 0.0;
    }

    // No projection needed; all variables are free.
    double n = d.norm();
    if (n == 0.0) {
      --k;
      continue;
    }        // extremely unlikely
    d /= n;  // makes eps a consistent step

    const double fd = centralDiff(z0, d, eps);
    const double ad = g.dot(d);
    const double denom = std::max(1e-12, std::abs(fd) + std::abs(ad));
    const double rel = std::abs(fd - ad) / denom;

    max_rel_err = std::max(max_rel_err, rel);
    if (rel > 1e-3)
      printf("\033[1;31m [dir %d] fd=%.6f ad=%.6f rel_err=%.6f \033[0m\n", k, fd, ad, rel);
  }
  return max_rel_err;
}

double SolverLBFGS::checkGradCoordinates(const VecXd& z0, int max_coords, double eps,
                                         unsigned seed) {
  VecXd g(z0.size());
  // (void)evaluateObjectiveAndGradient(z0, g);
  (void)evaluateObjectiveAndGradientFused(z0, g);

  std::vector<int> idx;
  idx.reserve(static_cast<std::size_t>(z0.size()));
  for (int index = 0; index < z0.size(); ++index) {
    if (refinement_variable_mask_.empty() ||
        refinement_variable_mask_[static_cast<std::size_t>(index)]) {
      idx.push_back(index);
    }
  }

  std::mt19937 rng(seed);
  std::shuffle(idx.begin(), idx.end(), rng);
  if ((int)idx.size() > max_coords) idx.resize(max_coords);

  double worst = 0.0;
  for (int i : idx) {
    VecXd ei = VecXd::Zero(z0.size());
    ei[i] = 1.0;  // unit basis direction (no projection)

    const double g_fd = centralDiff(z0, ei, eps);
    const double g_ad = g[i];
    const double abserr = std::abs(g_fd - g_ad);
    const double rel = abserr / std::max(1e-12, std::abs(g_fd) + std::abs(g_ad));
    worst = std::max(worst, rel);

    if (rel > 1e-3)
      printf("\033[1;31m [idx %d] g_fd=%.6f g_ad=%.6f abs_err=%.6f rel_err=%.6f \033[0m\n", i, g_fd,
             g_ad, abserr, rel);
  }
  return worst;
}

void SolverLBFGS::prepareSolverForReplan(double t0, const vec_Vec3f& global_wps,
                                         const state& initial_state, const state& goal_state,
                                         const std::vector<double>& initial_yaws,
                                         bool fix_goal_yaw,
                                         double& initial_guess_computation_time) {
  // Initialize the solver with parameters that changes for replanning
  t0_ = t0;  // Current time in the trajectory

  // Global waypoints
  global_wps_.clear();
  for (const auto& wp : global_wps) {
    global_wps_.emplace_back(wp.x(), wp.y(), wp.z());
  }

  // Number of segments
  M_ = global_wps_.size() - 1;  // Number of segments is one less than the number of waypoints

  if (initial_yaws.size() != global_wps_.size() ||
      !std::all_of(initial_yaws.begin(), initial_yaws.end(),
                   [](double yaw) { return std::isfinite(yaw); })) {
    throw std::invalid_argument(
        "joint trajectory requires one finite yaw for every position waypoint");
  }
  initial_yaws_ = initial_yaws;
  for (std::size_t i = 1; i < initial_yaws_.size(); ++i) {
    initial_yaws_[i] = initial_yaws_[i - 1] +
        std::remainder(initial_yaws_[i] - initial_yaws_[i - 1], 2.0 * M_PI);
  }
  yaw0_ = initial_state.yaw;
  initial_yaws_.front() = yaw0_;
  yawf_ = fix_goal_yaw
      ? initial_yaws_.back() +
            std::remainder(goal_state.yaw - initial_yaws_.back(), 2.0 * M_PI)
      : initial_yaws_.back();
  initial_yaws_.back() = yawf_;
  fix_goal_yaw_ = fix_goal_yaw;
  yaw_variation_penalty_weight_ = time_weight_ / std::max(w_max_, 1.0e-6);
  optimization_safety_intervals_.resize(static_cast<std::size_t>(M_));
  for (int segment = 0; segment < M_; ++segment) {
    optimization_safety_intervals_[static_cast<std::size_t>(segment)] =
        safety_sampling_policy_.intervals(
            (global_wps_[segment + 1] - global_wps_[segment]).norm(),
            std::abs(initial_yaws_[segment + 1] - initial_yaws_[segment]));
  }

  // free control points indices
  K_pos_ = dof_per_knot_ * std::max(0, M_ - 1);
  K_yaw_ = 3 * std::max(0, M_ - 1) + (fix_goal_yaw_ ? 0 : 1);
  K_cp_ = K_pos_ + K_yaw_;
  K_sig_ = M_;           // Number of slack times (K_sig_ = M_)
  K_ = K_cp_ + K_sig_;   // Total number of decision variables
  risk_local_segments_.clear();
  refinement_variable_mask_.clear();

  // Set initial and goal states
  x0_ = initial_state.pos;
  v0_ = initial_state.vel;
  a0_ = initial_state.accel;

  xf_ = global_wps_.back();
  vf_ = goal_state.vel;
  af_ = goal_state.accel;

  if (pos_anchor_weight_ > 0.0) {
    P_anchor_ = global_wps_;
  }

  // Single initial guess: use min-jerk helper
  initial_guess_wps_.clear();
  initial_guess_wps_.push_back(global_wps_);

  list_z0_.clear();
  coupled_derivative_seed_.resize(0);
  coupled_derivative_activation_scales_.clear();
  coupled_derivative_seed_pending_ = false;
  derivative_seed_positions_.clear();
  derivative_seed_target_velocities_.clear();
  derivative_seed_target_accelerations_.clear();
  derivative_seed_target_yaw_knots_.clear();
  derivative_seed_durations_.clear();

  // Allocate containers
  Eigen::VectorXd z0;
  std::vector<Vec3> P, V, A;
  std::vector<double> T;

  // The common quintic smoothstep seed must satisfy both translational and
  // yaw limits before the first feasible L-BFGS evaluation.
  constexpr double kSmoothstepMaxRate = 1.875;
  constexpr double kSmoothstepMaxAccel = 5.773502691896258;
  constexpr double kSmoothstepMaxJerk = 60.0;
  constexpr double kInitialMargin = 1.01;
  for (int i = 0; i < M_; ++i) {
    double dist = (global_wps_[i + 1] - global_wps_[i]).norm();
    const double yaw_travel = std::abs(initial_yaws_[i + 1] - initial_yaws_[i]);
    double t = std::max({
        dc_,
        kSmoothstepMaxRate * dist / V_max_,
        std::sqrt(kSmoothstepMaxAccel * dist / A_max_),
        std::cbrt(kSmoothstepMaxJerk * dist / J_max_),
        kSmoothstepMaxRate * yaw_travel / w_max_,
        std::sqrt(kSmoothstepMaxAccel * yaw_travel / yaw_acc_max_)});
    t *= kInitialMargin;
    T.push_back(t);
  }

  if (topp_init_) {
    mighty::ToppLimits limits;
    limits.v_max = V_max_;
    limits.a_max = A_max_;
    limits.j_max = J_max_;
    limits.w_max = w_max_;
    limits.yaw_acc_max = yaw_acc_max_;
    limits.min_duration = dc_;
    mighty::ToppProfile profile;
    if (mighty::computeToppSeed(global_wps_, initial_yaws_, v0_, a0_, vf_, af_,
                                limits, &profile) &&
        profile.valid &&
        static_cast<int>(profile.durations.size()) == M_) {
      // Adopt TOPP durations only when they are sane: a degenerate profile
      // (e.g. a rest-to-rest single segment) must not replace the smoothstep
      // lower bound with an absurd T, which would freeze direct delivery.
      bool sane = true;
      for (double t : profile.durations) {
        if (!std::isfinite(t) || t <= 0.0 || t > 60.0) {
          sane = false;
          break;
        }
      }
      if (sane) {
        T = profile.durations;
      }
    }
  }

  for (int i = 0; i < M_ + 1; ++i) {
    P.push_back(global_wps_[i]);
  }

  // P_anchor_ follows the collision-certified frontend waypoints.

  std::vector<mighty::YawKnot> yaw_knots(M_ + 1);
  for (int i = 0; i <= M_; ++i) yaw_knots[i].yaw = initial_yaws_[i];

  // Start from the certified rest-to-rest pose edges, then admit as much of
  // the global minimum-jerk derivative field as each local seam can safely
  // support.  This keeps narrow seams from forcing every open-space seam to
  // stop while preserving a feasible warm start for strict L-BFGS.
  auto start = std::chrono::high_resolution_clock::now();
  V.assign(M_ + 1, Vec3::Zero());
  A.assign(M_ + 1, Vec3::Zero());
  V.front() = v0_;
  A.front() = a0_;
  V.back() = vf_;
  A.back() = af_;

  std::vector<Vec3> target_V;
  std::vector<Vec3> target_A;
  if (M_ == 1) {
    target_V = {v0_, vf_};
    target_A = {a0_, af_};
  } else {
    solveMinJerkVelAcc(global_wps_, T, v0_, a0_, vf_, af_, target_V,
                       target_A);
  }

  std::vector<Vec3> yaw_positions(M_ + 1, Vec3::Zero());
  for (int i = 0; i <= M_; ++i) yaw_positions[i].x() = initial_yaws_[i];
  std::vector<Vec3> target_yaw_rates(M_ + 1, Vec3::Zero());
  std::vector<Vec3> target_yaw_accelerations(M_ + 1, Vec3::Zero());
  if (M_ > 1) {
    solveMinJerkVelAcc(yaw_positions, T, Vec3::Zero(), Vec3::Zero(),
                       Vec3::Zero(), Vec3::Zero(), target_yaw_rates,
                       target_yaw_accelerations);
  }

  derivative_activation_scales_.assign(M_ + 1, 0.0);
  derivative_activation_scales_.front() = 1.0;
  derivative_activation_scales_.back() = 1.0;
  std::vector<double> yaw_activation_scales(M_ + 1, 0.0);
  yaw_activation_scales.front() = 1.0;
  yaw_activation_scales.back() = 1.0;
  // The whole-body certified seed keeps yaw derivatives at zero. This makes
  // each frontend edge a monotone rest-to-rest yaw transition, so the seed
  // cannot introduce a winding absent from the collision-certified path.
  // Solver-only uses retain the unconstrained minimum-jerk yaw seed.
  // When the regular ladder only admits 6.25% or rest, probe smaller nonzero
  // derivatives and refine the adjacent rejected interval.  No monotonic
  // safety assumption is used.
  constexpr std::array<double, 6> kActivationLadder{
      1.0, 0.75, 0.5, 0.25, 0.125, 0.0625};
  constexpr std::array<double, 3> kLightActivationLadder{1.0, 0.5, 0.25};
  constexpr std::array<double, 4> kLowActivationTail{
      0.03125, 0.015625, 0.0078125, 0.00390625};
  constexpr int kActivationRefinementSamples = 8;
  const auto maximize_scale = [&](double initial_scale, auto&& try_scale) {
    double best = initial_scale;
    double rejected_upper = 1.0;
    if (light_activation_ladder_) {
      for (const double scale : kLightActivationLadder) {
        if (scale <= best) continue;
        if (try_scale(scale)) return scale;
        rejected_upper = scale;
      }
      return best;
    }
    for (const double scale : kActivationLadder) {
      if (scale <= best) continue;
      if (try_scale(scale)) {
        best = scale;
        break;
      }
      rejected_upper = scale;
    }
    if (best > kActivationLadder.back()) return best;
    if (best == 0.0) {
      for (const double scale : kLowActivationTail) {
        if (try_scale(scale)) {
          best = scale;
          break;
        }
        rejected_upper = scale;
      }
    }
    if (best == 0.0 || rejected_upper <= best) return best;

    const double interval = rejected_upper - best;
    for (int sample = kActivationRefinementSamples - 1; sample >= 1;
         --sample) {
      const double scale =
          best + interval * static_cast<double>(sample) /
                     static_cast<double>(kActivationRefinementSamples);
      if (try_scale(scale)) return scale;
    }
    return best;
  };
  const auto activate_knot = [&](int knot) {
    // Most seams admit the unconstrained minimum-jerk field.  Certify that
    // common case once instead of issuing separate translation and yaw
    // queries; only constrained seams pay for the coordinate-wise search.
    if (derivative_activation_scales_[knot] == 0.0 &&
        (whole_body_safety_ || yaw_activation_scales[knot] == 0.0)) {
      const Vec3 old_velocity = V[knot];
      const Vec3 old_acceleration = A[knot];
      const mighty::YawKnot old_yaw = yaw_knots[knot];
      V[knot] = target_V[knot];
      A[knot] = target_A[knot];
      if (!whole_body_safety_) {
        yaw_knots[knot].rate = target_yaw_rates[knot].x();
        yaw_knots[knot].accel = target_yaw_accelerations[knot].x();
      }
      if (!whole_body_safety_ || adjacentSegmentsWholeBodySafe(
              P, V, A, yaw_knots, T, knot)) {
        derivative_activation_scales_[knot] = 1.0;
        if (!whole_body_safety_) yaw_activation_scales[knot] = 1.0;
        return;
      }
      V[knot] = old_velocity;
      A[knot] = old_acceleration;
      yaw_knots[knot] = old_yaw;
    }

    const auto try_translation_scale = [&](double scale) {
      const Vec3 old_velocity = V[knot];
      const Vec3 old_acceleration = A[knot];
      V[knot] = scale * target_V[knot];
      A[knot] = scale * target_A[knot];
      if (!whole_body_safety_ || adjacentSegmentsWholeBodySafe(
              P, V, A, yaw_knots, T, knot)) {
        derivative_activation_scales_[knot] = scale;
        return true;
      }
      V[knot] = old_velocity;
      A[knot] = old_acceleration;
      return false;
    };
    derivative_activation_scales_[knot] = maximize_scale(
        derivative_activation_scales_[knot], try_translation_scale);

    if (!whole_body_safety_) {
      const auto try_yaw_scale = [&](double scale) {
        yaw_knots[knot].rate = scale * target_yaw_rates[knot].x();
        yaw_knots[knot].accel = scale * target_yaw_accelerations[knot].x();
        yaw_activation_scales[knot] = scale;
        return true;
      };
      yaw_activation_scales[knot] = maximize_scale(
          yaw_activation_scales[knot], try_yaw_scale);
    }

  };
  for (int knot = 1; knot < M_; ++knot) activate_knot(knot);
  for (int knot = M_ - 1; knot >= 1; --knot) activate_knot(knot);

  // Scaling a single minimum-jerk derivative ray is insufficient near an
  // obstacle: if that direction clips the body, its only feasible point may
  // be almost zero. Search a compact, geometry-derived direction set and
  // decouple velocity from acceleration. Every accepted alternative is
  // certified on both adjacent segments, so this expands the seed space
  // without weakening safety or adding shape-specific rules.
  if (whole_body_safety_ && minimum_forward_speed_ > 0.0) {
    const auto improve_restricted_knot = [&](int knot) {
      if (V[knot].norm() + 1.0e-6 >= minimum_forward_speed_) return;
      const Vec3 incoming = global_wps_[knot] - global_wps_[knot - 1];
      const Vec3 outgoing = global_wps_[knot + 1] - global_wps_[knot];
      const Vec3 incoming_unit = incoming.norm() > 1.0e-9
          ? incoming.normalized() : Vec3::Zero();
      const Vec3 outgoing_unit = outgoing.norm() > 1.0e-9
          ? outgoing.normalized() : Vec3::Zero();
      std::array<Vec3, 5> directions{
          incoming_unit + outgoing_unit,
          global_wps_[knot + 1] - global_wps_[knot - 1],
          outgoing, incoming, target_V[knot]};
      // Start high because the subsequent dynamics retime scales every knot
      // velocity down together. The certified geometry check rejects any
      // overshooting candidate before it can become a warm start.
      constexpr std::array<double, 8> kSpeedMargins{
          8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.25, 1.0};
      for (Vec3 direction : directions) {
        if (direction.norm() <= 1.0e-9) continue;
        direction.normalize();
        for (const double margin : kSpeedMargins) {
          const Vec3 old_velocity = V[knot];
          const Vec3 old_acceleration = A[knot];
          V[knot] = std::min(V_max_, margin * minimum_forward_speed_) *
                    direction;
          A[knot].setZero();
          if (adjacentSegmentsWholeBodySafe(
                  P, V, A, yaw_knots, T, knot)) {
            derivative_activation_scales_[knot] =
                target_V[knot].norm() > 1.0e-9
                    ? V[knot].norm() / target_V[knot].norm()
                    : 1.0;
            return;
          }
          V[knot] = old_velocity;
          A[knot] = old_acceleration;
        }
      }
    };
    for (int knot = 1; knot < M_; ++knot) improve_restricted_knot(knot);
    for (int knot = M_ - 1; knot >= 1; --knot) improve_restricted_knot(knot);
  }
  bool derivative_modes_differ = false;
  for (int knot = 1; knot < M_; ++knot) {
    derivative_modes_differ = derivative_modes_differ ||
        std::abs(derivative_activation_scales_[knot] -
                 yaw_activation_scales[knot]) > 1.0e-12;
  }
  if (M_ > 1 && derivative_modes_differ) {
    coupled_derivative_seed_pending_ = true;
    derivative_seed_positions_ = P;
    derivative_seed_target_velocities_ = target_V;
    derivative_seed_target_accelerations_ = target_A;
    derivative_seed_durations_ = T;
    derivative_seed_target_yaw_knots_.resize(M_ + 1);
    for (int knot = 0; knot <= M_; ++knot) {
      auto& target = derivative_seed_target_yaw_knots_[knot];
      target.yaw = initial_yaws_[knot];
      target.rate = target_yaw_rates[knot].x();
      target.accel = target_yaw_accelerations[knot].x();
    }
  }

  // A single segment has no interior derivative to activate.
  if (M_ == 1) {
    V = target_V;
    A = target_A;
  }
  auto end = std::chrono::high_resolution_clock::now();
  // Measure time in microseconds and convert to milliseconds
  initial_guess_computation_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

  // Generate the initial guess in z form
  packDecisionVariables(P, V, A, yaw_knots, T, z0);

  // Store the results
  list_z0_.push_back(z0);

  // Sanity check
  sanityCheck();

  // Check cost objective and gradient
  // VecXd g_test1(z0.size());
  // double f1 = evaluateObjectiveAndGradientFused(z0, g_test1);
  // VecXd g_test2(z0.size());
  // double f2 = evaluateObjectiveAndGradient(z0, g_test2);

  // // Check the difference
  // double err_g = (g_test1 - g_test2).norm();
  // double err_f = std::abs(f1 - f2);

  // if (err_g > 1e-5 || err_f > 1e-5)
  // {
  //     // print in red
  //     std::cout << "\033[1;31m |f1-f2|=" << err_f << " ||g1-g2||=" << err_g << "\033[0m\n";
  // }

  // wherever you have a valid z (e.g., right before starting L-BFGS)
  // checkGradDirectional(z0, /*num_dirs=*/8, /*eps=*/1e-5, /*seed=*/42);
  // checkGradCoordinates(z0, /*max_coords=*/256, /*eps=*/1e-5, /*seed=*/43);
}

bool SolverLBFGS::adjacentSegmentsWholeBodySafe(
    const std::vector<Vec3>& positions,
    const std::vector<Vec3>& velocities,
    const std::vector<Vec3>& accelerations,
    const std::vector<mighty::YawKnot>& yaw_knots,
    const std::vector<double>& durations, int knot) {
  if (!whole_body_safety_) return true;
  if (!whole_body_safety_context_) return false;
  // Candidate decisions must be certified independently.  Reusing a warm
  // start across different derivative scales can couple the selected seed to
  // the order in which ladder values were tried.
  auto candidate_context = whole_body_safety_->createContext();
  Eigen::VectorXd candidate;
  packDecisionVariables(positions, velocities, accelerations, yaw_knots,
                        durations, candidate);
  reconstructPVATCPopt(candidate);
  const int first_segment = std::max(0, knot - 1);
  const int last_segment = std::min(M_ - 1, knot);
  for (int segment = first_segment; segment <= last_segment; ++segment) {
    double path_length = 0.0;
    double yaw_travel = 0.0;
    Vec3 previous_position = evalStateDeriv(segment, 0.0).pos;
    double previous_yaw = evalYawState(segment, 0.0).yaw;
    for (int probe = 1; probe <= 8; ++probe) {
      const double tau = static_cast<double>(probe) / 8.0;
      const Vec3 position = evalStateDeriv(segment, tau).pos;
      const double yaw = evalYawState(segment, tau).yaw;
      path_length += (position - previous_position).norm();
      yaw_travel += std::abs(yaw - previous_yaw);
      previous_position = position;
      previous_yaw = yaw;
    }

    const int intervals = safety_sampling_policy_.intervals(
        path_length, yaw_travel);
    for (int sample = 0; sample <= intervals; ++sample) {
      const double tau = static_cast<double>(sample) / intervals;
      const Vec3 position = evalStateDeriv(segment, tau).pos;
      const double yaw = evalYawState(segment, tau).yaw;
      const auto safety = whole_body_safety_->evaluate(
          position, yaw, candidate_context, true);
      if (!safety.success ||
          safety.statistics.minimum_beta < whole_body_safety_->acceptanceBeta()) {
        whole_body_safety_context_->cumulative_statistics.merge(
            candidate_context.cumulative_statistics);
        return false;
      }
    }
  }
  whole_body_safety_context_->cumulative_statistics.merge(
      candidate_context.cumulative_statistics);
  return true;
}

// -----------------------------------------------------------------------------

void SolverLBFGS::getGlobalPath(vec_Vecf<3>& global_path) {
  global_path.clear();
  global_path.reserve(global_wps_.size());
  for (auto const& wp : global_wps_) {
    // wp is Eigen::Vector3d; Vec3f is Eigen::Matrix<double,3,1>
    global_path.emplace_back(wp.x(), wp.y(), wp.z());
  }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::packDecisionVariables(const std::vector<Vec3>& P, const std::vector<Vec3>& V,
                                        const std::vector<Vec3>& A,
                                        const std::vector<mighty::YawKnot>& yaw_knots,
                                        const std::vector<double>& T,
                                        VecXd& z) const {
  // CP block (interior only) + time block (all segments)
  z.resize(K_);

  // per-knot T̄ for v̂, â scaling
  std::vector<double> Tbar(M_ + 1);
  build_Tbar(T, Tbar);

  // ---- pack interior knots only: i = 1..M_-1 ----
  for (int i = 1; i < M_; ++i) {
    const int base = zcp_offset_for_knot(i, M_, dof_per_knot_);
    const double Tb = Tbar[i];
    const double Tb2 = Tb * Tb;

    // positions
    z(base + 0) = P[i].x();
    z(base + 1) = P[i].y();
    z(base + 2) = P[i].z();

    // v̂ = T̄ * V
    const Vec3 vhat = Tb * V[i];
    z(base + 3) = vhat.x();
    z(base + 4) = vhat.y();
    z(base + 5) = vhat.z();

    // â = T̄^2 * A (quintic only)
    if (degree_ == 5) {
      const Vec3 ahat = Tb2 * A[i];
      z(base + 6) = ahat.x();
      z(base + 7) = ahat.y();
      z(base + 8) = ahat.z();
    }
  }

  // ---- pack yaw, yaw-rate and yaw-acceleration at interior knots ----
  for (int i = 1; i < M_; ++i) {
    const int base = K_pos_ + 3 * (i - 1);
    const double Tb = Tbar[i];
    z[base] = yaw_knots[i].yaw;
    z[base + 1] = yaw_knots[i].rate * Tb;
    z[base + 2] = yaw_knots[i].accel * Tb * Tb;
  }
  if (!fix_goal_yaw_) z[K_pos_ + 3 * std::max(0, M_ - 1)] = yawf_;

  // ---- pack τ for time (all segments) ----
  for (int s = 0; s < M_; ++s) z[K_cp_ + s] = T_to_tau(T[s]);
}

// -----------------------------------------------------------------------------

void SolverLBFGS::reconstruct(const VecXd& z, std::vector<Vec3>& P, std::vector<Vec3>& V,
                              std::vector<Vec3>& A, std::vector<std::vector<Vec3>>& CP,
                              std::vector<double>& T) const {
  const int knots = M_ + 1;
  P.resize(knots);
  V.resize(knots);
  A.resize(knots);
  T.resize(M_);
  CP.resize(M_);

  // 1) τ �? T for all segments
  for (int s = 0; s < M_; ++s) T[s] = tau_to_T(z[K_cp_ + s]);

  // 2) T̄ for v̂/â decode of interior knots
  std::vector<double> Tbar(knots);
  build_Tbar(T, Tbar);

  // 3) set endpoints from boundary conditions
  P[0] = x0_;
  V[0] = v0_;
  A[0] = (degree_ == 5) ? a0_ : Vec3::Zero();

  P[M_] = xf_;
  V[M_] = vf_;
  A[M_] = (degree_ == 5) ? af_ : Vec3::Zero();

  // 4) decode interior knots from z
  for (int i = 1; i < M_; ++i) {
    const int base = zcp_offset_for_knot(i, M_, dof_per_knot_);
    const double Tb = std::max(1e-12, Tbar[i]);

    // position
    P[i] = Vec3(z[base + 0], z[base + 1], z[base + 2]);

    // v̂ �? V
    const Vec3 vhat(z[base + 3], z[base + 4], z[base + 5]);
    V[i] = vhat / Tb;

    // â �? A (quintic only; cubic has no acceleration DOF)
    if (degree_ == 5) {
      const Vec3 ahat(z[base + 6], z[base + 7], z[base + 8]);
      A[i] = ahat / (Tb * Tb);
    } else {
      A[i] = Vec3::Zero();
    }

  }

  // 5) compute Bernstein control points from Hermite data
  for (int s = 0; s < M_; ++s) {
    const Vec3 &p0 = P[s], &v0s = V[s], &a0s = A[s];
    const Vec3 &p1 = P[s + 1], &v1s = V[s + 1], &a1s = A[s + 1];
    const double Ts = T[s];
    auto& c = CP[s];

    if (degree_ == 3) {
      // Cubic Hermite �? 4 Bernstein CPs
      c.resize(4);
      c[0] = p0;
      c[1] = p0 + (Ts / 3.0) * v0s;
      c[2] = p1 - (Ts / 3.0) * v1s;
      c[3] = p1;
    } else {
      // Quintic Hermite �? 6 Bernstein CPs
      const double T2 = Ts * Ts;
      c.resize(6);
      c[0] = p0;
      c[1] = p0 + (Ts / 5.0) * v0s;
      c[2] = p0 + (2.0 * Ts / 5.0) * v0s + (T2 / 20.0) * a0s;
      c[3] = p1 - (2.0 * Ts / 5.0) * v1s + (T2 / 20.0) * a1s;
      c[4] = p1 - (Ts / 5.0) * v1s;
      c[5] = p1;
    }
  }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::reconstructYaw(
    const Eigen::VectorXd& z, const std::vector<double>& segment_times,
    std::vector<mighty::YawKnot>& knots,
    std::vector<std::array<double, 6>>& control_points) const {
  knots.assign(M_ + 1, {});
  knots.front() = {yaw0_, 0.0, 0.0};
  knots.back() = {
      fix_goal_yaw_ ? yawf_ : z[K_pos_ + 3 * std::max(0, M_ - 1)],
      0.0, 0.0};

  std::vector<double> Tbar;
  build_Tbar(segment_times, Tbar);
  for (int i = 1; i < M_; ++i) {
    const int base = K_pos_ + 3 * (i - 1);
    const double Tb = std::max(1.0e-12, Tbar[i]);
    knots[i] = {z[base], z[base + 1] / Tb,
                z[base + 2] / (Tb * Tb)};
  }

  control_points.resize(M_);
  for (int segment = 0; segment < M_; ++segment) {
    control_points[segment] = yaw_control_points(
        knots[segment], knots[segment + 1], segment_times[segment]);
  }
}

//  Closed-form quintic control‐points for one segment
std::vector<Vec3> SolverLBFGS::computeQuinticCP(const Vec3& P0, const Vec3& V0, const Vec3& A0,
                                                  const Vec3& P1, const Vec3& V1, const Vec3& A1,
                                                  double T) {
  double T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
  Vec3 c0 = P0;
  Vec3 c1 = V0;
  Vec3 c2 = 0.5 * A0;
  Vec3 c3 =
      (-20.0 * P0 + 20.0 * P1 - (8.0 * V1 + 12.0 * V0) * T - (3.0 * A0 - A1) * T2) / (2.0 * T3);
  Vec3 c4 = (30.0 * P0 - 30.0 * P1 + (14.0 * V1 + 16.0 * V0) * T + (3.0 * A0 - 2.0 * A1) * T2) /
            (2.0 * T4);
  Vec3 c5 = (-12.0 * P0 + 12.0 * P1 - (6.0 * V1 + 6.0 * V0) * T - (A0 - A1) * T2) / (2.0 * T5);

  std::vector<Vec3> CP(6);
  for (int i = 0; i < 6; ++i) {
    double t = T * (double(i) / 5.0);
    double t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
    CP[i] = c0 + c1 * t + c2 * t2 + c3 * t3 + c4 * t4 + c5 * t5;
  }
  return CP;
}

// -----------------------------------------------------------------------------

inline Eigen::Matrix4d SolverLBFGS::K_r(double T) {
  double t2 = T * T, t3 = t2 * T;
  double i2 = 1.0 / t2, i3 = 1.0 / t3;
  Eigen::Matrix4d K;
  K << 192 * i3, 36 * i2, 168 * i3, -24 * i2, 36 * i2, 9.0 / T, 24 * i2, -3.0 / T, 168 * i3,
      24 * i2, 192 * i3, -36 * i2, -24 * i2, -3.0 / T, -36 * i2, 9.0 / T;
  return K;
}

// -----------------------------------------------------------------------------

inline std::array<double, 4> SolverLBFGS::k_r(double T, double P0, double P1) {
  // Example placeholder �? you must substitute your real formulas:
  double k0 = (-66960 * P0 * T + 66960 * P1 * T) / (2 * T * T * T * T * T) +
              (-24480 * P0 * T + 24480 * P1 * T) / (2 * T * T * T * T * T) +
              (4320 * P0 * T - 4320 * P1 * T) / (2 * T * T * T * T * T) +
              (25920 * P0 * T - 25920 * P1 * T) / (2 * T * T * T * T * T) +
              (61920 * P0 * T - 61920 * P1 * T) / (2 * T * T * T * T * T);

  double k1 = (-11880 * P0 * T * T + 11880 * P1 * T * T) / (2 * T * T * T * T * T) +
              (-5400 * P0 * T * T + 5400 * P1 * T * T) / (2 * T * T * T * T * T) +
              (1080 * P0 * T * T - 1080 * P1 * T * T) / (2 * T * T * T * T * T) +
              (4320 * P0 * T * T - 4320 * P1 * T * T) / (2 * T * T * T * T * T) +
              (12000 * P0 * T * T - 12000 * P1 * T * T) / (2 * T * T * T * T * T);

  double k2 = (-62640 * P0 * T + 62640 * P1 * T) / (2 * T * T * T * T * T) +
              (-18720 * P0 * T + 18720 * P1 * T) / (2 * T * T * T * T * T) +
              (2880 * P0 * T - 2880 * P1 * T) / (2 * T * T * T * T * T) +
              (25920 * P0 * T - 25920 * P1 * T) / (2 * T * T * T * T * T) +
              (53280 * P0 * T - 53280 * P1 * T) / (2 * T * T * T * T * T);

  double k3 = (-7680 * P0 * T * T + 7680 * P1 * T * T) / (2 * T * T * T * T * T) +
              (-4320 * P0 * T * T + 4320 * P1 * T * T) / (2 * T * T * T * T * T) +
              (-360 * P0 * T * T + 360 * P1 * T * T) / (2 * T * T * T * T * T) +
              (2520 * P0 * T * T - 2520 * P1 * T * T) / (2 * T * T * T * T * T) +
              (9720 * P0 * T * T - 9720 * P1 * T * T) / (2 * T * T * T * T * T);
  return {k0, k1, k2, k3};
}

// -----------------------------------------------------------------------------

void SolverLBFGS::assemble_H_b(const std::vector<Vec3>& wps, const std::vector<double>& T,
                               const Vec3& v0, const Vec3& a0, const Vec3& vf, const Vec3& af,
                               Eigen::MatrixXd& H, Eigen::MatrixXd& b) {
  int M = (int)T.size();
  int N = M - 1;  // # interior junctions
  int D = 3;      // 3d

  H = Eigen::MatrixXd::Zero(2 * N, 2 * N);
  b = Eigen::MatrixXd::Zero(2 * N, D);

  // for each segment r = 0..M-1
  for (int r = 0; r < M; ++r) {
    // 1) Hessian block
    Eigen::Matrix4d Kr = K_r(T[r]);

    // 2) constant-term block (4×3)
    Eigen::Matrix<double, 4, 3> kr;
    for (int dim = 0; dim < 3; ++dim) {
      auto c = k_r(T[r], wps[r](dim), wps[r + 1](dim));
      for (int i = 0; i < 4; ++i) kr(i, dim) = c[i];
    }

    // 3) left half �? junction r-1
    if (r >= 1) {
      int i = r - 1;
      H.block<2, 2>(2 * i, 2 * i) += Kr.block<2, 2>(0, 0);
      b.block<2, 3>(2 * i, 0) += kr.block<2, 3>(0, 0);
    }

    // 4) right half �? junction r
    if (r <= M - 2) {
      int j = r;
      H.block<2, 2>(2 * j, 2 * j) += Kr.block<2, 2>(2, 2);
      b.block<2, 3>(2 * j, 0) += kr.block<2, 3>(2, 0);
    }

    // 5) coupling off-diagonals (H only)
    if (r >= 1 && r <= M - 2) {
      int i = r - 1, j = r;
      H.block<2, 2>(2 * i, 2 * j) += Kr.block<2, 2>(0, 2);
      H.block<2, 2>(2 * j, 2 * i) += Kr.block<2, 2>(2, 0);
    }
  }

  // 6) endpoint cross-terms into b
  {
    // segment 0: x0 (known) �? x1 interior
    Eigen::Matrix4d K0 = K_r(T[0]);
    Eigen::Matrix<double, 2, 3> v0a0;
    v0a0.row(0) = v0.transpose();
    v0a0.row(1) = a0.transpose();
    // b[0:2] += K0[2:4,0:2]*[v0;a0]
    b.block<2, 3>(0, 0) += K0.block<2, 2>(2, 0) * v0a0;
  }
  {
    // segment M-1: x_{M} known �? x_{M-1} interior
    Eigen::Matrix4d KM = K_r(T.back());
    Eigen::Matrix<double, 2, 3> vfaf;
    vfaf.row(0) = vf.transpose();
    vfaf.row(1) = af.transpose();
    int i = 2 * (M - 2);
    // b[i:i+2] += KM[0:2,2:4]*[vf;af]
    b.block<2, 3>(i, 0) += KM.block<2, 2>(0, 2) * vfaf;
  }
}

// -----------------------------------------------------------------------------

//  Build and solve the banded min-jerk system:
void SolverLBFGS::solveMinJerkVelAcc(const std::vector<Vec3>& wps,  // size M+1
                                     const std::vector<double>& T,  // size M
                                     const Vec3& v0, const Vec3& a0, const Vec3& vf, const Vec3& af,
                                     std::vector<Vec3>& V,  // out size M+1
                                     std::vector<Vec3>& A   // out size M+1
) {
  int M = (int)T.size();

  Eigen::MatrixXd H, b;
  assemble_H_b(wps, T, v0, a0, vf, af, H, b);

  // Solve H x = -b   �?   x is (2N×3)
  Eigen::MatrixXd X = H.ldlt().solve(-b);

  // Unpack into V,A
  V.clear();
  A.clear();
  V.resize(M + 1);
  A.resize(M + 1);
  V[0] = v0;
  A[0] = a0;
  V[M] = vf;
  A[M] = af;
  for (int i = 1; i < M; ++i) {
    // row 2*(i-1) of X is the velocity at junction i
    V[i] = X.row(2 * (i - 1)).transpose();  // Vec3 = (3×1)
    // row 2*(i-1)+1 of X is the acceleration at junction i
    A[i] = X.row(2 * (i - 1) + 1).transpose();
  }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::getGoalSetpoints(std::vector<state>& goal_setpoints,
                                   bool include_boundary_samples) {
  // 1) total time & sample count (clamped: a degenerate T must not overflow
  // the int cast or blow up the setpoint buffer)
  double total_time = std::accumulate(T_opt_.begin(), T_opt_.end(), 0.0);
  const long long n64 =
      include_boundary_samples
          ? static_cast<long long>(std::ceil(total_time / dc_)) + 1
          : static_cast<long long>(total_time / dc_);
  const int N = static_cast<int>(std::clamp<long long>(n64, 2, 100000));

  // 2) resize output
  goal_setpoints.resize(N);

  // 3) precompute timestamps
  std::vector<double> timestamps(N);
  for (int i = 0; i < N; ++i) {
    timestamps[i] =
        include_boundary_samples
            ? std::min(static_cast<double>(i) * dc_, total_time)
            : static_cast<double>(i + 1) * dc_;
  }

  // 4) cumulative segment‑end times
  int M = static_cast<int>(T_opt_.size());
  std::vector<double> ends(M);
  if (M > 0) {
    ends[0] = T_opt_[0];
    for (int s = 1; s < M; ++s) {
      ends[s] = ends[s - 1] + T_opt_[s];
    }
  }

// 5) parallel fill of goal_setpoints
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    double ti = timestamps[i];

    // find which segment we're in
    auto it = std::upper_bound(ends.begin(), ends.end(), ti);
    int seg = static_cast<int>(it - ends.begin());
    seg = std::min(seg, std::max(0, M - 1));

    double t0 = (seg > 0 ? ends[seg - 1] : 0.0);
    double tau = (T_opt_[seg] > 0.0) ? (ti - t0) / T_opt_[seg] : 0.0;

    StateDeriv D = evalStateDeriv(seg, tau);

    state st;
    st.setTimeStamp(t0_ + ti);
    st.setPos(D.pos.x(), D.pos.y(), D.pos.z());
    st.setVel(D.vel.x(), D.vel.y(), D.vel.z());
    st.setAccel(D.accel.x(), D.accel.y(), D.accel.z());
    st.setJerk(D.jerk.x(), D.jerk.y(), D.jerk.z());
    const mighty::YawSample yaw = evalYawState(seg, tau);
    st.yaw = yaw.yaw;
    st.dyaw = yaw.rate;
    st.use_tracking_yaw = true;
    goal_setpoints[i] = st;
  }

  // 6) Preserve the exact optimized endpoint and enforce the requested
  // terminal rest condition.
  {
    auto& last = goal_setpoints.back();
    // BUG #2: .transpose() is unnecessary if vel/accel are Vector3d
    last.vel = Eigen::Vector3d::Zero();
    last.accel = Eigen::Vector3d::Zero();
    last.jerk = Eigen::Vector3d::Zero();
  }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::getPieceWisePol(PieceWiseQuinticPol& pwp) {
  // 1) reset
  pwp.clear();

  // 2) build the time‐breaks (absolute times)
  pwp.times.push_back(t0_);
  double t_acc = 0.0;
  for (int i = 0; i < M_; ++i) {
    t_acc += T_opt_[i];
    pwp.times.push_back(t0_ + t_acc);
  }

  // 3) for each segment, convert Hermite �? power basis (zero-padded to quintic format)
  for (int s = 0; s < M_; ++s) {
    double T = T_opt_[s];
    const auto &P0 = P_opt_[s], &P1 = P_opt_[s + 1];
    const auto &V0 = V_opt_[s], &V1 = V_opt_[s + 1];

    Eigen::Matrix<double, 6, 1> cx, cy, cz;

    if (degree_ == 3) {
      // Cubic: p(u) = c3*u³ + c2*u² + c1*u + c0, stored as [0, 0, c3, c2, c1, c0]
      double T2 = T * T, T3 = T2 * T;
      auto cubic_coeffs = [&](int axis) -> Eigen::Matrix<double, 6, 1> {
        double p0 = (axis == 0) ? P0.x() : (axis == 1) ? P0.y() : P0.z();
        double p1 = (axis == 0) ? P1.x() : (axis == 1) ? P1.y() : P1.z();
        double v0 = (axis == 0) ? V0.x() : (axis == 1) ? V0.y() : V0.z();
        double v1 = (axis == 0) ? V1.x() : (axis == 1) ? V1.y() : V1.z();
        double c0 = p0;
        double c1 = v0;
        double c2 = (3.0 * (p1 - p0) - (2.0 * v0 + v1) * T) / T2;
        double c3 = (2.0 * (p0 - p1) + (v0 + v1) * T) / T3;
        Eigen::Matrix<double, 6, 1> c;
        c << 0.0, 0.0, c3, c2, c1, c0;
        return c;
      };
      cx = cubic_coeffs(0);
      cy = cubic_coeffs(1);
      cz = cubic_coeffs(2);
    } else {
      // Quintic: p(u) = a*u^5 + b*u^4 + c*u^3 + d*u^2 + e*u + f
      const auto &A0 = A_opt_[s], &A1 = A_opt_[s + 1];
      double T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
      Eigen::Vector3d C0 = P1 - (P0 + V0 * T + 0.5 * A0 * T2);
      Eigen::Vector3d C1 = V1 - (V0 + A0 * T);
      Eigen::Vector3d C2 = A1 - A0;

      auto quintic_coeffs = [&](int axis) -> Eigen::Matrix<double, 6, 1> {
        double f = (axis == 0) ? P0.x() : (axis == 1) ? P0.y() : P0.z();
        double e = (axis == 0) ? V0.x() : (axis == 1) ? V0.y() : V0.z();
        double d = 0.5 * ((axis == 0) ? A0.x() : (axis == 1) ? A0.y() : A0.z());
        double c0v = (axis == 0) ? C0.x() : (axis == 1) ? C0.y() : C0.z();
        double c1v = (axis == 0) ? C1.x() : (axis == 1) ? C1.y() : C1.z();
        double c2v = (axis == 0) ? C2.x() : (axis == 1) ? C2.y() : C2.z();
        double c = (10.0 * c0v - 4.0 * c1v + 0.5 * c2v) / T3;
        double b = (-15.0 * c0v + 7.0 * c1v - c2v) / T4;
        double a = (6.0 * c0v - 3.0 * c1v + 0.5 * c2v) / T5;
        Eigen::Matrix<double, 6, 1> coeff;
        coeff << a, b, c, d, e, f;
        return coeff;
      };
      cx = quintic_coeffs(0);
      cy = quintic_coeffs(1);
      cz = quintic_coeffs(2);
    }

    pwp.coeff_x.push_back(cx);
    pwp.coeff_y.push_back(cy);
    pwp.coeff_z.push_back(cz);
  }
}

// -----------------------------------------------------------------------------

void SolverLBFGS::getControlPoints(std::vector<Eigen::Matrix<double, 3, 6>>& cps) {
  // Resize the control points vector
  cps.resize(M_);

  // Fill the control points
  for (int i = 0; i < M_; i++) {
    Eigen::Matrix<double, 3, 6> cp_i;
    for (int j = 0; j < 6; j++) {
      cp_i(0, j) = CP_opt_[i][j][0];
      cp_i(1, j) = CP_opt_[i][j][1];
      cp_i(2, j) = CP_opt_[i][j][2];
    }
    cps[i] = cp_i;
  }
}

// -----------------------------------------------------------------------------

StateDeriv SolverLBFGS::evalStateDeriv(int s, double tau) const {
  const auto& P0 = P_opt_[s];
  const auto& V0 = V_opt_[s];
  const auto& P1 = P_opt_[s + 1];
  const auto& V1 = V_opt_[s + 1];
  double T = T_opt_[s];

  if (degree_ == 3) {
    // Cubic Hermite basis
    double t2 = tau * tau, t3 = t2 * tau;
    double h0 = 2 * t3 - 3 * t2 + 1;
    double h1 = t3 - 2 * t2 + tau;
    double h2 = -2 * t3 + 3 * t2;
    double h3 = t3 - t2;
    Eigen::Vector3d p = P0 * h0 + V0 * (h1 * T) + P1 * h2 + V1 * (h3 * T);

    double dh0 = 6 * t2 - 6 * tau;
    double dh1 = 3 * t2 - 4 * tau + 1;
    double dh2 = -6 * t2 + 6 * tau;
    double dh3 = 3 * t2 - 2 * tau;
    Eigen::Vector3d v = (P0 * dh0 + V0 * (dh1 * T) + P1 * dh2 + V1 * (dh3 * T)) / T;

    double d2h0 = 12 * tau - 6;
    double d2h1 = 6 * tau - 4;
    double d2h2 = -12 * tau + 6;
    double d2h3 = 6 * tau - 2;
    Eigen::Vector3d a = (P0 * d2h0 + V0 * (d2h1 * T) + P1 * d2h2 + V1 * (d2h3 * T)) / (T * T);

    // Jerk is constant for cubic: d³h/dτ³ = {12, 6T, -12, 6T}
    Eigen::Vector3d j = (P0 * 12.0 + V0 * (6.0 * T) + P1 * (-12.0) + V1 * (6.0 * T)) / (T * T * T);

    return {p, v, a, j};
  } else {
    // Quintic Hermite basis
    const auto& A0 = A_opt_[s];
    const auto& A1 = A_opt_[s + 1];
    double t2 = tau * tau, t3 = t2 * tau, t4 = t3 * tau, t5 = t4 * tau;

    double h0 = 1 - 10 * t3 + 15 * t4 - 6 * t5;
    double h1 = tau - 6 * t3 + 8 * t4 - 3 * t5;
    double h2 = 0.5 * (t2 - 3 * t3 + 3 * t4 - t5);
    double h3 = 10 * t3 - 15 * t4 + 6 * t5;
    double h4 = -4 * t3 + 7 * t4 - 3 * t5;
    double h5 = 0.5 * (t3 - 2 * t4 + t5);
    Eigen::Vector3d p =
        P0 * h0 + V0 * (h1 * T) + A0 * (h2 * T * T) + P1 * h3 + V1 * (h4 * T) + A1 * (h5 * T * T);

    double dh0 = -30 * t2 + 60 * t3 - 30 * t4;
    double dh1 = 1 - 18 * t2 + 32 * t3 - 15 * t4;
    double dh2 = 0.5 * (2 * tau - 9 * t2 + 12 * t3 - 5 * t4);
    double dh3 = 30 * t2 - 60 * t3 + 30 * t4;
    double dh4 = -12 * t2 + 28 * t3 - 15 * t4;
    double dh5 = 0.5 * (3 * t2 - 8 * t3 + 5 * t4);
    Eigen::Vector3d v = (P0 * dh0 + V0 * (dh1 * T) + A0 * (dh2 * T * T) + P1 * dh3 +
                         V1 * (dh4 * T) + A1 * (dh5 * T * T)) / T;

    double d2h0 = -60 * tau + 180 * t2 - 120 * t3;
    double d2h1 = -36 * tau + 96 * t2 - 60 * t3;
    double d2h2 = 0.5 * (2 - 18 * tau + 36 * t2 - 20 * t3);
    double d2h3 = 60 * tau - 180 * t2 + 120 * t3;
    double d2h4 = -24 * tau + 84 * t2 - 60 * t3;
    double d2h5 = 0.5 * (6 * tau - 24 * t2 + 20 * t3);
    Eigen::Vector3d a = (P0 * d2h0 + V0 * (d2h1 * T) + A0 * (d2h2 * T * T) + P1 * d2h3 +
                         V1 * (d2h4 * T) + A1 * (d2h5 * T * T)) / (T * T);

    double d3h0 = -60 + 360 * tau - 360 * t2;
    double d3h1 = -36 + 192 * tau - 180 * t2;
    double d3h2 = 0.5 * (-18 + 72 * tau - 60 * t2);
    double d3h3 = 60 - 360 * tau + 360 * t2;
    double d3h4 = -24 + 168 * tau - 180 * t2;
    double d3h5 = 0.5 * (6 - 48 * tau + 60 * t2);
    Eigen::Vector3d j = (P0 * d3h0 + V0 * (d3h1 * T) + A0 * (d3h2 * T * T) + P1 * d3h3 +
                         V1 * (d3h4 * T) + A1 * (d3h5 * T * T)) / (T * T * T);

    return {p, v, a, j};
  }
}

// -----------------------------------------------------------------------------

// degree-5 Bézier: p(u) = Σ_{j=0..5} B5_j(u) * CP[j]
inline Eigen::Vector3d evalBezierQuintic(const std::array<Eigen::Vector3d, 6>& cp, double u) {
  u = std::min(std::max(u, 0.0), 1.0);
  const double omu = 1.0 - u;
  const double omu2 = omu * omu;
  const double omu3 = omu2 * omu;
  const double omu4 = omu3 * omu;
  const double omu5 = omu4 * omu;
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double u4 = u3 * u;
  const double u5 = u4 * u;

  const double b0 = omu5;
  const double b1 = 5.0 * u * omu4;
  const double b2 = 10.0 * u2 * omu3;
  const double b3 = 10.0 * u3 * omu2;
  const double b4 = 5.0 * u4 * omu;
  const double b5 = u5;

  return b0 * cp[0] + b1 * cp[1] + b2 * cp[2] + b3 * cp[3] + b4 * cp[4] + b5 * cp[5];
}

// Sample robot positions at absolute times t_i = t0 + i*dt, i=1..N
// Uses a single pass over segments (monotone times) for speed.
inline void sampleRobotPositionsUniform(const std::vector<std::array<Eigen::Vector3d, 6>>& CP,
                                        const std::vector<double>& T, double t0, int N,
                                        std::vector<double>& t_samples,
                                        std::vector<Eigen::Vector3d>& p_samples) {
  const int M = static_cast<int>(T.size());
  t_samples.resize(N);
  p_samples.resize(N);

  // absolute edges: [t0, t0+T0, t0+T0+T1, ...]
  std::vector<double> edges(M + 1, t0);
  for (int i = 1; i <= M; ++i) edges[i] = edges[i - 1] + T[i - 1];

  const double total_T = edges.back() - t0;
  const double dt = (N > 0) ? (total_T / N) : 0.0;

  int s = 0;  // current segment
  for (int i = 0; i < N; ++i) {
    const double ti = t0 + (i + 1) * dt;  // [t0+dt, ..., t0+N*dt]
    t_samples[i] = ti;

    while (s + 1 < (int)edges.size() && ti > edges[s + 1]) ++s;
    const double Ts = T[std::min(s, M - 1)];
    double u = 0.0;
    if (Ts > 0.0) {
      u = (ti - edges[s]) / Ts;
      if (u < 0.0) u = 0.0;
      if (u > 1.0) u = 1.0;
    }
    p_samples[i] = evalBezierQuintic(CP[std::min(s, M - 1)], u);
  }
}

// -----------------------------------------------------------------------------

double SolverLBFGS::evaluateObjectiveAndGradientFused(const Eigen::VectorXd& z,
                                                      Eigen::VectorXd& grad) {
  // -------------------------------------------------------------------------
  // Reconstruct once (P, V, A, CP, T) from z  [matches LBFGS layout / scaling]
  // -------------------------------------------------------------------------
  std::vector<Vec3> P, V, A;
  std::vector<std::vector<Vec3>> CP;
  std::vector<double> T;
  reconstruct(z, P, V, A, CP, T);  // uses τ→T, T̄ decode for interior v̂/â to V/A
  std::vector<mighty::YawKnot> yaw_knots;
  std::vector<std::array<double, 6>> yaw_cp;
  reconstructYaw(z, T, yaw_knots, yaw_cp);
  const int M = static_cast<int>(T.size());
  const int knots = M + 1;
  grad.setZero(K_);

  if (M == 0) return 0.0;

  // -------------------------------------------------------------------------
  // Objective accumulators (UNWEIGHTED); weights applied in final sum
  // -------------------------------------------------------------------------
  double J_time = 0.0;
  double J_panchor = 0.0;
  double J_reference = 0.0;
  double J_progress = 0.0;
  double J_jerk = 0.0;
  double J_yaw_jerk = 0.0;
  double J_yaw_reference = 0.0;
  double J_stat = 0.0;
  double J_vel = 0.0;
  double J_acc = 0.0;
  double J_jmax = 0.0;
  double J_om = 0.0;
  double J_tilt = 0.0;
  double J_thr = 0.0;

  // Grad accumulators in (P,V,A,T) space
  std::vector<Vec3> gP(knots, Vec3::Zero());
  std::vector<Vec3> gV(knots, Vec3::Zero());
  std::vector<Vec3> gA(knots, Vec3::Zero());
  std::vector<double> gT(M, 0.0);
  std::vector<double> gYaw(knots, 0.0);
  std::vector<double> gYawRate(knots, 0.0);
  std::vector<double> gYawAccel(knots, 0.0);

  // The acceptance gate bounds total yaw travel, not just pointwise error.
  // Penalize travel beyond the certified reference so smooth but unnecessary
  // winding receives a descent direction before final validation.
  double yaw_travel_cost = 0.0;
  if (yaw_reference_weight_ > 0.0) {
    constexpr int intervals = mighty::kYawValidationIntervals;
    std::vector<std::array<double, 6>> travel_gradient(M);
    double travel = 0.0, reference_travel = 0.0;
    for (int s = 0; s < M; ++s) {
      reference_travel += std::abs(initial_yaws_[s + 1] - initial_yaws_[s]);
      std::array<double, 6> previous{};
      previous[0] = 1.0;
      for (int sample = 1; sample <= intervals; ++sample) {
        std::array<double, 6> basis{};
        bernstein5(static_cast<double>(sample) / intervals, basis.data());
        double delta = 0.0;
        for (int k = 0; k < 6; ++k) delta += (basis[k] - previous[k]) * yaw_cp[s][k];
        const double magnitude = std::sqrt(delta * delta + 1.0e-12);
        travel += magnitude - 1.0e-6;
        for (int k = 0; k < 6; ++k) {
          travel_gradient[s][k] += delta / magnitude * (basis[k] - previous[k]);
        }
        previous = basis;
      }
    }
    const double excess = std::max(0.0, travel - reference_travel);
    const double weight = yaw_variation_penalty_weight_;
    yaw_travel_cost = weight * excess * excess;
    for (int s = 0; s < M; ++s) {
      for (double& value : travel_gradient[s]) value *= 2.0 * weight * excess;
      accumulate_yaw_cp_gradient(s, travel_gradient[s], yaw_knots, T,
                                 gYaw, gYawRate, gYawAccel, gT);
    }
  }

  // -------------------------------------------------------------------------
  // Constants / sampling setup
  // -------------------------------------------------------------------------
  const int kappa = (integral_resolution_ > 0 ? integral_resolution_ : 30);
  const double mu = (hinge_mu_ > 0.0 ? hinge_mu_ : 1e-2);
  const double Vmax2 = V_max_ * V_max_;
  const double Amax2 = A_max_ * A_max_;
  const double Jmax2 = J_max_ * J_max_;
  const double Om2 = omega_max_ * omega_max_;
  const double cos_tilt_max = std::cos(tilt_max_rad_);
  const double m = mass_, g = g_;
  const Vec3 e3(0.0, 0.0, 1.0);

  std::vector<Vec3> progress_tangents(M + 1, Vec3::Zero());
  if (forward_progress_weight_ > 0.0 && M > 0) {
    for (int knot = 0; knot <= M; ++knot) {
      Vec3 direction = Vec3::Zero();
      if (knot > 0) {
        const Vec3 incoming = global_wps_[knot] - global_wps_[knot - 1];
        if (incoming.norm() > 1.0e-9) direction += incoming.normalized();
      }
      if (knot < M) {
        const Vec3 outgoing = global_wps_[knot + 1] - global_wps_[knot];
        if (outgoing.norm() > 1.0e-9) direction += outgoing.normalized();
      }
      if (direction.norm() <= 1.0e-9) {
        direction = global_wps_[std::min(knot + 1, M)] -
                    global_wps_[std::max(knot - 1, 0)];
      }
      if (direction.norm() > 1.0e-9) {
        progress_tangents[knot] = direction.normalized();
      }
    }
  }

  // GCOPTER thrust ring parameters (mean + radius)
  const double f_mean = 0.5 * (f_min_ + f_max_);
  const double f_radi = 0.5 * std::abs(f_max_ - f_min_);
  const double f_radi2 = f_radi * f_radi;

  // Precompute Bernstein basis for j=0..kappa (shared by all segments)
  // For quintic (degree 5): B5, B4, B3, B2
  // For cubic (degree 3): B3, B2, B1
  std::vector<std::array<double, 6>> B5(kappa + 1);
  std::vector<std::array<double, 5>> B4(kappa + 1);
  std::vector<std::array<double, 4>> B3(kappa + 1);
  std::vector<std::array<double, 3>> B2(kappa + 1);
  std::vector<std::array<double, 2>> B1(kappa + 1);
  std::vector<double> wj(kappa + 1, 1.0);
  if (kappa > 0) {
    for (int j = 0; j <= kappa; ++j) {
      const double tau = static_cast<double>(j) / static_cast<double>(kappa);
      if (degree_ == 5) {
        bernstein5(tau, B5[j].data());
        bernstein4(tau, B4[j].data());
      }
      bernstein3(tau, B3[j].data());
      bernstein2(tau, B2[j].data());
      bernstein1(tau, B1[j].data());
      wj[j] = (j == 0 || j == kappa) ? 0.5 : 1.0;
    }
  }

  // -------------------------------------------------------------------------
  // Segment loop
  // -------------------------------------------------------------------------
  for (int s = 0; s < M; ++s) {
    const double Ts = T[s];
    const double invT = 1.0 / (Ts + 1e-16);
    const double invT2 = invT * invT;
    const double invT3 = invT2 * invT;
    const double invT4 = invT2 * invT2;

    // ---- (1) Closed-form jerk cost and its gradient wrt CP and T
    std::vector<Vec3> gCP_jerk(num_cp_, Vec3::Zero());
    double gT_jerk = 0.0;
    std::array<double, 6> gYawCP{};
    double gT_yaw_jerk = 0.0;

    if (degree_ == 3) {
      // Cubic: 1 D3 vector (constant jerk per segment), coeff = 36/T^5
      const Vec3 d30 = CP[s][3] - 3.0 * CP[s][2] + 3.0 * CP[s][1] - CP[s][0];
      const double S = d30.squaredNorm();
      const double Cj = 36.0 * std::pow(invT, 5);
      J_jerk += Cj * S;
      gCP_jerk[0] = -2.0 * Cj * d30;
      gCP_jerk[1] = 2.0 * Cj * (3.0 * d30);
      gCP_jerk[2] = 2.0 * Cj * (-3.0 * d30);
      gCP_jerk[3] = 2.0 * Cj * d30;
      gT_jerk = (-5.0) * 36.0 * std::pow(invT, 6) * S;
    } else {
      // Quintic: 3 D3 vectors, coeff = 3600/T^5
      const Vec3 d30 = CP[s][3] - 3.0 * CP[s][2] + 3.0 * CP[s][1] - CP[s][0];
      const Vec3 d31 = CP[s][4] - 3.0 * CP[s][3] + 3.0 * CP[s][2] - CP[s][1];
      const Vec3 d32 = CP[s][5] - 3.0 * CP[s][4] + 3.0 * CP[s][3] - CP[s][2];
      const double S = d30.squaredNorm() + d31.squaredNorm() + d32.squaredNorm();
      const double Cj = 3600.0 * std::pow(invT, 5);
      J_jerk += Cj * S;
      gCP_jerk[0] = -2.0 * Cj * d30;
      gCP_jerk[1] = 2.0 * Cj * (3.0 * d30 - d31);
      gCP_jerk[2] = 2.0 * Cj * (-3.0 * d30 + 3.0 * d31 - d32);
      gCP_jerk[3] = 2.0 * Cj * (d30 - 3.0 * d31 + 3.0 * d32);
      gCP_jerk[4] = 2.0 * Cj * (d31 - 3.0 * d32);
      gCP_jerk[5] = 2.0 * Cj * d32;
      gT_jerk = (-5.0) * 3600.0 * std::pow(invT, 6) * S;
    }

    if (degree_ == 5) {
      const double d30 = yaw_cp[s][3] - 3.0 * yaw_cp[s][2] +
                         3.0 * yaw_cp[s][1] - yaw_cp[s][0];
      const double d31 = yaw_cp[s][4] - 3.0 * yaw_cp[s][3] +
                         3.0 * yaw_cp[s][2] - yaw_cp[s][1];
      const double d32 = yaw_cp[s][5] - 3.0 * yaw_cp[s][4] +
                         3.0 * yaw_cp[s][3] - yaw_cp[s][2];
      const double sum = d30 * d30 + d31 * d31 + d32 * d32;
      const double coefficient = 3600.0 * std::pow(invT, 5);
      J_yaw_jerk += coefficient * sum;
      gYawCP[0] = yaw_jerk_weight_ * (-2.0 * coefficient * d30);
      gYawCP[1] = yaw_jerk_weight_ *
                  (2.0 * coefficient * (3.0 * d30 - d31));
      gYawCP[2] = yaw_jerk_weight_ *
                  (2.0 * coefficient * (-3.0 * d30 + 3.0 * d31 - d32));
      gYawCP[3] = yaw_jerk_weight_ *
                  (2.0 * coefficient * (d30 - 3.0 * d31 + 3.0 * d32));
      gYawCP[4] = yaw_jerk_weight_ *
                  (2.0 * coefficient * (d31 - 3.0 * d32));
      gYawCP[5] = yaw_jerk_weight_ * (2.0 * coefficient * d32);
      gT_yaw_jerk = yaw_jerk_weight_ *
                    (-5.0 * 3600.0 * std::pow(invT, 6) * sum);
    }

    // ---- (2) Sampled vehicle-dynamics terms, accumulate CP/T gradients
    std::vector<Vec3> gCP_samp(num_cp_, Vec3::Zero());
    for (int k = 0; k < num_cp_; ++k) gCP_samp[k].setZero();
    double gT_samp = 0.0;

    if (kappa > 0) {
      const double dt = Ts / static_cast<double>(kappa);

      // Precompute Bezier finite differences in shape-space
      const int nd1 = num_cp_ - 1;  // 3 or 5
      const int nd2 = num_cp_ - 2;  // 2 or 4
      std::vector<Vec3> D1(nd1), D2s(nd2);
      for (int j = 0; j < nd1; ++j) D1[j] = CP[s][j + 1] - CP[s][j];
      for (int j = 0; j < nd2; ++j) D2s[j] = CP[s][j + 2] - 2.0 * CP[s][j + 1] + CP[s][j];

      // Pack into dynamic matrices for vectorized evaluation
      Eigen::Matrix<double, 3, Eigen::Dynamic> CP_mat(3, num_cp_);
      Eigen::Matrix<double, 3, Eigen::Dynamic> D1_mat(3, nd1);
      Eigen::Matrix<double, 3, Eigen::Dynamic> D2_mat(3, nd2);
      for (int k = 0; k < num_cp_; ++k) CP_mat.col(k) = CP[s][k];
      for (int k = 0; k < nd1; ++k) D1_mat.col(k) = D1[k];
      for (int k = 0; k < nd2; ++k) D2_mat.col(k) = D2s[k];

      for (int j = 0; j <= kappa; ++j) {
        const double tau = static_cast<double>(j) /
                           static_cast<double>(kappa);
        const double wseg = wj[j] * dt;

        // Evaluate position, velocity, acceleration using degree-appropriate bases
        Vec3 x, d1v, d2v, v, acc, jrk;

        if (degree_ == 3) {
          const auto& b3 = B3[j];
          const auto& b2 = B2[j];
          const auto& b1 = B1[j];
          Eigen::Map<const Eigen::Vector4d> b3_vec(b3.data());
          Eigen::Map<const Eigen::Vector3d> b2_vec(b2.data());
          Eigen::Map<const Eigen::Vector2d> b1_vec(b1.data());

          x = CP_mat * b3_vec;
          d1v = D1_mat * b2_vec;
          d2v = D2_mat * b1_vec;

          v = vel_scale_ * invT * d1v;     // 3/T * d1
          acc = acc_scale_ * invT2 * d2v;  // 6/T² * d2
          // Cubic jerk is constant per segment (handled in closed-form above)
          jrk = Vec3::Zero();
        } else {
          const auto& b5 = B5[j];
          const auto& b4 = B4[j];
          const auto& b3 = B3[j];
          const auto& b2 = B2[j];
          Eigen::Map<const Eigen::Matrix<double, 6, 1>> b5_vec(b5.data());
          Eigen::Map<const Eigen::Matrix<double, 5, 1>> b4_vec(b4.data());
          Eigen::Map<const Eigen::Matrix<double, 4, 1>> b3_vec(b3.data());
          Eigen::Map<const Eigen::Matrix<double, 3, 1>> b2_vec(b2.data());

          // Quintic also needs D3
          const int nd3 = num_cp_ - 3;  // 3
          Eigen::Matrix<double, 3, Eigen::Dynamic> D3_mat(3, nd3);
          for (int k = 0; k < nd3; ++k)
            D3_mat.col(k) = CP[s][k + 3] - 3.0 * CP[s][k + 2] + 3.0 * CP[s][k + 1] - CP[s][k];

          x = CP_mat * b5_vec;
          d1v = D1_mat * b4_vec;
          d2v = D2_mat * b3_vec;
          Vec3 d3v = D3_mat * b2_vec;

          v = vel_scale_ * invT * d1v;       // 5/T * d1
          acc = acc_scale_ * invT2 * d2v;    // 20/T² * d2
          jrk = jrk_scale_ * invT3 * d3v;   // 60/T³ * d3
        }

        // Alias for gradient basis vectors (degree-appropriate)
        // bN_data points to the position-level basis for gradient stencils
        const double* bN_data = (degree_ == 3) ? B3[j].data() : B5[j].data();

        // Optional position reference tube.  Yaw uses its own continuous
        // objective below so an added 2π revolution is never hidden by angle
        // wrapping.
        if (reference_weight_ > 0.0) {
          const double phase = mighty::quinticPosePhase(tau);
          const Vec3 reference_position =
              global_wps_[s] + phase * (global_wps_[s + 1] - global_wps_[s]);
          const Vec3 position_error = x - reference_position;
          const double sample_weight =
              wj[j] / static_cast<double>(kappa);
          J_reference += sample_weight * position_error.squaredNorm();
          const Vec3 position_gradient =
              2.0 * reference_weight_ * sample_weight * position_error;
          for (int k = 0; k < num_cp_; ++k) {
            gCP_samp[k] += bN_data[k] * position_gradient;
          }
        }

        if (yaw_reference_weight_ > 0.0) {
          double yaw_basis[6];
          bernstein5(tau, yaw_basis);
          double yaw = 0.0;
          for (int k = 0; k < 6; ++k) yaw += yaw_basis[k] * yaw_cp[s][k];
          const Vec3 reference_edge = global_wps_[s + 1] - global_wps_[s];
          const double reference_edge_squared = reference_edge.squaredNorm();
          const double yaw_delta =
              initial_yaws_[s + 1] - initial_yaws_[s];
          // Couple rotation to actual translational progress along the
          // collision-certified frontend edge.  A time-parameter reference
          // keeps advancing while translation stalls and therefore rewards
          // rotating nearly in place.  The unbounded projection is kept
          // deliberately smooth for L-BFGS; the position reference tube and
          // safety objective keep the trajectory close to the edge.
          const double yaw_phase =
              spatial_progress_yaw_coupling_ &&
                      reference_edge_squared > 1.0e-12
                  ? (x - global_wps_[s]).dot(reference_edge) /
                        reference_edge_squared
                  : tau;
          const double reference_yaw =
              initial_yaws_[s] + yaw_phase * yaw_delta;
          const double yaw_error = yaw - reference_yaw;
          const double sample_weight =
              wj[j] / static_cast<double>(kappa);
          J_yaw_reference += sample_weight * yaw_error * yaw_error;
          const double yaw_gradient = 2.0 * yaw_reference_weight_ *
              sample_weight * yaw_error;
          for (int k = 0; k < 6; ++k) {
            gYawCP[k] += yaw_basis[k] * yaw_gradient;
          }
          if (spatial_progress_yaw_coupling_ &&
              reference_edge_squared > 1.0e-12 && yaw_delta != 0.0) {
            const Vec3 position_gradient =
                -yaw_gradient * yaw_delta * reference_edge /
                reference_edge_squared;
            for (int k = 0; k < num_cp_; ++k) {
              gCP_samp[k] += bN_data[k] * position_gradient;
            }
          }
        }

        // Give stationary interior knots a genuine descent direction along
        // the collision-certified frontend edge. A speed-norm penalty has a
        // zero derivative at rest and therefore cannot remove these stops.
        // The fixed zero-velocity global boundaries are excluded.
        if (forward_progress_weight_ > 0.0 &&
            minimum_forward_speed_ > 0.0 &&
            !((s == 0 && j == 0) ||
              (s == M - 1 && j == kappa))) {
          Vec3 tangent = (1.0 - tau) * progress_tangents[s] +
                         tau * progress_tangents[s + 1];
          if (tangent.norm() > 1.0e-9) {
            tangent.normalize();
            const double shortfall =
                minimum_forward_speed_ - v.dot(tangent);
            if (shortfall > 0.0) {
              const double penalty = shortfall * shortfall;
              J_progress += penalty * wseg;
              const Vec3 gv = -2.0 * forward_progress_weight_ *
                              shortfall * tangent * wseg;
              const Vec3 g1 = gv * invT;
              const double* bD1 =
                  (degree_ == 3) ? B2[j].data() : B4[j].data();
              for (int k = 0; k < nd1; ++k) {
                const Vec3 G1k = (vel_scale_ * bD1[k]) * g1;
                gCP_samp[k] -= G1k;
                gCP_samp[k + 1] += G1k;
              }
              gT_samp += forward_progress_weight_ *
                          (wj[j] / static_cast<double>(kappa)) * penalty;
              gT_samp += -vel_scale_ * gv.dot(d1v) * invT2;
            }
          }
        }

        // ---------- Velocity: y = ||v||^2 - Vmax^2
        if (dyn_constr_vel_weight_ > 0.0) {
          const double yv = v.squaredNorm() - Vmax2;
          J_vel += smoothed_l1(yv, mu) * wseg;

          if (yv > 0.0) {
            const double phi_p = smoothed_l1_prime(yv, mu);
            const Vec3 gv = (2.0 * phi_p) * v * (dyn_constr_vel_weight_ * wseg);  // ∂J/∂v
            // v = (5/T) d1  �?  ∂J/∂d1 = (5/T) gv
            const Vec3 g1 = gv * invT;
            const double* bD1 = (degree_ == 3) ? B2[j].data() : B4[j].data();
            for (int k = 0; k < nd1; ++k) {
              const Vec3 G1k = (vel_scale_ * bD1[k]) * g1;
              gCP_samp[k] -= G1k;
              gCP_samp[k + 1] += G1k;
            }
            // dt-only + time-scaling path for v
            gT_samp +=
                dyn_constr_vel_weight_ * (wj[j] / static_cast<double>(kappa)) * smoothed_l1(yv, mu);
            gT_samp += -vel_scale_ * gv.dot(d1v) * invT2;
          }
        }

        // ---------- Acceleration: y = ||a||^2 - Amax^2  (present in LBFGS)
        if (dyn_constr_acc_weight_ > 0.0) {
          const double ya = acc.squaredNorm() - Amax2;
          J_acc += smoothed_l1(ya, mu) * wseg;

          if (ya > 0.0) {
            const double phi_p = smoothed_l1_prime(ya, mu);
            const Vec3 ga = (2.0 * phi_p) * acc * (dyn_constr_acc_weight_ * wseg);  // ∂J/∂a
            // a = (20/T^2) d2 �? ∂J/∂d2 = (20/T^2) ga  �?  push via 2nd-diff stencil
            const Vec3 g2 = ga * invT2;
            const double* bD2 = (degree_ == 3) ? B1[j].data() : B3[j].data();
            for (int k = 0; k < nd2; ++k) {
              const Vec3 G2k = (acc_scale_ * bD2[k]) * g2;
              gCP_samp[k] += G2k;
              gCP_samp[k + 1] -= 2.0 * G2k;
              gCP_samp[k + 2] += G2k;
            }
            // dt-only + time-scaling path for a
            gT_samp +=
                dyn_constr_acc_weight_ * (wj[j] / static_cast<double>(kappa)) * smoothed_l1(ya, mu);
            gT_samp += -2.0 * acc_scale_ * ga.dot(d2v) * invT3;
          }
        }

        // --- jerk max constraint y = ||j||^2 �? Jmax^2 (quintic sampled; cubic uses closed-form)
        if (dyn_constr_jerk_weight_ > 0.0 && degree_ == 5) {
          const double yj = jrk.squaredNorm() - Jmax2;
          J_jmax += smoothed_l1(yj, mu) * wseg;

          if (yj > 0.0) {
            const double phi_p = smoothed_l1_prime(yj, mu);
            const Vec3 gj_phys = (2.0 * phi_p) * jrk * (dyn_constr_jerk_weight_ * wseg);  // ∂J/∂j

            // j = (60/T^3) d3  �?  ∂J/∂d3 = (60/T^3) * ∂J/∂j
            const Vec3 g3 = gj_phys * invT3;  // 60 is applied in the stencil below

            // push to CPs via 3rd‑derivative Bézier stencil
            for (int k = 0; k < 3; ++k) {
              const Vec3 G3k = (60.0 * B2[j][k]) * g3;
              gCP_samp[k] -= G3k;
              gCP_samp[k + 1] += 3.0 * G3k;
              gCP_samp[k + 2] -= 3.0 * G3k;
              gCP_samp[k + 3] += G3k;
            }

            // dt‑only term + T‑scaling path
            gT_samp += dyn_constr_jerk_weight_ * (wj[j] / static_cast<double>(kappa)) *
                       smoothed_l1(yj, mu);
            // ∂j/∂T = -(180/T^4) d3
            // Recompute D3 for time-scaling (quintic only, this block is gated on degree==5)
            Eigen::Matrix<double, 3, 3> D3_tmp;
            for (int kk = 0; kk < 3; ++kk)
              D3_tmp.col(kk) = CP[s][kk + 3] - 3.0 * CP[s][kk + 2] + 3.0 * CP[s][kk + 1] - CP[s][kk];
            Eigen::Map<const Eigen::Vector3d> b2_jrk(B2[j].data());
            Vec3 d3_tmp = D3_tmp * b2_jrk;
            gT_samp += -3.0 * jrk_scale_ * gj_phys.dot(d3_tmp) * invT4;
          }
        }

        // ---------- Body-rate: b3dot = (P j)/r,  r=||a+ge3|| (quintic/UAV only)
        if (dyn_constr_bodyrate_weight_ > 0.0 && degree_ == 5) {
          const Vec3 n = acc + g * e3;
          const double r = std::sqrt(n.squaredNorm() + 1e-24);
          const Vec3 b3_rate = n / r;
          // Replace Pmat * jrk with direct formula: jrk - (jrk.dot(b3_rate)) * b3_rate
          // Saves 9 multiplications + 6 additions (matrix-vector mult) -> 5 ops (dot + scale +
          // subtract)
          const Vec3 Pjrk = jrk - jrk.dot(b3_rate) * b3_rate;
          const Vec3 b3dot = Pjrk / r;
          const double yom = b3dot.squaredNorm() - Om2;
          J_om += smoothed_l1(yom, mu) * wseg;

          if (yom > 0.0) {
            const double phi_p = smoothed_l1_prime(yom, mu);
            const Vec3 gb =
                (2.0 * phi_p) * b3dot * (dyn_constr_bodyrate_weight_ * wseg);  // ∂J/∂b3dot

            // Apply projection formula directly instead of matrix multiply
            const Vec3 Pgb = gb - gb.dot(b3_rate) * b3_rate;
            // Pjrk already computed above
            const double alpha = Pjrk.dot(gb);
            const double beta = b3_rate.dot(jrk);
            const double gamma = b3_rate.dot(gb);

            // ∂J/∂j, ∂J/∂a (physical)
            const Vec3 g3_phys = Pgb / r;
            const Vec3 ga_phys = -(beta * Pgb + alpha * b3_rate + gamma * Pjrk) / (r * r);

            // Map to shape derivatives and push to CP
            const Vec3 g2 = ga_phys * invT2;  // 20 in stencil
            const Vec3 g3 = g3_phys * invT3;  // 60 in stencil
            for (int k = 0; k < 4; ++k) {
              const Vec3 G2k = (20.0 * B3[j][k]) * g2;
              gCP_samp[k] += G2k;
              gCP_samp[k + 1] -= 2.0 * G2k;
              gCP_samp[k + 2] += G2k;
            }
            for (int k = 0; k < 3; ++k) {
              const Vec3 G3k = (60.0 * B2[j][k]) * g3;
              gCP_samp[k] -= G3k;
              gCP_samp[k + 1] += 3.0 * G3k;
              gCP_samp[k + 2] -= 3.0 * G3k;
              gCP_samp[k + 3] += G3k;
            }

            // dt-only + time-scaling paths for a and j
            gT_samp += dyn_constr_bodyrate_weight_ * (wj[j] / static_cast<double>(kappa)) *
                       smoothed_l1(yom, mu);
            gT_samp += -40.0 * ga_phys.dot(d2v) * invT3;   // a path
            // Recompute D3 for this quintic-only block
            Eigen::Matrix<double, 3, 3> D3_br;
            for (int kk = 0; kk < 3; ++kk)
              D3_br.col(kk) = CP[s][kk + 3] - 3.0 * CP[s][kk + 2] + 3.0 * CP[s][kk + 1] - CP[s][kk];
            Eigen::Map<const Eigen::Vector3d> b2_br(B2[j].data());
            Vec3 d3_br = D3_br * b2_br;
            gT_samp += -180.0 * g3_phys.dot(d3_br) * invT4;  // j path
          }
        }

        // ---------- Tilt: y = cosθ_max - (e3·b3) (quintic/UAV only)
        if (dyn_constr_tilt_weight_ > 0.0 && degree_ == 5) {
          const Vec3 n = acc + g * e3;
          const double r = std::sqrt(n.squaredNorm() + 1e-24);
          const Vec3 b3_rate = n / r;
          const double cos_t = b3_rate.z();
          const double yt = cos_tilt_max - cos_t;
          J_tilt += smoothed_l1(yt, mu) * wseg;

          if (yt > 0.0) {
            const double phi_p = smoothed_l1_prime(yt, mu);
            const Vec3 dcos_da = (e3 - cos_t * b3_rate) / r;
            const Vec3 ga_phys = -(phi_p)*dcos_da * (dyn_constr_tilt_weight_ * wseg);

            const Vec3 g2 = ga_phys * invT2;
            for (int k = 0; k < 4; ++k) {
              const Vec3 G2k = (20.0 * B3[j][k]) * g2;
              gCP_samp[k] += G2k;
              gCP_samp[k + 1] -= 2.0 * G2k;
              gCP_samp[k + 2] += G2k;
            }

            gT_samp += dyn_constr_tilt_weight_ * (wj[j] / static_cast<double>(kappa)) *
                       smoothed_l1(yt, mu);
            gT_samp += -40.0 * ga_phys.dot(d2v) * invT3;
          }
        }

        // ---------- Thrust ring: φ((f - f_mean)^2 - f_radi^2), f = m * ||a+ge3||
        if (dyn_constr_thrust_weight_ > 0.0 && degree_ == 5) {
          const Vec3 n = acc + g * e3;
          const double r = std::sqrt(n.squaredNorm() + 1e-24);
          const double f = m * r;
          const double df = f - f_mean;
          const double yth = df * df - f_radi2;
          J_thr += smoothed_l1(yth, mu) * wseg;

          if (yth > 0.0) {
            const double phi_p = smoothed_l1_prime(yth, mu);
            const double dphi_df = 2.0 * df * phi_p;
            const Vec3 ga_phys = (dyn_constr_thrust_weight_ * wseg) * (dphi_df * m) * (n / r);

            const Vec3 g2 = ga_phys * invT2;
            for (int k = 0; k < 4; ++k) {
              const Vec3 G2k = (20.0 * B3[j][k]) * g2;
              gCP_samp[k] += G2k;
              gCP_samp[k + 1] -= 2.0 * G2k;
              gCP_samp[k + 2] += G2k;
            }

            gT_samp += dyn_constr_thrust_weight_ * (wj[j] / static_cast<double>(kappa)) *
                       smoothed_l1(yth, mu);
            gT_samp += -40.0 * ga_phys.dot(d2v) * invT3;
          }
        }

      }  // samples
    }    // if kappa > 0

    // Whole-body safety uses its own geometry/yaw-adaptive resolution instead
    // of the generic dynamics integration resolution.
    const bool optimize_segment_safety =
        risk_local_segments_.empty() || risk_local_segments_[s];
    if (whole_body_safety_ && optimize_segment_safety) {
      const int safety_samples =
          optimization_safety_intervals_[static_cast<std::size_t>(s)];
      for (int sample = 0; sample <= safety_samples; ++sample) {
        const double tau = static_cast<double>(sample) / safety_samples;
        const double trapezoid =
            (sample == 0 || sample == safety_samples) ? 0.5 : 1.0;
        std::array<double, 6> basis{};
        if (degree_ == 3) bernstein3(tau, basis.data());
        else bernstein5(tau, basis.data());
        Vec3 position = Vec3::Zero();
        for (int k = 0; k < num_cp_; ++k) position += basis[k] * CP[s][k];
        double yaw = 0.0;
        for (int k = 0; k < 6; ++k) yaw += basis[k] * yaw_cp[s][k];
        if (!whole_body_safety_context_) {
          grad.setZero(z.size());
          return lbfgs::rejectedObjective();
        }
        // Preserve a finite, differentiable objective across the safety
        // boundary. Rejecting unsafe trial points makes Wolfe line search
        // stall at low-speed knots. The hard zone already has a steep
        // exponential penalty; final trajectories still require independent
        // cold whole-body validation before they can be committed.
        const auto safety = whole_body_safety_->evaluateForOptimization(
            position, yaw, *whole_body_safety_context_);
        if (!safety.success) {
          grad.setZero(z.size());
          return lbfgs::rejectedObjective();
        }
        const double constraint_weight = trapezoid;
        J_stat += safety.cost * constraint_weight;
        const Vec3 position_gradient =
            safety.gradient_position * constraint_weight;
        for (int k = 0; k < num_cp_; ++k) {
          gCP_samp[k] += basis[k] * position_gradient;
        }
        for (int k = 0; k < 6; ++k) {
          gYawCP[k] += basis[k] * safety.gradient_yaw * constraint_weight;
        }
      }
    }

    // ---- (3) Push jerk+sample CP grads into (P,V,A) and T (via CP(T))
    // Combine jerk (weighted) + sampled CP gradients
    std::vector<Vec3> gCP_all(num_cp_, Vec3::Zero());
    for (int i = 0; i < num_cp_; ++i) gCP_all[i] = jerk_weight_ * gCP_jerk[i] + gCP_samp[i];

    accumulate_cp_to_pvaT(s, gCP_all, V, A, T, gP, gV, gA, gT, degree_);  // adds the CP(T) path into gT[s]
    accumulate_yaw_cp_gradient(s, gYawCP, yaw_knots, T, gYaw, gYawRate,
                               gYawAccel, gT);
    gT[s] += jerk_weight_ * gT_jerk + gT_samp;  // add the factor/time-scaling paths
    gT[s] += gT_yaw_jerk;

    // ---- (4) Time part of objective
    J_time += Ts;
  }  // segments

  // --- NEW: prox-to-initial P regularizer ------------------------------------
  // Penalize squared distance from the initial P (first reconstruct).
  // Only adds to gP (no dependency on V, A, or T).
  if (pos_anchor_weight_ > 0.0) {
    // If your endpoints are fixed (common), restrict to interior knots so that
    // the objective matches the variables that receive gradients.
    const int i0 = 1;
    const int i1 = knots - 1;
    for (int i = i0; i < i1; ++i) {
      const Vec3 diff = P[i] - P_anchor_[i];
      J_panchor += diff.squaredNorm();  // unweighted objective accumulator
      gP[i] +=
          (2.0 * pos_anchor_weight_) * diff;  // gradient scaled by weight (�?/∂P_i  ||P_i-P0_i||^2)
    }

    // (Optional) If you prefer averaging to make the scale independent of number of knots:
    // if (i1 > i0) J_panchor /= double(i1 - i0);
  }

  // -------------------------------------------------------------------------
  // Scatter once into z, add T̄ coupling (interior only), then chain ∂T→∂τ
  // -------------------------------------------------------------------------
  // (a) scatter (P,V,A) -> z=[p, v̂, â, τ] with T̄-decoding
  std::vector<double> Tbar(knots);
  build_Tbar(T, Tbar);

  auto vhat_from_z = [&](int i) -> Vec3 {
    const int base = zcp_offset_for_knot(i, M_, dof_per_knot_);
    return (base < 0) ? Vec3::Zero() : Vec3(z[base + 3], z[base + 4], z[base + 5]);
  };
  auto ahat_from_z = [&](int i) -> Vec3 {
    const int base = zcp_offset_for_knot(i, M_, dof_per_knot_);
    return (base < 0) ? Vec3::Zero() : Vec3(z[base + 6], z[base + 7], z[base + 8]);
  };

  std::vector<double> gTbar(knots, 0.0);
  for (int i = 1; i < M_; ++i)  // interior knots only
  {
    const int base = zcp_offset_for_knot(i, M_, dof_per_knot_);
    const double Tb = std::max(1e-12, Tbar[i]);

    grad.segment<3>(base + 0) += gP[i];
    grad.segment<3>(base + 3) += (1.0 / Tb) * gV[i];

    // T̄ coupling (decode) �? only where v̂, â exist
    gTbar[i] += -gV[i].dot(vhat_from_z(i)) / (Tb * Tb);

    if (degree_ == 5) {
      grad.segment<3>(base + 6) += (1.0 / (Tb * Tb)) * gA[i];
      gTbar[i] += -2.0 * gA[i].dot(ahat_from_z(i)) / (Tb * Tb * Tb);
    }
  }

  // Yaw shares the same segment times and normalized knot-derivative
  // parameterization as position.
  for (int i = 1; i < M_; ++i) {
    const int base = K_pos_ + 3 * (i - 1);
    const double Tb = std::max(1e-12, Tbar[i]);
    grad[base] += gYaw[i];
    grad[base + 1] += gYawRate[i] / Tb;
    grad[base + 2] += gYawAccel[i] / (Tb * Tb);
    gTbar[i] += -gYawRate[i] * z[base + 1] / (Tb * Tb) -
                2.0 * gYawAccel[i] * z[base + 2] / (Tb * Tb * Tb);
  }
  if (!fix_goal_yaw_) {
    grad[K_pos_ + 3 * std::max(0, M_ - 1)] += gYaw[M_];
  }

  // (b) add the T̄→{T_s} contributions to segment ∂J/∂T
  std::vector<double> gTextra(M, 0.0);
  distribute_gTbar_to_segments(gTbar, gTextra);
  for (int s = 0; s < M; ++s) gT[s] += gTextra[s];

  // (c) chain to τ via dT/dτ and add time term �?(time)/∂�?
  for (int s = 0; s < M; ++s) {
    const double dTdtau = dT_dtau(z[K_cp_ + s]);
    grad[K_cp_ + s] += gT[s] * dTdtau;
    grad[K_cp_ + s] += time_weight_ * dTdtau;  // time term gradient
  }

  // -------------------------------------------------------------------------
  // Final weighted objective (mirrors lbfgs evaluateObjective)
  // -------------------------------------------------------------------------
  const double f = yaw_travel_cost + time_weight_ * J_time +
                   pos_anchor_weight_ * J_panchor + reference_weight_ * J_reference +
                   forward_progress_weight_ * J_progress +
                   jerk_weight_ * J_jerk + yaw_jerk_weight_ * J_yaw_jerk +
                   yaw_reference_weight_ * J_yaw_reference +
                   J_stat +
                   dyn_constr_vel_weight_ * J_vel + dyn_constr_acc_weight_ * J_acc +
                   dyn_constr_jerk_weight_ * J_jmax + dyn_constr_bodyrate_weight_ * J_om +
                   dyn_constr_tilt_weight_ * J_tilt + dyn_constr_thrust_weight_ * J_thr;

  if (!refinement_variable_mask_.empty()) {
    for (int index = 0; index < grad.size(); ++index) {
      if (refinement_variable_mask_[static_cast<std::size_t>(index)] == 0) {
        grad[index] = 0.0;
      }
    }
  }

  return f;
}

// -----------------------------------------------------------------------------

int SolverLBFGS::progressCallback(void* instance, const Eigen::VectorXd& z,
                                  const Eigen::VectorXd& g, const double f, const double step,
                                  const int k, const int ls) {
  // std::cout << "iter=" << k << " f=" << f << " z=" << z.transpose() << " |g|=" << g.norm() <<
  // std::endl; std::cout << "iter=" << k << " f=" << f << " |g|=" << g.norm() << std::endl;
  // printf("iter=%d f=%.6f |g|=%.6f\n", k, f, g.norm());
  return static_cast<SolverLBFGS*>(instance)->optimizationDeadlineExpired()
             ? 1
             : 0;
}

// -----------------------------------------------------------------------------

double SolverLBFGS::evalObjGradCallback(void* instance, const Eigen::VectorXd& x,
                                        Eigen::VectorXd& g) {
  auto* solver = static_cast<SolverLBFGS*>(instance);
  if (solver->optimizationDeadlineExpired()) {
    g.setZero(x.size());
    return lbfgs::rejectedObjective();
  }
  return solver->evaluateObjectiveAndGradientFused(x, g);
  // return static_cast<SolverLBFGS *>(instance)->evaluateObjectiveAndGradient(x, g);
}

// -----------------------------------------------------------------------------

int SolverLBFGS::optimize(const Eigen::VectorXd& z0, Eigen::VectorXd& z_opt, double& f_opt,
                          const lbfgs::lbfgs_parameter_t& param) const {
  optimization_timed_out_ = false;
  // copy initial guess
  Eigen::VectorXd z = z0;
  if (whole_body_safety_context_) {
    whole_body_safety_context_->optimization_derivative_step = 1.0e-5;
  }

  int status =
      lbfgs::lbfgs_optimize(z,                                  // in/out decision vector
                            f_opt,                              // out final cost
                            &SolverLBFGS::evalObjGradCallback,  // objective+gradient callback
                            /*stepbound*/ nullptr,
                            /*progress*/ &SolverLBFGS::progressCallback,
                            // /*progress*/ nullptr,
                            const_cast<SolverLBFGS*>(this), param);

  if (whole_body_safety_context_ && std::isfinite(f_opt) && z.allFinite() &&
      (status == lbfgs::LBFGSERR_MAXIMUMLINESEARCH ||
       status == lbfgs::LBFGSERR_WIDTHTOOSMALL) &&
      !optimizationDeadlineExpired()) {
    // A pose difference can straddle a contact switch and cease to give a
    // descent direction. Retry once with a more local derivative and fresh
    // L-BFGS history from the last accepted iterate, under the same deadline.
    // The exact objective and all final acceptance gates remain unchanged.
    whole_body_safety_context_->optimization_derivative_step = 1.0e-7;
    status = lbfgs::lbfgs_optimize(
        z, f_opt, &SolverLBFGS::evalObjGradCallback, nullptr,
        &SolverLBFGS::progressCallback, const_cast<SolverLBFGS*>(this), param);
  }

  // copy result
  z_opt = z;
  return status;
}

// -----------------------------------------------------------------------------

bool SolverLBFGS::refineYawVariationPenaltyForCurrentTrajectory() {
  if (yaw_reference_weight_ <= 0.0) return false;
  double travel = 0.0, reference_travel = 0.0;
  for (int segment = 0; segment < M_; ++segment) {
    reference_travel += std::abs(initial_yaws_[segment + 1] - initial_yaws_[segment]);
    double previous = evalYawState(segment, 0.0).yaw;
    for (int sample = 1; sample <= mighty::kYawValidationIntervals; ++sample) {
      const double yaw = evalYawState(
          segment, static_cast<double>(sample) / mighty::kYawValidationIntervals).yaw;
      travel += std::abs(yaw - previous);
      previous = yaw;
    }
  }
  if (travel <= reference_travel + mighty::kAdditionalYawVariationBudget) return false;
  // Re-solve the same trajectory with a stronger constraint penalty. The
  // caller retains the same deadline and still applies independent validation.
  yaw_variation_penalty_weight_ *= 10.0;
  return true;
}

bool SolverLBFGS::optimizationDeadlineExpired() const {
  if (!optimization_deadline_ ||
      std::chrono::steady_clock::now() < *optimization_deadline_) {
    return false;
  }
  optimization_timed_out_ = true;
  return true;
}

bool SolverLBFGS::setRiskLocalRefinementSegments(
    const std::vector<bool>& segments) {
  if (segments.empty()) {
    risk_local_segments_.clear();
    refinement_variable_mask_.clear();
    return true;
  }
  if (segments.size() != static_cast<std::size_t>(M_) || K_ <= 0) {
    return false;
  }

  risk_local_segments_ = segments;
  refinement_variable_mask_.assign(static_cast<std::size_t>(K_), 0);
  for (int segment = 0; segment < M_; ++segment) {
    if (segments[static_cast<std::size_t>(segment)]) {
      refinement_variable_mask_[static_cast<std::size_t>(K_cp_ + segment)] = 1;
    }
  }
  for (int knot = 1; knot < M_; ++knot) {
    if (!segments[static_cast<std::size_t>(knot - 1)] ||
        !segments[static_cast<std::size_t>(knot)]) {
      continue;
    }
    const int position_offset = zcp_offset_for_knot(knot, M_, dof_per_knot_);
    for (int dof = 0; dof < dof_per_knot_; ++dof) {
      refinement_variable_mask_[static_cast<std::size_t>(position_offset + dof)] = 1;
    }
    const int yaw_offset = K_pos_ + 3 * (knot - 1);
    for (int dof = 0; dof < 3; ++dof) {
      refinement_variable_mask_[static_cast<std::size_t>(yaw_offset + dof)] = 1;
    }
  }
  if (!fix_goal_yaw_ && segments.back()) {
    const int terminal_yaw_offset =
        K_pos_ + 3 * std::max(0, M_ - 1);
    refinement_variable_mask_[static_cast<std::size_t>(terminal_yaw_offset)] =
        1;
  }
  return true;
}

bool SolverLBFGS::refineSafetySamplingForCurrentTrajectory() {
  if (!whole_body_safety_ || M_ <= 0 ||
      optimization_safety_intervals_.size() != static_cast<std::size_t>(M_)) {
    return false;
  }
  bool changed = false;
  for (int segment = 0; segment < M_; ++segment) {
    if (!risk_local_segments_.empty() && !risk_local_segments_[segment]) {
      continue;
    }
    double path_length = 0.0;
    double yaw_travel = 0.0;
    Vec3 previous_position = evalStateDeriv(segment, 0.0).pos;
    double previous_yaw = evalYawState(segment, 0.0).yaw;
    for (int probe = 1; probe <= 8; ++probe) {
      const double tau = static_cast<double>(probe) / 8.0;
      const Vec3 position = evalStateDeriv(segment, tau).pos;
      const double yaw = evalYawState(segment, tau).yaw;
      path_length += (position - previous_position).norm();
      yaw_travel += std::abs(yaw - previous_yaw);
      previous_position = position;
      previous_yaw = yaw;
    }
    int& intervals =
        optimization_safety_intervals_[static_cast<std::size_t>(segment)];
    const int required =
        safety_sampling_policy_.intervals(path_length, yaw_travel);
    if (required > intervals) {
      intervals = required;
      changed = true;
    }
  }
  return changed;
}

bool SolverLBFGS::hasRiskLocalRefinementVariables() const {
  return std::any_of(refinement_variable_mask_.begin(),
                     refinement_variable_mask_.end(),
                     [](unsigned char enabled) { return enabled != 0; });
}

void SolverLBFGS::reconstructPVATCPopt(const Eigen::VectorXd& z) {
  reconstruct(z, P_opt_, V_opt_, A_opt_, CP_opt_, T_opt_);
  reconstructYaw(z, T_opt_, yaw_knots_opt_, yaw_cp_opt_);

  // std::cout << "Reconstructed P_opt size: " << P_opt_.size() << std::endl;
  // for (const auto &p : P_opt_)
  // {
  //     std::cout << p.transpose() << std::endl;
  // }
  // std::cout << "Reconstructed V_opt size: " << V_opt_.size() << std::endl;
  // for (const auto &v : V_opt_)
  // {
  //     std::cout << v.transpose() << std::endl;
  // }
  // std::cout << "Reconstructed A_opt size: " << A_opt_.size() << std::endl;
  // for (const auto &a : A_opt_)
  // {
  //     std::cout << a.transpose() << std::endl;
  // }
  // std::cout << "Reconstructed T_opt size: " << T_opt_.size() << std::endl;
  // for (const auto &t : T_opt_)
  // {
  //     std::cout << t << " ";
  // }
}

//------------------------------------------------------------------------------

bool SolverLBFGS::retimeForDynamics(double margin) {
  if (T_opt_.empty() || P_opt_.size() != T_opt_.size() + 1 ||
      V_opt_.size() != P_opt_.size() || A_opt_.size() != P_opt_.size() ||
      yaw_knots_opt_.size() != P_opt_.size()) {
    return false;
  }
  double scale = 1.0;
  const int samples = std::max(128, integral_resolution_ * 4);
  for (std::size_t segment = 0; segment < T_opt_.size(); ++segment) {
    for (int sample = 0; sample <= samples; ++sample) {
      const double tau = static_cast<double>(sample) / samples;
      const StateDeriv derivatives =
          evalStateDeriv(static_cast<int>(segment), tau);
      const mighty::YawSample yaw =
          evalYawState(static_cast<int>(segment), tau);
      if (dyn_constr_vel_weight_ > 0.0 && V_max_ > 0.0) {
        scale = std::max(scale, derivatives.vel.norm() / V_max_);
      }
      if (dyn_constr_acc_weight_ > 0.0 && A_max_ > 0.0) {
        scale = std::max(scale,
                         std::sqrt(derivatives.accel.norm() / A_max_));
      }
      if (dyn_constr_jerk_weight_ > 0.0 && J_max_ > 0.0) {
        scale = std::max(scale,
                         std::cbrt(derivatives.jerk.norm() / J_max_));
      }
      if (w_max_ > 0.0) {
        scale = std::max(scale, std::abs(yaw.rate) / w_max_);
      }
      if (yaw_acc_max_ > 0.0) {
        scale = std::max(
            scale, std::sqrt(std::abs(yaw.accel) / yaw_acc_max_));
      }
    }
  }
  if (!std::isfinite(scale)) return false;
  scale = std::max(1.0, scale * std::max(1.0, margin));
  const double scale2 = scale * scale;
  for (double& duration : T_opt_) duration *= scale;
  for (Vec3& velocity : V_opt_) velocity /= scale;
  for (Vec3& acceleration : A_opt_) acceleration /= scale2;
  for (auto& knot : yaw_knots_opt_) {
    knot.rate /= scale;
    knot.accel /= scale2;
  }
  CP_opt_.resize(T_opt_.size());
  yaw_cp_opt_.resize(T_opt_.size());
  for (std::size_t segment = 0; segment < T_opt_.size(); ++segment) {
    CP_opt_[segment] = computeQuinticCP(
        P_opt_[segment], V_opt_[segment], A_opt_[segment],
        P_opt_[segment + 1], V_opt_[segment + 1], A_opt_[segment + 1],
        T_opt_[segment]);
    yaw_cp_opt_[segment] = yaw_control_points(
        yaw_knots_opt_[segment], yaw_knots_opt_[segment + 1],
        T_opt_[segment]);
  }
  return yawDynamicsFeasible() && uavDynamicsFeasible();
}

//------------------------------------------------------------------------------

bool SolverLBFGS::getCurrentDecisionVariables(Eigen::VectorXd& decision) const {
  if (T_opt_.empty() || P_opt_.size() != T_opt_.size() + 1 ||
      V_opt_.size() != P_opt_.size() || A_opt_.size() != P_opt_.size() ||
      yaw_knots_opt_.size() != P_opt_.size()) {
    return false;
  }
  packDecisionVariables(P_opt_, V_opt_, A_opt_, yaw_knots_opt_, T_opt_,
                        decision);
  return decision.allFinite();
}

//------------------------------------------------------------------------------

bool SolverLBFGS::getCoupledDerivativeSeed(
    Eigen::VectorXd& decision,
    std::vector<double>& activation_scales) {
  if (coupled_derivative_seed_.size() > 0 &&
      coupled_derivative_seed_.allFinite() &&
      coupled_derivative_activation_scales_.size() ==
          static_cast<std::size_t>(M_ + 1)) {
    decision = coupled_derivative_seed_;
    activation_scales = coupled_derivative_activation_scales_;
    return true;
  }
  if (!coupled_derivative_seed_pending_ || M_ <= 1 ||
      derivative_seed_positions_.size() !=
          static_cast<std::size_t>(M_ + 1) ||
      derivative_seed_target_velocities_.size() !=
          static_cast<std::size_t>(M_ + 1) ||
      derivative_seed_target_accelerations_.size() !=
          static_cast<std::size_t>(M_ + 1) ||
      derivative_seed_target_yaw_knots_.size() !=
          static_cast<std::size_t>(M_ + 1) ||
      derivative_seed_durations_.size() != static_cast<std::size_t>(M_)) {
    return false;
  }
  coupled_derivative_seed_pending_ = false;
  const auto deadline_reached = [&] {
    return optimization_deadline_ &&
           std::chrono::steady_clock::now() >= *optimization_deadline_;
  };
  if (deadline_reached()) return false;
  bool construction_timed_out = false;

  Eigen::VectorXd original_decision;
  const bool restore_original =
      getCurrentDecisionVariables(original_decision);
  std::vector<Vec3> velocities(M_ + 1, Vec3::Zero());
  std::vector<Vec3> accelerations(M_ + 1, Vec3::Zero());
  std::vector<mighty::YawKnot> yaw_knots(M_ + 1);
  velocities.front() = v0_;
  accelerations.front() = a0_;
  velocities.back() = vf_;
  accelerations.back() = af_;
  for (int knot = 0; knot <= M_; ++knot) {
    yaw_knots[knot].yaw = derivative_seed_target_yaw_knots_[knot].yaw;
  }
  coupled_derivative_activation_scales_.assign(M_ + 1, 0.0);
  coupled_derivative_activation_scales_.front() = 1.0;
  coupled_derivative_activation_scales_.back() = 1.0;
  constexpr std::array<double, 6> kActivationLadder{
      1.0, 0.75, 0.5, 0.25, 0.125, 0.0625};
  constexpr std::array<double, 4> kLowActivationTail{
      0.03125, 0.015625, 0.0078125, 0.00390625};
  constexpr int kActivationRefinementSamples = 8;
  const auto maximize_scale = [&](double initial_scale, auto&& try_scale) {
    double best = initial_scale;
    double rejected_upper = 1.0;
    for (const double scale : kActivationLadder) {
      if (deadline_reached()) {
        construction_timed_out = true;
        return best;
      }
      if (scale <= best) continue;
      if (try_scale(scale)) {
        best = scale;
        break;
      }
      rejected_upper = scale;
    }
    if (best > kActivationLadder.back()) return best;
    if (best == 0.0) {
      for (const double scale : kLowActivationTail) {
        if (deadline_reached()) {
          construction_timed_out = true;
          return best;
        }
        if (try_scale(scale)) {
          best = scale;
          break;
        }
        rejected_upper = scale;
      }
    }
    if (best == 0.0 || rejected_upper <= best) return best;
    const double interval = rejected_upper - best;
    for (int sample = kActivationRefinementSamples - 1; sample >= 1;
         --sample) {
      if (deadline_reached()) {
        construction_timed_out = true;
        return best;
      }
      const double scale =
          best + interval * static_cast<double>(sample) /
                     static_cast<double>(kActivationRefinementSamples);
      if (try_scale(scale)) return scale;
    }
    return best;
  };
  const auto activate_knot = [&](int knot) {
    const auto try_scale = [&](double scale) {
      const Vec3 old_velocity = velocities[knot];
      const Vec3 old_acceleration = accelerations[knot];
      const mighty::YawKnot old_yaw = yaw_knots[knot];
      velocities[knot] =
          scale * derivative_seed_target_velocities_[knot];
      accelerations[knot] =
          scale * derivative_seed_target_accelerations_[knot];
      yaw_knots[knot].rate =
          scale * derivative_seed_target_yaw_knots_[knot].rate;
      yaw_knots[knot].accel =
          scale * derivative_seed_target_yaw_knots_[knot].accel;
      if (!whole_body_safety_ || adjacentSegmentsWholeBodySafe(
              derivative_seed_positions_, velocities, accelerations,
              yaw_knots, derivative_seed_durations_, knot)) {
        coupled_derivative_activation_scales_[knot] = scale;
        return true;
      }
      velocities[knot] = old_velocity;
      accelerations[knot] = old_acceleration;
      yaw_knots[knot] = old_yaw;
      return false;
    };
    coupled_derivative_activation_scales_[knot] = maximize_scale(
        coupled_derivative_activation_scales_[knot], try_scale);
  };
  for (int knot = 1; knot < M_ && !construction_timed_out; ++knot) {
    activate_knot(knot);
  }
  for (int knot = M_ - 1; knot >= 1 && !construction_timed_out; --knot) {
    activate_knot(knot);
  }
  if (construction_timed_out) {
    if (restore_original) reconstructPVATCPopt(original_decision);
    coupled_derivative_activation_scales_.clear();
    return false;
  }
  packDecisionVariables(
      derivative_seed_positions_, velocities, accelerations, yaw_knots,
      derivative_seed_durations_, coupled_derivative_seed_);
  if (restore_original) reconstructPVATCPopt(original_decision);
  if (!coupled_derivative_seed_.allFinite()) {
    coupled_derivative_seed_.resize(0);
    coupled_derivative_activation_scales_.clear();
    return false;
  }
  decision = coupled_derivative_seed_;
  activation_scales = coupled_derivative_activation_scales_;
  return true;
}

//------------------------------------------------------------------------------

bool SolverLBFGS::makeRestToRestDecisionVariables(
    Eigen::VectorXd& decision) const {
  if (T_opt_.empty() || P_opt_.size() != T_opt_.size() + 1 ||
      V_opt_.size() != P_opt_.size() || A_opt_.size() != P_opt_.size() ||
      yaw_knots_opt_.size() != P_opt_.size()) {
    return false;
  }
  auto velocities = V_opt_;
  auto accelerations = A_opt_;
  auto yaw_knots = yaw_knots_opt_;
  for (std::size_t knot = 1; knot + 1 < P_opt_.size(); ++knot) {
    velocities[knot].setZero();
    accelerations[knot].setZero();
    yaw_knots[knot].rate = 0.0;
    yaw_knots[knot].accel = 0.0;
  }
  packDecisionVariables(P_opt_, velocities, accelerations, yaw_knots, T_opt_,
                        decision);
  return decision.allFinite();
}

//------------------------------------------------------------------------------

bool SolverLBFGS::evaluateAtElapsedTime(double elapsed_time,
                                        state& sample) const {
  if (T_opt_.empty() || !std::isfinite(elapsed_time)) return false;
  const double total_time =
      std::accumulate(T_opt_.begin(), T_opt_.end(), 0.0);
  const double clamped_time = std::clamp(elapsed_time, 0.0, total_time);
  double segment_start = 0.0;
  int segment = static_cast<int>(T_opt_.size()) - 1;
  for (std::size_t candidate = 0; candidate < T_opt_.size(); ++candidate) {
    if (clamped_time <= segment_start + T_opt_[candidate]) {
      segment = static_cast<int>(candidate);
      break;
    }
    segment_start += T_opt_[candidate];
  }
  const double duration = std::max(1.0e-12, T_opt_[segment]);
  const double tau = std::clamp(
      (clamped_time - segment_start) / duration, 0.0, 1.0);
  const StateDeriv derivatives = evalStateDeriv(segment, tau);
  const mighty::YawSample yaw = evalYawState(segment, tau);
  sample.setZero();
  sample.t = t0_ + clamped_time;
  sample.pos = derivatives.pos;
  sample.vel = derivatives.vel;
  sample.accel = derivatives.accel;
  sample.jerk = derivatives.jerk;
  sample.yaw = yaw.yaw;
  sample.dyaw = yaw.rate;
  sample.use_tracking_yaw = true;
  return sample.pos.allFinite() && sample.vel.allFinite() &&
         sample.accel.allFinite() && sample.jerk.allFinite() &&
         std::isfinite(sample.yaw) && std::isfinite(sample.dyaw);
}

//------------------------------------------------------------------------------

mighty::YawSample SolverLBFGS::evalYawState(int segment, double tau) const {
  mighty::YawSample sample;
  if (yaw_cp_opt_.empty() || T_opt_.empty()) return sample;

  segment = std::clamp(segment, 0, static_cast<int>(yaw_cp_opt_.size()) - 1);
  tau = std::clamp(tau, 0.0, 1.0);
  const auto& cp = yaw_cp_opt_[segment];
  const double duration = std::max(1.0e-12, T_opt_[segment]);
  double b5[6], b4[5], b3[4], b2[3];
  bernstein5(tau, b5);
  bernstein4(tau, b4);
  bernstein3(tau, b3);
  bernstein2(tau, b2);

  for (int k = 0; k < 6; ++k) sample.yaw += b5[k] * cp[k];
  for (int k = 0; k < 5; ++k) {
    sample.rate += 5.0 * b4[k] * (cp[k + 1] - cp[k]) / duration;
  }
  for (int k = 0; k < 4; ++k) {
    sample.accel += 20.0 * b3[k] *
                    (cp[k + 2] - 2.0 * cp[k + 1] + cp[k]) /
                    (duration * duration);
  }
  for (int k = 0; k < 3; ++k) {
    sample.jerk += 60.0 * b2[k] *
                   (cp[k + 3] - 3.0 * cp[k + 2] + 3.0 * cp[k + 1] - cp[k]) /
                   (duration * duration * duration);
  }
  return sample;
}

//------------------------------------------------------------------------------

bool SolverLBFGS::yawDynamicsFeasible(double tolerance) const {
  if (yaw_cp_opt_.size() != T_opt_.size()) return false;
  for (std::size_t segment = 0; segment < T_opt_.size(); ++segment) {
    const auto& cp = yaw_cp_opt_[segment];
    const double duration = T_opt_[segment];
    if (!std::isfinite(duration) || duration <= 0.0) return false;

    std::array<double, 4> acceleration_controls{};
    for (int k = 0; k < 4; ++k) {
      acceleration_controls[k] =
          20.0 * (cp[k + 2] - 2.0 * cp[k + 1] + cp[k]) /
          (duration * duration);
    }
    const auto& q = acceleration_controls;
    std::vector<double> rate_candidates{0.0, 1.0};
    const auto acceleration_roots = real_roots_in_unit_interval(
        {q[0], 3.0 * (q[1] - q[0]),
         3.0 * (q[0] - 2.0 * q[1] + q[2]),
         -q[0] + 3.0 * q[1] - 3.0 * q[2] + q[3]});
    rate_candidates.insert(rate_candidates.end(), acceleration_roots.begin(),
                           acceleration_roots.end());

    std::array<double, 3> jerk_controls{};
    for (int k = 0; k < 3; ++k) {
      jerk_controls[k] =
          60.0 * (cp[k + 3] - 3.0 * cp[k + 2] +
                  3.0 * cp[k + 1] - cp[k]) /
          (duration * duration * duration);
    }
    const auto& r = jerk_controls;
    std::vector<double> acceleration_candidates{0.0, 1.0};
    const auto jerk_roots = real_roots_in_unit_interval(
        {r[0], 2.0 * (r[1] - r[0]), r[0] - 2.0 * r[1] + r[2]});
    acceleration_candidates.insert(acceleration_candidates.end(),
                                   jerk_roots.begin(), jerk_roots.end());

    for (double tau : rate_candidates) {
      if (std::abs(evalYawState(static_cast<int>(segment), tau).rate) >
          w_max_ + tolerance) {
        return false;
      }
    }
    for (double tau : acceleration_candidates) {
      if (std::abs(evalYawState(static_cast<int>(segment), tau).accel) >
          yaw_acc_max_ + tolerance) {
        return false;
      }
    }
  }
  return true;
}

//------------------------------------------------------------------------------

bool SolverLBFGS::uavDynamicsFeasible(double relative_tolerance) const {
  if (T_opt_.empty()) return false;
  const double tolerance = std::max(0.0, relative_tolerance);
  const auto within_upper = [tolerance](double value, double limit) {
    return std::isfinite(value) && value <= limit * (1.0 + tolerance) + 1.0e-9;
  };
  const auto within_lower = [tolerance](double value, double limit) {
    return std::isfinite(value) && value + std::abs(limit) * tolerance + 1.0e-9 >= limit;
  };
  const Eigen::Vector3d gravity_axis(0.0, 0.0, 1.0);
  const int samples = std::max(64, integral_resolution_ * 2);
  for (std::size_t segment = 0; segment < T_opt_.size(); ++segment) {
    if (!std::isfinite(T_opt_[segment]) || T_opt_[segment] <= 0.0) return false;
    for (int sample = 0; sample <= samples; ++sample) {
      const double tau = static_cast<double>(sample) / samples;
      const StateDeriv derivatives =
          evalStateDeriv(static_cast<int>(segment), tau);
      if (!derivatives.pos.allFinite() || !derivatives.vel.allFinite() ||
          !derivatives.accel.allFinite() || !derivatives.jerk.allFinite()) {
        return false;
      }
      if ((dyn_constr_vel_weight_ > 0.0 &&
           !within_upper(derivatives.vel.norm(), V_max_)) ||
          (dyn_constr_acc_weight_ > 0.0 &&
           !within_upper(derivatives.accel.norm(), A_max_)) ||
          (dyn_constr_jerk_weight_ > 0.0 &&
           !within_upper(derivatives.jerk.norm(), J_max_))) {
        return false;
      }

      const Eigen::Vector3d thrust_vector = derivatives.accel + g_ * gravity_axis;
      const double thrust_norm = thrust_vector.norm();
      if (!std::isfinite(thrust_norm) || thrust_norm <= 1.0e-12) return false;
      const Eigen::Vector3d body_z = thrust_vector / thrust_norm;
      const double tilt = std::acos(std::clamp(body_z.z(), -1.0, 1.0));
      const Eigen::Vector3d projected_jerk =
          derivatives.jerk - derivatives.jerk.dot(body_z) * body_z;
      const double body_rate = projected_jerk.norm() / thrust_norm;
      const double thrust = mass_ * thrust_norm;
      if ((dyn_constr_bodyrate_weight_ > 0.0 &&
           !within_upper(body_rate, omega_max_)) ||
          (dyn_constr_tilt_weight_ > 0.0 &&
           !within_upper(tilt, tilt_max_rad_)) ||
          (dyn_constr_thrust_weight_ > 0.0 &&
           (!within_lower(thrust, f_min_) || !within_upper(thrust, f_max_)))) {
        return false;
      }
    }
  }
  return true;
}

//------------------------------------------------------------------------------

int SolverLBFGS::findSegment(double ti, const std::vector<double>& T) const {
  int M = int(T.size());
  static std::vector<double> ends;
  ends.resize(M);
  ends[0] = T[0];
  for (int i = 1; i < M; ++i) ends[i] = ends[i - 1] + T[i];
  auto it = std::upper_bound(ends.begin(), ends.end(), ti);
  int s = int(it - ends.begin());
  return (s < M ? s : M - 1);
}

//------------------------------------------------------------------------------

// Fast and stable version
void SolverLBFGS::evalBernstein5(double tau, double B[6], double dB[6]) const {
  // clamp into [0,1] to avoid tiny overshoots
  if (tau <= 0.0)
    tau = 0.0;
  else if (tau >= 1.0)
    tau = 1.0;

  double u = tau;
  double v = 1.0 - tau;

  // compute all powers with just multiplies
  double u2 = u * u;
  double u3 = u2 * u;
  double u4 = u3 * u;
  double u5 = u4 * u;

  double v2 = v * v;
  double v3 = v2 * v;
  double v4 = v3 * v;
  double v5 = v4 * v;

  // degree�?5 Bernstein basis
  B[0] = v5;
  B[1] = 5.0 * u * v4;
  B[2] = 10.0 * u2 * v3;
  B[3] = 10.0 * u3 * v2;
  B[4] = 5.0 * u4 * v;
  B[5] = u5;

  // degree�?4 Bernstein used for the derivative formula
  //    B4[k] = C(4,k) u^k v^(4−k)
  double B4_0 = v4;
  double B4_1 = 4 * u * v3;
  double B4_2 = 6 * u2 * v2;
  double B4_3 = 4 * u3 * v;
  double B4_4 = u4;

  // dB[j] = 5 * ( B4[j�?1] �? B4[j] ), with B4[�?1]=B4[5]=0
  dB[0] = -5.0 * B4_0;
  dB[1] = 5.0 * (B4_0 - B4_1);
  dB[2] = 5.0 * (B4_1 - B4_2);
  dB[3] = 5.0 * (B4_2 - B4_3);
  dB[4] = 5.0 * (B4_3 - B4_4);
  dB[5] = 5.0 * B4_4;
}

//------------------------------------------------------------------------------

Eigen::Vector3d SolverLBFGS::evalSample(const std::vector<std::array<Eigen::Vector3d, 6>>& CP,
                                        const std::vector<double>& T, int sample_i, int num_samples,
                                        int& seg, double& tau, double& t0) const {
  // 1) absolute time of this sample
  double total_time = std::accumulate(T.begin(), T.end(), 0.0);
  double alpha = double(sample_i) / double(num_samples - 1);
  double ti = total_time * alpha;

  // 2) find segment index
  seg = findSegment(ti, T);

  // 3) compute t0 and τ
  t0 = (seg > 0) ? std::accumulate(T.begin(), T.begin() + seg, 0.0) : 0.0;
  double Ti = T[seg];
  tau = (ti - t0) / Ti;

  // 4) Bernstein basis at τ
  double B[6], dB[6];
  evalBernstein5(tau, B, dB);

  // 5) form position
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  for (int j = 0; j < 6; ++j) {
    p += CP[seg][j] * B[j];
  }
  return p;
}

//------------------------------------------------------------------------------

Eigen::Vector3d SolverLBFGS::evalObs(const Eigen::Matrix<double, 6, 1>& cx,
                                     const Eigen::Matrix<double, 6, 1>& cy,
                                     const Eigen::Matrix<double, 6, 1>& cz, double t) const {
  double tp = 1.0;
  double x = 0, y = 0, zv = 0;
  for (int j = 0; j < 6; ++j) {
    x += cx[j] * tp;
    y += cy[j] * tp;
    zv += cz[j] * tp;
    tp *= t;
  }
  return {x, y, zv};
}

// -----------------------------------------------------------------------------

void SolverLBFGS::sanityCheck() const {
  // A single rest-to-rest segment is a supported direct trajectory; only an
  // empty route is invalid.
  if (M_ < 1) {
    std::cout << "\033[31mError: M_ must be positive.\033[0m" << std::endl;
  }

  if (K_ <= 0) {
    std::cout << "\033[31mError: K_ must be positive.\033[0m" << std::endl;
  }

  // Check if the number of segments matches (the number of global waypoints - 1)
  if (M_ != global_wps_.size() - 1) {
    // red color
    std::cout << "\033[31mError: Number of segments does not match the number of global waypoints "
                 "- 1.\033[0m"
              << std::endl;
    std::cout << "M_ = " << M_ << ", global_wps_.size() = " << global_wps_.size() << std::endl;
  }
}
