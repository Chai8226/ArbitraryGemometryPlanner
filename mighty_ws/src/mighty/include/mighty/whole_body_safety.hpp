#pragma once

#include <Eigen/Core>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mighty::whole_body_safety {

struct SafetySamplingPolicy {
  double position_step{0.2};
  double yaw_step{15.0 * M_PI / 180.0};
  int minimum_intervals{8};
  int maximum_intervals{256};

  int intervals(double position_length, double yaw_travel) const {
    const int position_intervals = position_step > 0.0
        ? static_cast<int>(std::ceil(position_length / position_step))
        : maximum_intervals;
    const int yaw_intervals = yaw_step > 0.0
        ? static_cast<int>(std::ceil(yaw_travel / yaw_step))
        : maximum_intervals;
    return std::clamp(std::max(position_intervals, yaw_intervals),
                      minimum_intervals, maximum_intervals);
  }
};

struct ConvexPolytope {
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vertices;

  bool valid(double tolerance = 1.0e-9) const;
  // Recompute a strictly interior scaling centre and the corresponding radial
  // half-space offsets.  Plane normals are normalized in-place.
  void finalize(double outward_offset = 0.0);

  Eigen::VectorXd radial_offsets;
  double aabb_padding{0.0};
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ProximityOptions {
  bool use_sdlp{true};
  double residual_tolerance{1.0e-7};
  double initial_barrier{1.0e-1};
  double final_barrier{1.0e-9};
  double barrier_reduction{0.1};
  double kkt_regularization{1.0e-9};
  int max_newton_iterations{12};
  int max_line_search_iterations{24};
};

struct ProximityWarmStart {
  Eigen::Vector4d primal{Eigen::Vector4d::Zero()};
  // A certified LP optimum in four variables is normally supported by four
  // independent active planes.  Keeping their row indices turns the next
  // nearby query into one 4x4 solve plus a full KKT check.
  std::array<int, 4> active_rows{{-1, -1, -1, -1}};
  int active_count{0};
  bool valid{false};
};

struct ProximityWorkspace {
  using ConstraintMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>;

  ConstraintMatrix constraints;
  Eigen::VectorXd bounds;
  Eigen::VectorXi permutation;
  Eigen::VectorXi next;
  Eigen::VectorXi previous;
  Eigen::Matrix<double, 5, Eigen::Dynamic, Eigen::ColMajor>
      homogeneous_halfspaces;
  Eigen::VectorXd sdlp_work;

  void reserveRows(std::size_t rows) {
    if (constraints.rows() < static_cast<Eigen::Index>(rows)) {
      constraints.resize(static_cast<Eigen::Index>(rows), 4);
      bounds.resize(static_cast<Eigen::Index>(rows));
    }
    const Eigen::Index columns = static_cast<Eigen::Index>(rows + 1);
    if (homogeneous_halfspaces.cols() < columns) {
      permutation.resize(static_cast<Eigen::Index>(rows));
      next.resize(columns);
      previous.resize(columns + 1);
      homogeneous_halfspaces.resize(5, columns);
      sdlp_work.resize(9 * (columns + 3));
    }
  }
};

struct ProximityResult {
  bool success{false};
  double beta{std::numeric_limits<double>::infinity()};
  Eigen::Vector3d contact{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gradient_position{Eigen::Vector3d::Zero()};
  double gradient_yaw{0.0};
  int newton_iterations{0};
  bool used_sdlp{false};
  bool used_legacy_solver{false};
  double primal_residual{std::numeric_limits<double>::infinity()};
  int failure_stage{0};
  ProximityWarmStart warm_start;
};

// Differentiable minimum-uniform-scaling query for a yaw-rotated body
// polytope and a static world-frame obstacle polytope.  The four primal
// variables are the contact point (x,y,z) and beta.
class ScalingProximitySolver {
 public:
  explicit ScalingProximitySolver(ProximityOptions options = {});

  ProximityResult solve(const ConvexPolytope& body,
                        const ConvexPolytope& obstacle,
                        const Eigen::Vector3d& position, double yaw,
                        const ProximityWarmStart* warm_start = nullptr,
                        ProximityWorkspace* workspace = nullptr) const;
  bool minimumBeta(const ConvexPolytope& body, const ConvexPolytope& obstacle,
                   const Eigen::Vector3d& position, double yaw, double& beta,
                   ProximityWorkspace* workspace = nullptr) const;

 private:
  ProximityOptions options_;
};

struct ExponentialHingeOptions {
  double beta_soft{1.15};
  double acceptance_beta{1.0 - 1.0e-4};
  double weight{1000.0};
  double exponential_gain{6.0};
  double hard_weight_multiplier{20.0};
  double maximum_exponent{40.0};
};

struct PairPenalty {
  double cost{0.0};
  double derivative_beta{0.0};
  bool active{false};
};

PairPenalty exponentialHinge(double beta,
                             const ExponentialHingeOptions& options);

struct Aabb {
  Eigen::Vector3d minimum{Eigen::Vector3d::Constant(
      std::numeric_limits<double>::infinity())};
  Eigen::Vector3d maximum{Eigen::Vector3d::Constant(
      -std::numeric_limits<double>::infinity())};

  bool valid() const;
  bool overlaps(const Aabb& other) const;
  void extend(const Eigen::Vector3d& point);
  void extend(const Aabb& other);
};

struct CandidatePair {
  std::size_t body{0};
  std::size_t obstacle{0};
};

// Immutable known-map safety scene.  PolyMap artifacts are loaded once and a
// static BVH is built over metric-inflated obstacle components and six virtual
// boundary walls.
class SafetyScene {
 public:
  static SafetyScene load(const std::string& map_polymap_path,
                          const std::string& body_polymap_path,
                          double body_clearance,
                          double virtual_ceiling_height,
                          double beta_soft = 1.15);

  std::vector<CandidatePair> candidates(const Eigen::Vector3d& position,
                                        double yaw,
                                        double beta_soft) const;
  void candidates(const Eigen::Vector3d& position, double yaw,
                  double beta_soft, std::vector<CandidatePair>& result,
                  std::vector<int>& obstacle_scratch) const;

  const ConvexPolytope& body(std::size_t index) const {
    return body_components_.at(index);
  }
  const ConvexPolytope& obstacle(std::size_t index) const {
    return obstacle_components_.at(index);
  }
  std::size_t bodyCount() const { return body_components_.size(); }
  std::size_t obstacleCount() const { return obstacle_components_.size(); }
  std::size_t maxConstraintRows() const { return max_constraint_rows_; }

 private:
  struct BvhNode {
    Aabb bounds;
    int left{-1};
    int right{-1};
    int obstacle{-1};
  };

  int buildBvh(std::vector<int>& indices, std::size_t begin,
               std::size_t end);
  void queryBvh(int node, const Aabb& query,
                std::vector<int>& result) const;

  std::vector<ConvexPolytope> body_components_;
  std::vector<ConvexPolytope> obstacle_components_;
  std::vector<Aabb> obstacle_bounds_;
  std::vector<BvhNode> bvh_;
  int bvh_root_{-1};
  double bvh_scale_{1.15};
  std::size_t max_constraint_rows_{0};
};

struct SafetyStatistics {
  std::size_t candidate_pairs{0};
  std::size_t proximity_queries{0};
  std::size_t active_pairs{0};
  std::size_t cold_retries{0};
  std::size_t failed_queries{0};
  int last_failure_stage{0};
  std::size_t total_newton_iterations{0};
  std::size_t active_set_fast_hits{0};
  std::size_t active_set_fast_misses{0};
  std::size_t legacy_solver_calls{0};
  std::size_t sdlp_calls{0};
  std::size_t sdlp_failures{0};
  double minimum_beta{std::numeric_limits<double>::infinity()};
  std::size_t minimum_beta_body{std::numeric_limits<std::size_t>::max()};
  std::size_t minimum_beta_obstacle{std::numeric_limits<std::size_t>::max()};

  void merge(const SafetyStatistics& other);
};

// Mutable, allocation-stable state for one sequential planning run.  The
// geometry scene and solver options remain immutable, so independent planning
// runs can use independent contexts without a global lock.
struct SafetyQueryContext {
  double optimization_derivative_step{1.0e-5};
  std::unordered_map<std::size_t, ProximityWarmStart> warm_starts;
  std::vector<CandidatePair> candidate_pairs;
  std::vector<int> obstacle_scratch;
  ProximityWorkspace proximity_workspace;
  SafetyStatistics cumulative_statistics;

  void clearWarmStarts();
  void clearStatistics() { cumulative_statistics = {}; }
};

struct PoseSafetyResult {
  bool success{true};
  double cost{0.0};
  Eigen::Vector3d gradient_position{Eigen::Vector3d::Zero()};
  double gradient_yaw{0.0};
  SafetyStatistics statistics;
};

// Shared immutable safety oracle used by both alternating optimizers.  All
// mutable warm-start and scratch state lives in SafetyQueryContext.
class WholeBodySafety {
 public:
  WholeBodySafety(std::shared_ptr<const SafetyScene> scene,
                  ExponentialHingeOptions penalty_options,
                  ProximityOptions warm_options = {},
                  ProximityOptions cold_options = {});

  SafetyQueryContext createContext() const;
  PoseSafetyResult evaluate(const Eigen::Vector3d& position, double yaw,
                            SafetyQueryContext& context,
                            bool stop_on_hard_collision = false) const;
  PoseSafetyResult evaluateForOptimization(
      const Eigen::Vector3d& position, double yaw,
      SafetyQueryContext& context) const;
  // Convenience cold query for tests and occasional diagnostics.  Optimizer
  // hot paths must use the context overload.
  PoseSafetyResult evaluate(const Eigen::Vector3d& position, double yaw) const;
  const SafetyScene& scene() const { return *scene_; }
  double betaSoft() const { return penalty_options_.beta_soft; }
  double acceptanceBeta() const { return penalty_options_.acceptance_beta; }

 private:
  PoseSafetyResult evaluateWithSolvers(
      const Eigen::Vector3d& position, double yaw, SafetyQueryContext& context,
      bool stop_on_hard_collision, const ScalingProximitySolver& primary,
      const ScalingProximitySolver& retry, bool allow_warm_start) const;
  std::size_t pairIndex(const CandidatePair& pair) const;

  std::shared_ptr<const SafetyScene> scene_;
  ExponentialHingeOptions penalty_options_;
  ScalingProximitySolver warm_solver_;
  ScalingProximitySolver cold_solver_;
};

}  // namespace mighty::whole_body_safety
