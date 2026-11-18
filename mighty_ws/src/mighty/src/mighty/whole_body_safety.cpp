#include <mighty/whole_body_safety.hpp>
#include <mighty/third_party/sdlp.hpp>

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/QR>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace mighty::whole_body_safety {
namespace {

constexpr double kMinimumRadialOffset = 1.0e-8;
using Json = nlohmann::json;

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
verticesFromHalfspaces(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                       double tolerance) {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> result;
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = i + 1; j < A.rows(); ++j) {
      for (int k = j + 1; k < A.rows(); ++k) {
        Eigen::Matrix3d active;
        active.row(0) = A.row(i);
        active.row(1) = A.row(j);
        active.row(2) = A.row(k);
        Eigen::FullPivLU<Eigen::Matrix3d> factor(active);
        if (factor.rank() != 3) continue;
        const Eigen::Vector3d candidate =
            factor.solve(Eigen::Vector3d(b[i], b[j], b[k]));
        if (!candidate.allFinite() ||
            (A * candidate - b).maxCoeff() > tolerance) {
          continue;
        }
        const bool duplicate = std::any_of(
            result.begin(), result.end(), [&](const Eigen::Vector3d& vertex) {
              return (vertex - candidate).norm() <= tolerance;
            });
        if (!duplicate) result.push_back(candidate);
      }
    }
  }
  return result;
}

Eigen::Matrix3d yawRotation(double yaw) {
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  Eigen::Matrix3d result;
  result << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
  return result;
}

Eigen::Matrix3d yawRotationDerivative(double yaw) {
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  Eigen::Matrix3d result;
  result << -s, -c, 0.0, c, -s, 0.0, 0.0, 0.0, 0.0;
  return result;
}

template <typename MatrixType, typename VectorType>
bool solveLinearProgram4(const Eigen::Vector4d& objective,
                         const MatrixType& constraints,
                         const VectorType& bounds,
                         ProximityWorkspace& workspace,
                         Eigen::Vector4d& optimum) {
  const int rows = static_cast<int>(bounds.size());
  const int marker = rows + 1;
  workspace.reserveRows(static_cast<std::size_t>(rows));
  auto halves = workspace.homogeneous_halfspaces.leftCols(marker);
  halves.col(0).setZero();
  halves(4, 0) = 1.0;
  halves.topRightCorner(4, rows) = -constraints.transpose();
  halves.bottomRightCorner(1, rows) = bounds.transpose();
  for (int column = 0; column < marker; ++column) {
    const double norm = halves.col(column).norm();
    if (!(norm > sdlp::eps) || !std::isfinite(norm)) return false;
    halves.col(column) /= norm;
  }

  Eigen::Matrix<double, 5, 1> numerator;
  numerator.head<4>() = objective;
  numerator[4] = 0.0;
  Eigen::Matrix<double, 5, 1> denominator =
      Eigen::Matrix<double, 5, 1>::Zero();
  denominator[4] = 1.0;
  Eigen::Matrix<double, 5, 1> homogeneous_optimum;

  sdlp::rand_permutation(rows, workspace.permutation.data());
  workspace.previous[0] = 0;
  workspace.next[0] = workspace.permutation[0] + 1;
  workspace.previous[workspace.permutation[0] + 1] = 0;
  for (int index = 0; index < rows - 1; ++index) {
    workspace.next[workspace.permutation[index] + 1] =
        workspace.permutation[index + 1] + 1;
    workspace.previous[workspace.permutation[index + 1] + 1] =
        workspace.permutation[index] + 1;
  }
  workspace.next[workspace.permutation[rows - 1] + 1] = marker;

  const int status = sdlp::linfracprog<4>(
      halves.data(), marker, marker, numerator.data(), denominator.data(),
      homogeneous_optimum.data(), workspace.sdlp_work.data(),
      workspace.next.data(), workspace.previous.data());
  if (status == sdlp::INFEASIBLE || status == sdlp::UNBOUNDED ||
      std::abs(homogeneous_optimum[4]) <= sdlp::eps) {
    return false;
  }
  optimum = homogeneous_optimum.head<4>() / homogeneous_optimum[4];
  return optimum.allFinite();
}

template <typename MatrixType, typename VectorType>
bool strictlyFeasible(const MatrixType& G, const VectorType& h,
                      const Eigen::Vector4d& z) {
  return ((h - G * z).array() > 0.0).all();
}

template <typename MatrixType, typename VectorType>
Eigen::Vector4d feasibleInitialPoint(const ConvexPolytope& body,
                                    const ConvexPolytope& obstacle,
                                    const Eigen::Vector3d& position,
                                    const Eigen::Matrix3d& rotation,
                                    const MatrixType& G,
                                    const VectorType& h) {
  Eigen::Vector4d z;
  const Eigen::Vector3d body_center = position + rotation * body.center;
  z.head<3>() = 0.5 * (body_center + obstacle.center);
  z[3] = 1.0;
  double required_beta = 0.0;
  for (int row = 0; row + 1 < G.rows(); ++row) {
    const double radial = -G(row, 3);
    if (radial <= kMinimumRadialOffset) continue;
    required_beta = std::max(
        required_beta,
        (G.row(row).template head<3>().dot(z.head<3>()) - h[row]) /
            radial);
  }
  z[3] = std::max(1.0, required_beta + 1.0);
  if (!strictlyFeasible(G, h, z)) {
    z[3] = std::max(2.0, required_beta + 10.0);
  }
  return z;
}

ConvexPolytope jsonPolytope(const Json& encoded, double outward_offset) {
  ConvexPolytope result;
  const auto& halfspaces = encoded.at("halfspaces");
  const auto& vertices = encoded.at("vertices");
  result.A.resize(static_cast<int>(halfspaces.size()), 3);
  result.b.resize(static_cast<int>(halfspaces.size()));
  for (std::size_t row = 0; row < halfspaces.size(); ++row) {
    if (!halfspaces[row].is_array() || halfspaces[row].size() != 4) {
      throw std::runtime_error("PolyMap halfspace must contain four values");
    }
    for (int column = 0; column < 3; ++column) {
      result.A(static_cast<int>(row), column) = halfspaces[row][column].get<double>();
    }
    result.b[static_cast<int>(row)] = halfspaces[row][3].get<double>();
  }
  for (const auto& vertex : vertices) {
    if (!vertex.is_array() || vertex.size() != 3) {
      throw std::runtime_error("PolyMap vertex must contain three values");
    }
    result.vertices.emplace_back(vertex[0].get<double>(), vertex[1].get<double>(),
                                 vertex[2].get<double>());
  }
  result.finalize(outward_offset);
  return result;
}

ConvexPolytope boxPolytope(const Eigen::Vector3d& minimum,
                           const Eigen::Vector3d& maximum,
                           double outward_offset) {
  ConvexPolytope result;
  result.A.resize(6, 3);
  result.b.resize(6);
  result.A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
  result.b << maximum.x(), -minimum.x(), maximum.y(), -minimum.y(),
      maximum.z(), -minimum.z();
  for (int sx : {0, 1}) for (int sy : {0, 1}) for (int sz : {0, 1}) {
    result.vertices.emplace_back(sx ? maximum.x() : minimum.x(),
                                 sy ? maximum.y() : minimum.y(),
                                 sz ? maximum.z() : minimum.z());
  }
  result.finalize(outward_offset);
  return result;
}

Aabb transformedScaledAabb(const ConvexPolytope& polytope,
                           const Eigen::Vector3d& position, double yaw,
                           double scale) {
  Aabb result;
  const Eigen::Matrix3d rotation = yawRotation(yaw);
  for (const auto& vertex : polytope.vertices) {
    result.extend(position + rotation *
        (polytope.center + scale * (vertex - polytope.center)));
  }
  result.minimum.array() -= scale * polytope.aabb_padding;
  result.maximum.array() += scale * polytope.aabb_padding;
  return result;
}

}  // namespace

bool ConvexPolytope::valid(double tolerance) const {
  return A.cols() == 3 && A.rows() >= 4 && A.rows() == b.size() &&
         A.allFinite() && b.allFinite() && center.allFinite() &&
         radial_offsets.size() == A.rows() && radial_offsets.allFinite() &&
         radial_offsets.minCoeff() > tolerance;
}

void ConvexPolytope::finalize(double outward_offset) {
  if (A.cols() != 3 || A.rows() < 4 || A.rows() != b.size() ||
      !A.allFinite() || !b.allFinite() || vertices.size() < 4 ||
      !std::isfinite(outward_offset) || outward_offset < 0.0) {
    throw std::invalid_argument("invalid convex polytope H/V representation");
  }
  for (const auto& vertex : vertices) {
    if (!vertex.allFinite()) {
      throw std::invalid_argument("non-finite convex polytope vertex");
    }
  }
  for (int row = 0; row < A.rows(); ++row) {
    const double norm = A.row(row).norm();
    if (!(norm > 1.0e-12) || !std::isfinite(norm)) {
      throw std::invalid_argument("degenerate convex polytope half-space");
    }
    A.row(row) /= norm;
    b[row] = b[row] / norm + outward_offset;
  }
  // Convex-hull exporters commonly triangulate a planar face and emit one
  // identical H-row per triangle. Remove those redundant constraints once at
  // load time; this preserves the polytope exactly while greatly reducing the
  // 4-D proximity KKT system and its dual degeneracy.
  std::vector<Eigen::RowVector3d> unique_normals;
  std::vector<double> unique_offsets;
  unique_normals.reserve(A.rows());
  unique_offsets.reserve(A.rows());
  for (int row = 0; row < A.rows(); ++row) {
    bool duplicate = false;
    for (std::size_t kept = 0; kept < unique_normals.size(); ++kept) {
      if (A.row(row).dot(unique_normals[kept]) > 1.0 - 1.0e-9 &&
          std::abs(b[row] - unique_offsets[kept]) <= 1.0e-7) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      unique_normals.push_back(A.row(row));
      unique_offsets.push_back(b[row]);
    }
  }
  if (unique_normals.size() != static_cast<std::size_t>(A.rows())) {
    A.resize(static_cast<int>(unique_normals.size()), 3);
    b.resize(static_cast<int>(unique_offsets.size()));
    for (std::size_t row = 0; row < unique_normals.size(); ++row) {
      A.row(static_cast<int>(row)) = unique_normals[row];
      b[static_cast<int>(row)] = unique_offsets[row];
    }
  }
  aabb_padding = 0.0;
  center.setZero();
  for (const auto& vertex : vertices) center += vertex;
  center /= static_cast<double>(vertices.size());
  radial_offsets = b - A * center;
  if (!radial_offsets.allFinite() ||
      radial_offsets.minCoeff() <= kMinimumRadialOffset) {
    throw std::invalid_argument("polytope scaling centre is not strictly interior");
  }
  if (outward_offset > 0.0) {
    auto inflated_vertices = verticesFromHalfspaces(A, b, 1.0e-7);
    if (inflated_vertices.size() < 4) {
      throw std::invalid_argument(
          "failed to reconstruct outward-offset convex polytope");
    }
    vertices = std::move(inflated_vertices);
  }
}

ScalingProximitySolver::ScalingProximitySolver(ProximityOptions options)
    : options_(options) {
  if (!(options_.residual_tolerance > 0.0) ||
      !(options_.initial_barrier > options_.final_barrier) ||
      !(options_.final_barrier > 0.0) ||
      !(options_.barrier_reduction > 0.0 && options_.barrier_reduction < 1.0) ||
      !(options_.kkt_regularization >= 0.0) ||
      options_.max_newton_iterations <= 0 ||
      options_.max_line_search_iterations <= 0) {
    throw std::invalid_argument("invalid scaling proximity options");
  }
}

bool ScalingProximitySolver::minimumBeta(
    const ConvexPolytope& body, const ConvexPolytope& obstacle,
    const Eigen::Vector3d& position, double yaw, double& beta,
    ProximityWorkspace* workspace) const {
  if (!body.valid() || !obstacle.valid() || !position.allFinite() ||
      !std::isfinite(yaw)) {
    return false;
  }
  const Eigen::Matrix3d rotation = yawRotation(yaw);
  const int body_rows = body.A.rows();
  const int obstacle_rows = obstacle.A.rows();
  const int constraint_rows = body_rows + obstacle_rows + 1;
  ProximityWorkspace local_workspace;
  ProximityWorkspace& scratch = workspace ? *workspace : local_workspace;
  scratch.reserveRows(static_cast<std::size_t>(constraint_rows));
  auto constraints = scratch.constraints.topRows(constraint_rows);
  auto bounds = scratch.bounds.head(constraint_rows);
  constraints.setZero();
  bounds.setZero();
  for (int row = 0; row < body_rows; ++row) {
    const Eigen::Vector3d local_normal = body.A.row(row).transpose();
    const Eigen::Vector3d world_normal = rotation * local_normal;
    constraints.row(row).head<3>() = world_normal.transpose();
    constraints(row, 3) = -body.radial_offsets[row];
    bounds[row] =
        world_normal.dot(position) + local_normal.dot(body.center);
  }
  for (int row = 0; row < obstacle_rows; ++row) {
    const int target = body_rows + row;
    constraints.row(target).head<3>() = obstacle.A.row(row);
    constraints(target, 3) = -obstacle.radial_offsets[row];
    bounds[target] = obstacle.A.row(row).dot(obstacle.center);
  }
  constraints(constraint_rows - 1, 3) = -1.0;
  Eigen::Vector4d optimum;
  const Eigen::Vector4d objective(0.0, 0.0, 0.0, 1.0);
  if (!solveLinearProgram4(objective, constraints, bounds, scratch, optimum) ||
      !optimum.allFinite() || optimum[3] < -1.0e-8 ||
      (constraints * optimum - bounds).maxCoeff() > 1.0e-7) {
    return false;
  }
  beta = std::max(0.0, optimum[3]);
  return true;
}

ProximityResult ScalingProximitySolver::solve(
    const ConvexPolytope& body, const ConvexPolytope& obstacle,
    const Eigen::Vector3d& position, double yaw,
    const ProximityWarmStart* warm_start,
    ProximityWorkspace* workspace) const {
  ProximityResult result;
  if (!body.valid() || !obstacle.valid() || !position.allFinite() ||
      !std::isfinite(yaw)) {
    result.failure_stage = 1;
    return result;
  }

  const Eigen::Matrix3d rotation = yawRotation(yaw);
  const Eigen::Matrix3d rotation_derivative = yawRotationDerivative(yaw);
  const int body_rows = body.A.rows();
  const int obstacle_rows = obstacle.A.rows();
  const int constraint_rows = body_rows + obstacle_rows + 1;
  ProximityWorkspace local_workspace;
  ProximityWorkspace& scratch = workspace ? *workspace : local_workspace;
  scratch.reserveRows(static_cast<std::size_t>(constraint_rows));
  auto G = scratch.constraints.topRows(constraint_rows);
  auto h = scratch.bounds.head(constraint_rows);
  G.setZero();
  h.setZero();

  for (int row = 0; row < body_rows; ++row) {
    const Eigen::Vector3d local_normal = body.A.row(row).transpose();
    const Eigen::Vector3d world_normal = rotation * local_normal;
    G.row(row).head<3>() = world_normal.transpose();
    G(row, 3) = -body.radial_offsets[row];
    h[row] = world_normal.dot(position) + local_normal.dot(body.center);
  }
  for (int row = 0; row < obstacle_rows; ++row) {
    const int target = body_rows + row;
    G.row(target).head<3>() = obstacle.A.row(row);
    G(target, 3) = -obstacle.radial_offsets[row];
    h[target] = obstacle.A.row(row).dot(obstacle.center);
  }
  G(constraint_rows - 1, 3) = -1.0;  // beta >= 0

  const Eigen::Vector4d objective_gradient(0.0, 0.0, 0.0, 1.0);
  const auto finish = [&](const Eigen::Vector4d& optimum,
                          const Eigen::VectorXd& optimum_dual,
                          const std::array<int, 4>& active_rows,
                          int active_count,
                          double residual) {
    ProximityResult solved = result;
    solved.beta = std::max(0.0, optimum[3]);
    solved.contact = optimum.head<3>();
    solved.primal_residual = residual;
    for (int row = 0; row < body_rows; ++row) {
      const Eigen::Vector3d local_normal = body.A.row(row).transpose();
      const Eigen::Vector3d world_normal = rotation * local_normal;
      solved.gradient_position -= optimum_dual[row] * world_normal;
      const Eigen::Vector3d normal_derivative =
          rotation_derivative * local_normal;
      solved.gradient_yaw += optimum_dual[row] *
          normal_derivative.dot(solved.contact - position);
    }
    solved.success = solved.gradient_position.allFinite() &&
                     std::isfinite(solved.gradient_yaw) &&
                     std::isfinite(solved.beta);
    solved.warm_start.primal = optimum;
    solved.warm_start.active_rows = active_rows;
    solved.warm_start.active_count = active_count;
    solved.warm_start.valid = solved.success;
    return solved;
  };

  // The active basis is locally constant over most neighboring trajectory
  // samples.  A primal/dual KKT certificate makes this an exact LP fast path,
  // not an approximation: if every constraint is feasible and lambda >= 0,
  // the candidate is globally optimal.
  if (warm_start && warm_start->valid && warm_start->active_count == 4) {
    Eigen::Matrix4d active;
    Eigen::Vector4d active_rhs;
    bool indices_valid = true;
    for (int index = 0; index < 4; ++index) {
      const int source = warm_start->active_rows[index];
      if (source < 0 || source >= constraint_rows) {
        indices_valid = false;
        break;
      }
      for (int previous = 0; previous < index; ++previous) {
        if (warm_start->active_rows[previous] == source) indices_valid = false;
      }
      active.row(index) = G.row(source);
      active_rhs[index] = h[source];
    }
    if (indices_valid) {
      Eigen::FullPivLU<Eigen::Matrix4d> factor(active);
      if (factor.rank() == 4) {
        const Eigen::Vector4d candidate = factor.solve(active_rhs);
        const Eigen::Vector4d lambda =
            active.transpose().fullPivLu().solve(-objective_gradient);
        const double primal_violation =
            (G * candidate - h).maxCoeff();
        const double dual_residual =
            (objective_gradient + active.transpose() * lambda)
                .lpNorm<Eigen::Infinity>();
        const double certificate_tolerance =
            std::max(1.0e-7, 100.0 * options_.residual_tolerance);
        if (candidate.allFinite() && candidate[3] >= -1.0e-8 &&
            lambda.allFinite() && lambda.minCoeff() >= -1.0e-9 &&
            primal_violation <= 1.0e-7 &&
            dual_residual <= certificate_tolerance) {
          Eigen::VectorXd full_dual = Eigen::VectorXd::Zero(constraint_rows);
          for (int index = 0; index < 4; ++index) {
            full_dual[warm_start->active_rows[index]] =
                std::max(0.0, lambda[index]);
          }
          return finish(candidate, full_dual, warm_start->active_rows, 4,
                        std::max(std::max(0.0, primal_violation), dual_residual));
        }
      }
    }
  }

  Eigen::Vector4d z;
  if (warm_start && warm_start->valid && warm_start->primal.allFinite()) {
    z = warm_start->primal;
    // The polished LP solution lies on its active planes and therefore is not
    // strictly feasible for a log barrier.  Inflate beta by the smallest
    // amount that gives every scaling plane a small positive slack; this keeps
    // the previous contact point and makes the cached primal a real IPM warm
    // start instead of silently falling back to a cold initialization.
    const double target_slack =
        std::max(1.0e-8, 10.0 * options_.residual_tolerance);
    double beta_increase = 0.0;
    const Eigen::VectorXd warm_slack = h - G * z;
    for (int row = 0; row + 1 < constraint_rows; ++row) {
      const double radial = -G(row, 3);
      if (radial > kMinimumRadialOffset && warm_slack[row] < target_slack) {
        beta_increase = std::max(
            beta_increase, (target_slack - warm_slack[row]) / radial);
      }
    }
    z[3] += beta_increase;
    if (!strictlyFeasible(G, h, z)) {
      z = feasibleInitialPoint(body, obstacle, position, rotation, G, h);
    }
  } else {
    z = feasibleInitialPoint(body, obstacle, position, rotation, G, h);
  }
  if (!strictlyFeasible(G, h, z)) { result.failure_stage = 2; return result; }

  Eigen::VectorXd dual;
  Eigen::Vector4d sdlp_optimum;
  result.used_sdlp = options_.use_sdlp;
  const bool sdlp_success = options_.use_sdlp &&
      solveLinearProgram4(objective_gradient, G, h, scratch, sdlp_optimum) &&
      sdlp_optimum[3] >= -1.0e-8 &&
      (G * sdlp_optimum - h).maxCoeff() <= 1.0e-7;
  if (sdlp_success) {
    z = sdlp_optimum;
  } else {
    result.used_legacy_solver = true;
    double barrier = options_.initial_barrier;
    while (barrier >= options_.final_barrier * (1.0 - 1.0e-12)) {
      for (int iteration = 0; iteration < options_.max_newton_iterations;
           ++iteration) {
      ++result.newton_iterations;
      const Eigen::VectorXd slack = h - G * z;
      if ((slack.array() <= 0.0).any() || !slack.allFinite()) {
        result.failure_stage = 3; return result;
      }
      const Eigen::VectorXd inverse_slack = slack.cwiseInverse();
      const Eigen::Vector4d gradient =
          objective_gradient + barrier * G.transpose() * inverse_slack;
      Eigen::Matrix4d hessian =
          barrier * G.transpose() * inverse_slack.cwiseAbs2().asDiagonal() * G;
      Eigen::LDLT<Eigen::Matrix4d> factor;
      double regularization = std::max(options_.kkt_regularization, 1.0e-12);
      bool factorized = false;
      for (int attempt = 0; attempt < 8; ++attempt) {
        Eigen::Matrix4d regularized = hessian;
        regularized.diagonal().array() += regularization;
        factor.compute(regularized);
        if (factor.info() == Eigen::Success) {
          factorized = true;
          break;
        }
        regularization *= 100.0;
      }
      if (!factorized) { result.failure_stage = 4; return result; }
      const Eigen::Vector4d step = factor.solve(-gradient);
      if (!step.allFinite()) { result.failure_stage = 5; return result; }
      if (gradient.lpNorm<Eigen::Infinity>() <= options_.residual_tolerance &&
          step.norm() <= options_.residual_tolerance) {
        break;
      }

      double alpha = 1.0;
      bool accepted = false;
      for (int line = 0; line < options_.max_line_search_iterations; ++line) {
        const Eigen::Vector4d candidate = z + alpha * step;
        if (strictlyFeasible(G, h, candidate)) {
          const Eigen::VectorXd candidate_slack = h - G * candidate;
          const double current_value = z[3] - barrier * slack.array().log().sum();
          const double candidate_value =
              candidate[3] - barrier * candidate_slack.array().log().sum();
          if (candidate_value <= current_value +
                                     1.0e-4 * alpha * gradient.dot(step)) {
            z = candidate;
            accepted = true;
            break;
          }
        }
        alpha *= 0.5;
      }
      // At very small barrier values numerical roundoff can prevent an
      // Armijo decrease even though the current iterate is already the best
      // strictly-feasible point. Move to the next barrier stage instead of
      // misclassifying that condition as an IPM failure.
      if (!accepted) break;
      }
      const Eigen::VectorXd slack = h - G * z;
      dual = barrier * slack.cwiseInverse();
      barrier *= options_.barrier_reduction;
    }
  }

  if (!z.allFinite() || !(z[3] >= 0.0)) {
    result.failure_stage = 6;
    return result;
  }
  // Recover the complete local optimum face from the LP candidate. A bounded
  // four-variable LP has an optimum supported by active planes; their slacks
  // reliably identify a small nearest-plane working set.
  Eigen::VectorXd barrier_slack = h - G * z;
  std::vector<int> working(static_cast<std::size_t>(constraint_rows));
  std::iota(working.begin(), working.end(), 0);
  std::sort(working.begin(), working.end(), [&](int lhs, int rhs) {
    return barrier_slack[lhs] < barrier_slack[rhs];
  });
  double best_beta = std::numeric_limits<double>::infinity();
  std::vector<Eigen::Vector4d> feasible_vertices;
  std::vector<std::array<int, 4>> feasible_bases;
  // Keep the full local face neighborhood.  Eight planes are usually enough
  // to recover beta, but can omit the opposite boundary of a degenerate
  // optimum face and thereby select a biased yaw subgradient.  Twelve planes
  // retain the symmetric minimum-norm behavior expected by the optimizer.
  constexpr std::size_t polish_limit_used = 12;
  const auto enumerate_vertices = [&](std::size_t requested_limit) {
    best_beta = std::numeric_limits<double>::infinity();
    feasible_vertices.clear();
    feasible_bases.clear();
    const std::size_t count = std::min(requested_limit, working.size());
    const std::uint64_t subsets = std::uint64_t{1} << count;
    for (std::uint64_t mask = 1; mask < subsets; ++mask) {
      if (__builtin_popcountll(mask) != 4) continue;
      Eigen::Matrix4d active;
      Eigen::Vector4d active_rhs;
      std::array<int, 4> active_indices{{-1, -1, -1, -1}};
      int row = 0;
      for (std::size_t bit = 0; bit < count; ++bit) {
        if ((mask & (std::uint64_t{1} << bit)) == 0) continue;
        active.row(row) = G.row(working[bit]);
        active_rhs[row] = h[working[bit]];
        active_indices[static_cast<std::size_t>(row)] = working[bit];
        ++row;
      }
      Eigen::FullPivLU<Eigen::Matrix4d> factor(active);
      if (factor.rank() != 4) continue;
      const Eigen::Vector4d candidate = factor.solve(active_rhs);
      if (!candidate.allFinite() || candidate[3] < -1.0e-8 ||
          (G * candidate - h).maxCoeff() > 1.0e-7) {
        continue;
      }
      best_beta = std::min(best_beta, candidate[3]);
      feasible_vertices.push_back(candidate);
      feasible_bases.push_back(active_indices);
    }
    return std::isfinite(best_beta) &&
           (!result.used_sdlp ||
            std::abs(best_beta - sdlp_optimum[3]) <= 1.0e-6);
  };

  // An arbitrary SDLP optimum may lie anywhere on a degenerate face.  The
  // twelve-plane neighborhood recovers that face's boundary vertices before
  // selecting the stable averaged contact and dual.
  const bool polished = enumerate_vertices(polish_limit_used);
  if (!polished) {
    result.failure_stage = 7;
    return result;
  }
  z.setZero();
  int optimum_vertex_count = 0;
  std::array<int, 4> fallback_active_rows{{-1, -1, -1, -1}};
  for (std::size_t candidate_index = 0;
       candidate_index < feasible_vertices.size(); ++candidate_index) {
    const auto& candidate = feasible_vertices[candidate_index];
    if (candidate[3] <= best_beta + 1.0e-7) {
      z += candidate;
      if (optimum_vertex_count == 0) {
        fallback_active_rows = feasible_bases[candidate_index];
      }
      ++optimum_vertex_count;
    }
  }
  z /= static_cast<double>(optimum_vertex_count);

  const Eigen::VectorXd polished_slack = h - G * z;
  std::vector<int> active_rows;
  for (int row = 0; row < constraint_rows; ++row) {
    if (polished_slack[row] <= 1.0e-6) active_rows.push_back(row);
  }
  const std::size_t active_limit = polish_limit_used;
  if (active_rows.size() > active_limit) {
    std::sort(active_rows.begin(), active_rows.end(), [&](int lhs, int rhs) {
      return polished_slack[lhs] < polished_slack[rhs];
    });
    active_rows.resize(active_limit);
  }
  double best_residual = std::numeric_limits<double>::infinity();
  Eigen::VectorXd dual_sum = Eigen::VectorXd::Zero(constraint_rows);
  std::size_t dual_count = 0;
  constexpr double kDualTieTolerance = 1.0e-9;
  const std::uint64_t active_subsets = std::uint64_t{1} << active_rows.size();
  for (std::uint64_t mask = 1; mask < active_subsets; ++mask) {
    const int count = static_cast<int>(__builtin_popcountll(mask));
    if (count > 4) continue;
    Eigen::MatrixXd active(4, count);
    std::vector<int> rows;
    int column = 0;
    for (std::size_t bit = 0; bit < active_rows.size(); ++bit) {
      if ((mask & (std::uint64_t{1} << bit)) == 0) continue;
      rows.push_back(active_rows[bit]);
      active.col(column++) = G.row(active_rows[bit]).transpose();
    }
    Eigen::VectorXd lambda =
        active.colPivHouseholderQr().solve(-objective_gradient);
    if (!lambda.allFinite() || lambda.minCoeff() < -1.0e-9) continue;
    lambda = lambda.cwiseMax(0.0);
    const double residual =
        (objective_gradient + active * lambda).lpNorm<Eigen::Infinity>();
    if (residual + kDualTieTolerance < best_residual) {
      best_residual = residual;
      dual_sum.setZero();
      dual_count = 0;
    } else if (residual > best_residual + kDualTieTolerance) {
      continue;
    } else {
      best_residual = std::min(best_residual, residual);
    }
    for (int index = 0; index < count; ++index) {
      dual_sum[rows[index]] += lambda[index];
    }
    ++dual_count;
  }
  if (dual_count == 0) {
    result.failure_stage = 7;
    return result;
  }
  dual = dual_sum / static_cast<double>(dual_count);
  result.primal_residual = best_residual;
  // Expose and validate the recovered KKT stationarity residual.
  if (result.primal_residual > 1.0e5 * options_.residual_tolerance) {
    result.failure_stage = 7; return result;
  }

  result.beta = z[3];
  result.contact = z.head<3>();
  for (int row = 0; row < body_rows; ++row) {
    const Eigen::Vector3d local_normal = body.A.row(row).transpose();
    const Eigen::Vector3d world_normal = rotation * local_normal;
    result.gradient_position -= dual[row] * world_normal;
    const Eigen::Vector3d normal_derivative =
        rotation_derivative * local_normal;
    result.gradient_yaw +=
        dual[row] * normal_derivative.dot(result.contact - position);
  }
  result.success = result.gradient_position.allFinite() &&
                   std::isfinite(result.gradient_yaw) &&
                   std::isfinite(result.beta);
  result.warm_start.primal = z;
  result.warm_start.active_rows = fallback_active_rows;
  // Do not collapse a degenerate optimum face to one arbitrary vertex.  The
  // legacy solver deliberately averages tied optimal vertices to obtain a
  // stable symmetric subgradient; only unique vertices are eligible for the
  // active-basis fast path on subsequent poses.
  result.warm_start.active_count = optimum_vertex_count == 1 ? 4 : 0;
  result.warm_start.valid = result.success;
  return result;
}

PairPenalty exponentialHinge(double beta,
                             const ExponentialHingeOptions& options) {
  PairPenalty result;
  if (!std::isfinite(beta) || !(options.beta_soft >= 1.0) ||
      !(options.weight >= 0.0) || !(options.exponential_gain >= 0.0) ||
      !(options.hard_weight_multiplier >= 0.0) ||
      !(options.maximum_exponent > 0.0)) {
    return result;
  }
  const auto add_term = [&](double error, double weight) {
    if (!(error > 0.0) || !(weight > 0.0)) return;
    const double exponent = std::min(
        options.maximum_exponent, options.exponential_gain * error);
    const double exponential = std::exp(exponent);
    const double error2 = error * error;
    result.cost += weight * exponential * error2;
    const double effective_gain =
        exponent < options.maximum_exponent ? options.exponential_gain : 0.0;
    result.derivative_beta -=
        weight * exponential * (2.0 * error + effective_gain * error2);
    result.active = true;
  };
  add_term(options.beta_soft - beta, options.weight);
  if (beta < options.beta_soft && options.hard_weight_multiplier > 0.0) {
    const double slack = beta - options.acceptance_beta;
    constexpr double kBarrierRelaxation = 2.0e-2;
    double reciprocal = 0.0;
    double reciprocal_derivative = 0.0;
    if (slack >= kBarrierRelaxation) {
      reciprocal = 1.0 / slack;
      reciprocal_derivative = -reciprocal * reciprocal;
    } else {
      const double delta = slack - kBarrierRelaxation;
      const double inverse_relaxation = 1.0 / kBarrierRelaxation;
      const double inverse_relaxation2 =
          inverse_relaxation * inverse_relaxation;
      const double inverse_relaxation3 =
          inverse_relaxation2 * inverse_relaxation;
      reciprocal = inverse_relaxation - delta * inverse_relaxation2 +
                   delta * delta * inverse_relaxation3;
      reciprocal_derivative = -inverse_relaxation2 +
                              2.0 * delta * inverse_relaxation3;
    }
    const double error = options.beta_soft - beta;
    const double barrier_weight =
        options.weight * options.hard_weight_multiplier;
    result.cost += barrier_weight * error * error * reciprocal;
    result.derivative_beta += barrier_weight *
        (-2.0 * error * reciprocal +
         error * error * reciprocal_derivative);
    result.active = true;
  }
  return result;
}

bool Aabb::valid() const {
  return minimum.allFinite() && maximum.allFinite() &&
         (minimum.array() <= maximum.array()).all();
}

bool Aabb::overlaps(const Aabb& other) const {
  return valid() && other.valid() &&
         (minimum.array() <= other.maximum.array()).all() &&
         (other.minimum.array() <= maximum.array()).all();
}

void Aabb::extend(const Eigen::Vector3d& point) {
  minimum = minimum.cwiseMin(point);
  maximum = maximum.cwiseMax(point);
}

void Aabb::extend(const Aabb& other) {
  if (!other.valid()) return;
  extend(other.minimum);
  extend(other.maximum);
}

SafetyScene SafetyScene::load(const std::string& map_polymap_path,
                              const std::string& body_polymap_path,
                              double body_clearance,
                              double virtual_ceiling_height,
                              double beta_soft) {
  if (!(body_clearance >= 0.0) || !std::isfinite(body_clearance) ||
      !std::isfinite(virtual_ceiling_height) || !(beta_soft >= 1.0) ||
      !std::isfinite(beta_soft)) {
    throw std::invalid_argument("invalid whole-body safety scene parameters");
  }
  const auto load_json = [](const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("failed to open PolyMap: " + path);
    Json document;
    input >> document;
    return document;
  };
  const Json map = load_json(map_polymap_path);
  const Json body = load_json(body_polymap_path);
  if (map.value("format", "") != "mighty.polymap" ||
      map.value("kind", "") != "map" ||
      body.value("format", "") != "mighty.polymap" ||
      body.value("kind", "") != "shape") {
    throw std::runtime_error("whole-body safety requires map and shape PolyMaps");
  }

  SafetyScene result;
  result.bvh_scale_ = beta_soft;
  for (const auto& encoded : body.at("polyhedra")) {
    result.body_components_.push_back(
        jsonPolytope(encoded, body_clearance));
  }
  for (const auto& encoded : map.at("polyhedra")) {
    result.obstacle_components_.push_back(jsonPolytope(encoded, 0.0));
  }
  if (result.body_components_.empty() || result.obstacle_components_.empty()) {
    throw std::runtime_error("whole-body safety PolyMap contains no components");
  }

  const Json& grid = map.at("grid");
  const auto vector3 = [](const Json& value) {
    return Eigen::Vector3d(value.at(0).get<double>(), value.at(1).get<double>(),
                           value.at(2).get<double>());
  };
  const Eigen::Vector3d minimum = vector3(grid.at("min"));
  Eigen::Vector3d maximum = vector3(grid.at("max"));
  maximum.z() = std::min(maximum.z(), virtual_ceiling_height);
  if ((minimum.array() >= maximum.array()).any()) {
    throw std::runtime_error("invalid PolyMap safety bounds");
  }
  // Keep virtual walls thin in their normal direction. Scaling a very thick
  // wall about its centre would move its inner face far into the map for
  // beta_soft > 1 and create false broad-phase candidates.
  const double thickness = std::max(1.0, 2.0 * body_clearance);
  const Eigen::Vector3d outer_min = minimum.array() - thickness;
  const Eigen::Vector3d outer_max = maximum.array() + thickness;
  result.obstacle_components_.push_back(boxPolytope(
      {outer_min.x(), outer_min.y(), outer_min.z()},
      {minimum.x(), outer_max.y(), outer_max.z()}, 0.0));
  result.obstacle_components_.push_back(boxPolytope(
      {maximum.x(), outer_min.y(), outer_min.z()},
      {outer_max.x(), outer_max.y(), outer_max.z()}, 0.0));
  result.obstacle_components_.push_back(boxPolytope(
      {outer_min.x(), outer_min.y(), outer_min.z()},
      {outer_max.x(), minimum.y(), outer_max.z()}, 0.0));
  result.obstacle_components_.push_back(boxPolytope(
      {outer_min.x(), maximum.y(), outer_min.z()},
      {outer_max.x(), outer_max.y(), outer_max.z()}, 0.0));
  result.obstacle_components_.push_back(boxPolytope(
      {outer_min.x(), outer_min.y(), outer_min.z()},
      {outer_max.x(), outer_max.y(), minimum.z()}, 0.0));
  result.obstacle_components_.push_back(boxPolytope(
      {outer_min.x(), outer_min.y(), maximum.z()},
      {outer_max.x(), outer_max.y(), outer_max.z()}, 0.0));

  result.obstacle_bounds_.reserve(result.obstacle_components_.size());
  for (const auto& obstacle : result.obstacle_components_) {
    result.obstacle_bounds_.push_back(transformedScaledAabb(
        obstacle, Eigen::Vector3d::Zero(), 0.0, beta_soft));
  }
  std::vector<int> indices(result.obstacle_components_.size());
  std::iota(indices.begin(), indices.end(), 0);
  result.bvh_root_ = result.buildBvh(indices, 0, indices.size());
  std::size_t max_body_rows = 0;
  for (const auto& component : result.body_components_) {
    max_body_rows = std::max(
        max_body_rows, static_cast<std::size_t>(component.A.rows()));
  }
  std::size_t max_obstacle_rows = 0;
  for (const auto& component : result.obstacle_components_) {
    max_obstacle_rows = std::max(
        max_obstacle_rows, static_cast<std::size_t>(component.A.rows()));
  }
  result.max_constraint_rows_ = max_body_rows + max_obstacle_rows + 1;
  return result;
}

int SafetyScene::buildBvh(std::vector<int>& indices, std::size_t begin,
                          std::size_t end) {
  const int node_index = static_cast<int>(bvh_.size());
  bvh_.push_back({});
  for (std::size_t i = begin; i < end; ++i) {
    bvh_[node_index].bounds.extend(obstacle_bounds_[indices[i]]);
  }
  if (end - begin == 1) {
    bvh_[node_index].obstacle = indices[begin];
    return node_index;
  }
  const Eigen::Vector3d span = bvh_[node_index].bounds.maximum -
                               bvh_[node_index].bounds.minimum;
  Eigen::Index axis = 0;
  span.maxCoeff(&axis);
  const std::size_t middle = begin + (end - begin) / 2;
  std::nth_element(indices.begin() + static_cast<std::ptrdiff_t>(begin),
                   indices.begin() + static_cast<std::ptrdiff_t>(middle),
                   indices.begin() + static_cast<std::ptrdiff_t>(end),
                   [&](int lhs, int rhs) {
                     return (obstacle_bounds_[lhs].minimum[axis] +
                             obstacle_bounds_[lhs].maximum[axis]) <
                            (obstacle_bounds_[rhs].minimum[axis] +
                             obstacle_bounds_[rhs].maximum[axis]);
                   });
  const int left = buildBvh(indices, begin, middle);
  const int right = buildBvh(indices, middle, end);
  bvh_[node_index].left = left;
  bvh_[node_index].right = right;
  return node_index;
}

void SafetyScene::queryBvh(int node, const Aabb& query,
                           std::vector<int>& result) const {
  if (node < 0 || !bvh_[node].bounds.overlaps(query)) return;
  if (bvh_[node].obstacle >= 0) {
    result.push_back(bvh_[node].obstacle);
    return;
  }
  queryBvh(bvh_[node].left, query, result);
  queryBvh(bvh_[node].right, query, result);
}

std::vector<CandidatePair> SafetyScene::candidates(
    const Eigen::Vector3d& position, double yaw, double beta_soft) const {
  std::vector<CandidatePair> result;
  std::vector<int> obstacle_scratch;
  candidates(position, yaw, beta_soft, result, obstacle_scratch);
  return result;
}

void SafetyScene::candidates(
    const Eigen::Vector3d& position, double yaw, double beta_soft,
    std::vector<CandidatePair>& result,
    std::vector<int>& obstacle_scratch) const {
  result.clear();
  if (!position.allFinite() || !std::isfinite(yaw) || !(beta_soft >= 1.0) ||
      beta_soft > bvh_scale_ + 1.0e-12) {
    return;
  }
  for (std::size_t body_index = 0; body_index < body_components_.size();
       ++body_index) {
    const Aabb query = transformedScaledAabb(
        body_components_[body_index], position, yaw, beta_soft);
    obstacle_scratch.clear();
    queryBvh(bvh_root_, query, obstacle_scratch);
    for (int obstacle : obstacle_scratch) {
      result.push_back({body_index, static_cast<std::size_t>(obstacle)});
    }
  }
}

WholeBodySafety::WholeBodySafety(
    std::shared_ptr<const SafetyScene> scene,
    ExponentialHingeOptions penalty_options,
    ProximityOptions warm_options,
    ProximityOptions cold_options)
    : scene_(std::move(scene)),
      penalty_options_(penalty_options),
      warm_solver_(warm_options),
      cold_solver_(cold_options) {
  if (!scene_) throw std::invalid_argument("whole-body safety scene is null");
  if (!(penalty_options_.acceptance_beta > 0.0) ||
      !(penalty_options_.acceptance_beta <= 1.0) ||
      !(penalty_options_.acceptance_beta <= penalty_options_.beta_soft)) {
    throw std::invalid_argument("invalid whole-body safety acceptance beta");
  }
}

void SafetyQueryContext::clearWarmStarts() {
  warm_starts.clear();
}

void SafetyStatistics::merge(const SafetyStatistics& other) {
  candidate_pairs += other.candidate_pairs;
  proximity_queries += other.proximity_queries;
  active_pairs += other.active_pairs;
  cold_retries += other.cold_retries;
  failed_queries += other.failed_queries;
  total_newton_iterations += other.total_newton_iterations;
  active_set_fast_hits += other.active_set_fast_hits;
  active_set_fast_misses += other.active_set_fast_misses;
  legacy_solver_calls += other.legacy_solver_calls;
  sdlp_calls += other.sdlp_calls;
  sdlp_failures += other.sdlp_failures;
  if (other.last_failure_stage != 0) {
    last_failure_stage = other.last_failure_stage;
  }
  if (other.minimum_beta < minimum_beta) {
    minimum_beta = other.minimum_beta;
    minimum_beta_body = other.minimum_beta_body;
    minimum_beta_obstacle = other.minimum_beta_obstacle;
  }
}

std::size_t WholeBodySafety::pairIndex(const CandidatePair& pair) const {
  return pair.body * scene_->obstacleCount() + pair.obstacle;
}

SafetyQueryContext WholeBodySafety::createContext() const {
  SafetyQueryContext context;
  context.warm_starts.reserve(scene_->bodyCount() * 16);
  context.candidate_pairs.reserve(scene_->bodyCount() * 16);
  context.obstacle_scratch.reserve(32);
  context.proximity_workspace.reserveRows(scene_->maxConstraintRows());
  return context;
}

PoseSafetyResult WholeBodySafety::evaluate(
    const Eigen::Vector3d& position, double yaw, SafetyQueryContext& context,
    bool stop_on_hard_collision) const {
  return evaluateWithSolvers(position, yaw, context, stop_on_hard_collision,
                             warm_solver_, cold_solver_, true);
}

PoseSafetyResult WholeBodySafety::evaluateForOptimization(
    const Eigen::Vector3d& position, double yaw,
    SafetyQueryContext& context) const {
  PoseSafetyResult result;
  scene_->candidates(position, yaw, penalty_options_.beta_soft,
                     context.candidate_pairs, context.obstacle_scratch);
  result.statistics.candidate_pairs = context.candidate_pairs.size();
  for (const CandidatePair& pair : context.candidate_pairs) {
    const auto query_beta = [&](const Eigen::Vector3d& query_position,
                                double query_yaw, double& beta) {
      ++result.statistics.proximity_queries;
      ++result.statistics.sdlp_calls;
      if (cold_solver_.minimumBeta(
              scene_->body(pair.body), scene_->obstacle(pair.obstacle),
              query_position, query_yaw, beta, &context.proximity_workspace)) {
        return true;
      }
      result.success = false;
      ++result.statistics.failed_queries;
      ++result.statistics.sdlp_failures;
      result.statistics.last_failure_stage = 8;
      return false;
    };
    double beta = 0.0;
    if (!query_beta(position, yaw, beta)) continue;
    if (beta < result.statistics.minimum_beta) {
      result.statistics.minimum_beta = beta;
      result.statistics.minimum_beta_body = pair.body;
      result.statistics.minimum_beta_obstacle = pair.obstacle;
    }
    const PairPenalty penalty = exponentialHinge(beta, penalty_options_);
    result.cost += penalty.cost;
    if (!penalty.active) continue;
    ++result.statistics.active_pairs;

    // Optimization only needs the LP value and its pose derivative.  Recovering
    // and averaging every vertex/dual of a degenerate contact face is much more
    // expensive than these small LPs, and still needs numerical yaw derivatives.
    // Differentiate the exact value on all four axes; final whole-body checks
    // retain their separate cold primal/dual certification path.
    const double derivative_step = context.optimization_derivative_step;
    for (int axis = 0; axis < 4; ++axis) {
      Eigen::Vector3d plus_position = position;
      Eigen::Vector3d minus_position = position;
      double plus_yaw = yaw;
      double minus_yaw = yaw;
      if (axis < 3) {
        plus_position[axis] += derivative_step;
        minus_position[axis] -= derivative_step;
      } else {
        plus_yaw += derivative_step;
        minus_yaw -= derivative_step;
      }
      double plus_beta = 0.0;
      double minus_beta = 0.0;
      if (!query_beta(plus_position, plus_yaw, plus_beta) ||
          !query_beta(minus_position, minus_yaw, minus_beta)) {
        continue;
      }
      const double derivative = penalty.derivative_beta *
          (plus_beta - minus_beta) / (2.0 * derivative_step);
      if (axis < 3) result.gradient_position[axis] += derivative;
      else result.gradient_yaw += derivative;
    }
  }
  result.success = result.success && std::isfinite(result.cost) &&
                   result.gradient_position.allFinite() &&
                   std::isfinite(result.gradient_yaw);
  context.cumulative_statistics.merge(result.statistics);
  return result;
}

PoseSafetyResult WholeBodySafety::evaluateWithSolvers(
    const Eigen::Vector3d& position, double yaw, SafetyQueryContext& context,
    bool stop_on_hard_collision, const ScalingProximitySolver& primary,
    const ScalingProximitySolver& retry, bool allow_warm_start) const {
  PoseSafetyResult result;
  scene_->candidates(position, yaw, penalty_options_.beta_soft,
                     context.candidate_pairs, context.obstacle_scratch);
  result.statistics.candidate_pairs = context.candidate_pairs.size();

  const auto merge_statistics = [&]() {
    context.cumulative_statistics.merge(result.statistics);
  };

  for (const CandidatePair& pair : context.candidate_pairs) {
    const std::size_t key = pairIndex(pair);
    ProximityWarmStart& warm = context.warm_starts[key];
    const ProximityWarmStart* warm_start =
        allow_warm_start && warm.valid ? &warm : nullptr;
    const bool attempted_active_set =
        warm_start && warm_start->active_count == 4;
    ProximityResult proximity = warm_start
        ? primary.solve(scene_->body(pair.body), scene_->obstacle(pair.obstacle),
                        position, yaw, warm_start,
                        &context.proximity_workspace)
        : retry.solve(scene_->body(pair.body), scene_->obstacle(pair.obstacle),
                      position, yaw, nullptr, &context.proximity_workspace);
    ++result.statistics.proximity_queries;
    result.statistics.total_newton_iterations += proximity.newton_iterations;
    if (proximity.used_sdlp) ++result.statistics.sdlp_calls;
    if (proximity.used_sdlp && proximity.used_legacy_solver) {
      ++result.statistics.sdlp_failures;
    }
    if (attempted_active_set && proximity.success &&
        proximity.newton_iterations == 0) {
      ++result.statistics.active_set_fast_hits;
    } else {
      if (attempted_active_set) ++result.statistics.active_set_fast_misses;
      if (proximity.used_legacy_solver) {
        ++result.statistics.legacy_solver_calls;
      }
    }
    if (!proximity.success && warm_start) {
      ++result.statistics.cold_retries;
      proximity = retry.solve(
          scene_->body(pair.body), scene_->obstacle(pair.obstacle), position,
          yaw, nullptr, &context.proximity_workspace);
      ++result.statistics.proximity_queries;
      result.statistics.total_newton_iterations += proximity.newton_iterations;
      if (proximity.used_sdlp) ++result.statistics.sdlp_calls;
      if (proximity.used_sdlp && proximity.used_legacy_solver) {
        ++result.statistics.sdlp_failures;
      }
      if (proximity.used_legacy_solver) {
        ++result.statistics.legacy_solver_calls;
      }
    }
    if (!proximity.success) {
      result.success = false;
      ++result.statistics.failed_queries;
      result.statistics.last_failure_stage = proximity.failure_stage;
      continue;
    }
    warm = proximity.warm_start;
    if (proximity.beta < result.statistics.minimum_beta) {
      result.statistics.minimum_beta = proximity.beta;
      result.statistics.minimum_beta_body = pair.body;
      result.statistics.minimum_beta_obstacle = pair.obstacle;
    }
    if (stop_on_hard_collision &&
        proximity.beta < penalty_options_.acceptance_beta) {
      merge_statistics();
      return result;
    }
    const PairPenalty penalty = exponentialHinge(proximity.beta,
                                                 penalty_options_);
    if (!penalty.active) continue;
    ++result.statistics.active_pairs;
    result.cost += penalty.cost;
    result.gradient_position +=
        penalty.derivative_beta * proximity.gradient_position;
    result.gradient_yaw += penalty.derivative_beta * proximity.gradient_yaw;
  }
  result.success = result.success && std::isfinite(result.cost) &&
                   result.gradient_position.allFinite() &&
                   std::isfinite(result.gradient_yaw);
  merge_statistics();
  return result;
}

PoseSafetyResult WholeBodySafety::evaluate(const Eigen::Vector3d& position,
                                           double yaw) const {
  SafetyQueryContext context = createContext();
  return evaluate(position, yaw, context);
}

}  // namespace mighty::whole_body_safety
