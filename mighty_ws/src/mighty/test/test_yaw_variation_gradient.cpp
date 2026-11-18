#include <mighty/lbfgs_solver.hpp>

#include <cmath>
#include <iostream>

bool checkMode(bool spatial) {
  lbfgs::planner_params_t parameters;
  parameters.time_weight = 500.0;
  parameters.yaw_reference_weight = 10.0;
  parameters.spatial_progress_yaw_coupling = spatial;
  parameters.topp_init = false;
  parameters.light_activation_ladder = false;
  lbfgs::SolverLBFGS solver;
  solver.initializeSolver(parameters);
  state start{}, goal{};
  start.pos = {0.0, 0.0, 1.0};
  goal.pos = {2.0, 0.0, 1.0};
  double elapsed = 0.0;
  solver.prepareSolverForReplan(0.0, {start.pos, {1.0, 0.2, 1.0}, goal.pos},
                                start, goal, {0.0, 0.0, 0.0}, false, elapsed);
  auto z = solver.getInitialGuesses().at(0);
  // Two segments: nine interior position/PVA variables precede the yaw knot.
  z[9] = 1.8;
  z[10] = 0.3;
  z[11] = -0.2;
  z[12] = -0.7;
  Eigen::VectorXd gradient;
  const double before = solver.evaluateObjectiveAndGradientFused(z, gradient);
  double error = solver.checkGradDirectional(z, 24, 1.0e-5, 42);
  solver.reconstructPVATCPopt(z);
  if (!solver.refineYawVariationPenaltyForCurrentTrajectory()) return false;
  const double after = solver.evaluateObjectiveAndGradientFused(z, gradient);
  error = std::max(error, solver.checkGradDirectional(z, 24, 1.0e-5, 43));
  std::cout << (spatial ? "spatial" : "time_phase")
            << " yaw variation gradient relative error=" << error << '\n';
  return std::isfinite(error) && error < 1.0e-4 && after > before;
}

int main() {
  return checkMode(true) && checkMode(false) ? 0 : 1;
}
