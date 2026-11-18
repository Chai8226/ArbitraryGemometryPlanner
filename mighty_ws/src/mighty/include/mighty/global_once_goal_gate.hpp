#pragma once

#include <atomic>
#include <cstdint>

namespace mighty {

enum class GlobalOncePhase : std::uint8_t { IDLE, PLANNING, EXECUTING };

enum class GlobalOnceGoalStart : std::uint8_t {
  STARTED,
  SUPERSEDED_PLANNING,
  SUPERSEDED_EXECUTION,
};

inline GlobalOnceGoalStart beginGlobalOnceGoal(
    std::atomic<GlobalOncePhase>& phase) {
  const GlobalOncePhase previous = phase.exchange(GlobalOncePhase::PLANNING);
  if (previous == GlobalOncePhase::IDLE) {
    return GlobalOnceGoalStart::STARTED;
  }
  return previous == GlobalOncePhase::EXECUTING
             ? GlobalOnceGoalStart::SUPERSEDED_EXECUTION
             : GlobalOnceGoalStart::SUPERSEDED_PLANNING;
}

inline bool completeGlobalOnceGoal(std::atomic<GlobalOncePhase>& phase) {
  GlobalOncePhase expected = GlobalOncePhase::EXECUTING;
  return phase.compare_exchange_strong(expected, GlobalOncePhase::IDLE);
}

}  // namespace mighty
