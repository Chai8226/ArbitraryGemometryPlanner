// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <sys/prctl.h>
#include <unistd.h>
#include <csignal>
namespace mighty {
inline bool armParentDeathSignal() {
  prctl(PR_SET_PDEATHSIG, SIGINT);
  return getppid() == 1;
}
}  // namespace mighty
