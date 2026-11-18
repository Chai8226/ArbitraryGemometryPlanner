// SPDX-License-Identifier: GPL-3.0-only
#include <mighty/mighty_node.hpp>
#include <mighty/runtime_guards.hpp>
int main(int argc, char** argv) {
  if (mighty::armParentDeathSignal()) return 0;
  ros::init(argc, argv, "mighty_node");
  try {
    mighty::MIGHTY_NODE node;
    node.start();
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL("%s", error.what());
    return 1;
  }
  return 0;
}
