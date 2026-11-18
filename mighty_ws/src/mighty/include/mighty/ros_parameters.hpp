#pragma once
#include <ros/ros.h>
#include <algorithm>
#include <stdexcept>
#include <string>
namespace mighty {
template<class T> T readParameter(const ros::NodeHandle& node, std::string key, const T& fallback) {
  std::replace(key.begin(), key.end(), '.', '/');
  T result;
  if (node.getParam(key, result)) return result;
  if (node.hasParam(key)) throw std::invalid_argument("invalid parameter type: " + node.resolveName(key));
  return fallback;
}
}
