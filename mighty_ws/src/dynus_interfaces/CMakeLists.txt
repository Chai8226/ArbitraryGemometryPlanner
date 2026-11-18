cmake_minimum_required(VERSION 3.10)
project(dynus_interfaces)
find_package(catkin REQUIRED COMPONENTS message_generation std_msgs geometry_msgs)
add_message_files(FILES
  QuinticPWPTraj.msg
  PWPTraj.msg
  QuinticCoeffPoly3.msg
  CoeffPoly3.msg
  DynTraj.msg
  DynTrajArray.msg
  State.msg
  Goal.msg
  Trajectory.msg
  Mode.msg
  YawOutput.msg
  PNAdaptation.msg
  ComputationTimes.msg
  GlobalPlanStatus.msg
  SpeedyPath.msg
  Frontier.msg
  FrontierList.msg)
add_service_files(FILES FlightCommand.srv)
generate_messages(DEPENDENCIES std_msgs geometry_msgs)
catkin_package(CATKIN_DEPENDS message_runtime std_msgs geometry_msgs)
