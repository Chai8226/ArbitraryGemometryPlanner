#!/usr/bin/python3
"""Send one goal after both the planner and simulator are ready."""
import argparse
import math
import time
import rospy
from geometry_msgs.msg import PoseStamped
from dynus_interfaces.msg import State
from std_msgs.msg import String


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--namespace', default='NX01')
    p.add_argument('--goal', nargs=3, type=float, required=True)
    p.add_argument('--yaw', type=float)
    p.add_argument('--frame', default='map')
    p.add_argument('--timeout', type=float, default=45)
    args = p.parse_args(rospy.myargv()[1:])
    rospy.init_node('mighty_send_goal', anonymous=True)
    prefix = '/' + args.namespace.strip('/')
    ready = {'state': False, 'planner': False}
    subs = [rospy.Subscriber(prefix + '/state', State, lambda _: ready.update(state=True)),
            rospy.Subscriber(prefix + '/shape_aware_astar/status', String,
                lambda m: ready.update(planner='ready' in m.data.lower()))]
    pub = rospy.Publisher(prefix + '/term_goal', PoseStamped, queue_size=1)
    deadline = time.monotonic() + args.timeout
    while not rospy.is_shutdown():
        if all(ready.values()) and pub.get_num_connections():
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('timeout waiting for planner/state/goal subscriber')
        time.sleep(0.05)
    if rospy.is_shutdown():
        return
    msg = PoseStamped()
    msg.header.frame_id = args.frame
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = args.goal
    if args.yaw is not None:
        yaw = math.radians(args.yaw)
        msg.pose.orientation.z, msg.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    pub.publish(msg)
    time.sleep(0.5)  # let the TCP queue drain before this one-shot publisher exits
    for sub in subs:
        sub.unregister()
    rospy.loginfo('Goal sent to %s', prefix)


if __name__ == '__main__':
    main()
