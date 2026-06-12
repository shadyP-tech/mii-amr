from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RosNodeWiringContext:
    Twist: Any
    NavPath: Any
    MarkerArray: Any
    LaserScan: Any
    Odometry: Any
    PoseWithCovarianceStamped: Any
    qos_profile_sensor_data: Any
    tf2_ros: Any
    time_sleep: Callable[[float], None]
    rviz_messages_available: Callable[[], bool]
    rviz_qos_profile: Callable[[], Any]


def initialize_ros_interfaces(node, args, context):
    node.pub = node.create_publisher(context.Twist, args.cmd_vel_topic, 10)
    node.rviz_path_pub = None
    node.rviz_waypoint_marker_pub = None
    node.rviz_obstacle_marker_pub = None
    if not args.no_rviz_visualization:
        if not context.rviz_messages_available():
            raise RuntimeError(
                "ROS RViz message types are unavailable. Source ROS 2 Humble "
                "before enabling RViz visualization."
            )
        rviz_qos = context.rviz_qos_profile()
        node.rviz_path_pub = node.create_publisher(
            context.NavPath,
            args.rviz_path_topic,
            rviz_qos,
        )
        node.rviz_waypoint_marker_pub = node.create_publisher(
            context.MarkerArray,
            args.rviz_waypoint_marker_topic,
            rviz_qos,
        )
        node.rviz_obstacle_marker_pub = node.create_publisher(
            context.MarkerArray,
            args.rviz_obstacle_marker_topic,
            rviz_qos,
        )
        if args.verbose:
            node.get_logger().info(
                "Publishing RViz visualization: "
                f"path={args.rviz_path_topic}, "
                f"waypoints={args.rviz_waypoint_marker_topic}, "
                f"obstacles={args.rviz_obstacle_marker_topic}"
            )
    node.scan_sub = node.create_subscription(
        context.LaserScan,
        args.scan_topic,
        node.scan_callback,
        context.qos_profile_sensor_data,
    )
    node.amcl_sub = node.create_subscription(
        context.PoseWithCovarianceStamped,
        args.amcl_topic,
        node.amcl_callback,
        10,
    )
    node.odom_sub = node.create_subscription(
        context.Odometry,
        args.odom_topic,
        node.odom_callback,
        context.qos_profile_sensor_data,
    )
    node.tf_buffer = context.tf2_ros.Buffer()
    node.tf_listener = context.tf2_ros.TransformListener(node.tf_buffer, node)
    context.time_sleep(1.0)
