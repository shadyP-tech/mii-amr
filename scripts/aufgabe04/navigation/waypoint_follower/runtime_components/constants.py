"""Shared constants for the ROS-facing waypoint-follower runtime."""

STALE_TF_RECOVERY_MAX_DURATION_SEC = 0.18
STALE_TF_RECOVERY_MAX_CALLBACKS = 48
STALE_TF_RECOVERY_SPIN_TIMEOUT_SEC = 0.005
AMCL_STALE_TF_RECOVERY_POLL_SEC = 0.05
SIMULATION_ODOM_FALLBACK_SOURCE = "simulation_direct_odom_after_tf_retry"

# Keep capacity for /clock, scan, and odometry even when their callback groups
# are simultaneously runnable. Production TF subscriptions live on a separate
# node/executor so they cannot be starved by this executor.
FOLLOWER_EXECUTOR_NUM_THREADS = 4
TF_LISTENER_NODE_NAME = "aufgabe04_waypoint_follower_tf_listener"
CALLBACK_SERVICE_CALLER_SPIN = "caller_spin"
CALLBACK_SERVICE_BACKGROUND_EXECUTOR = "background_executor"
