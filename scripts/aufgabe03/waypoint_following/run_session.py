from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RunSessionContext:
    parse_args: Callable[..., Any]
    load_waypoints: Callable[..., Any]
    prepare_executable_waypoints: Callable[..., Any]
    prepare_tracking_setup: Callable[..., Any]
    build_lookahead_guard: Callable[..., Any]
    print_dry_run: Callable[..., Any]
    require_motion_confirmation: Callable[..., Any]
    wait_before_follow_confirmation: Callable[..., Any]
    select_executable_waypoints: Callable[..., Any]
    WaypointFollower: Any
    BlockedByScanError: Any
    WaypointTimeoutError: Any
    notes_with_tracking_metadata: Callable[..., Any]
    notes_with_velocity_scheduler_metadata: Callable[..., Any]
    notes_with_smoothing_metadata: Callable[..., Any]
    notes_with_route_projection_metadata: Callable[..., Any]
    notes_with_guard_metadata: Callable[..., Any]
    notes_with_post_replan_recovery_metadata: Callable[..., Any]
    build_log_row: Callable[..., Any]
    append_csv_row: Callable[..., Any]
    CSV_HEADER: Any
    rclpy: Any
    sys_argv: Any
    stderr: Any
    time_now_sec: Callable[[], float]


def run(argv, context: RunSessionContext):
    parse_args = context.parse_args
    load_waypoints = context.load_waypoints
    prepare_executable_waypoints = context.prepare_executable_waypoints
    prepare_tracking_setup = context.prepare_tracking_setup
    build_lookahead_guard = context.build_lookahead_guard
    print_dry_run = context.print_dry_run
    require_motion_confirmation = context.require_motion_confirmation
    wait_before_follow_confirmation = context.wait_before_follow_confirmation
    select_executable_waypoints = context.select_executable_waypoints
    WaypointFollower = context.WaypointFollower
    BlockedByScanError = context.BlockedByScanError
    WaypointTimeoutError = context.WaypointTimeoutError
    notes_with_tracking_metadata = context.notes_with_tracking_metadata
    notes_with_velocity_scheduler_metadata = (
        context.notes_with_velocity_scheduler_metadata
    )
    notes_with_smoothing_metadata = context.notes_with_smoothing_metadata
    notes_with_route_projection_metadata = (
        context.notes_with_route_projection_metadata
    )
    notes_with_guard_metadata = context.notes_with_guard_metadata
    notes_with_post_replan_recovery_metadata = (
        context.notes_with_post_replan_recovery_metadata
    )
    build_log_row = context.build_log_row
    append_csv_row = context.append_csv_row
    CSV_HEADER = context.CSV_HEADER
    rclpy = context.rclpy
    sys_argv = context.sys_argv
    stderr = context.stderr
    time_now_sec = context.time_now_sec

    args = parse_args(argv if argv is not None else sys_argv[1:])

    try:
        raw_waypoints = load_waypoints(args.waypoints)
        preview_waypoints = prepare_executable_waypoints(
            raw_waypoints,
            skip_first=not args.no_skip_first_waypoint,
            min_spacing_m=args.min_waypoint_spacing_m,
        )
        preview_tracking_points, preview_tracking_validation = prepare_tracking_setup(
            args,
            raw_waypoints,
        )
        preview_lookahead_guard = build_lookahead_guard(args)
    except Exception as exc:
        print(f"follow_planned_waypoints.py: error: {exc}", file=stderr)
        return 2

    if args.dry_run:
        print_dry_run(
            args,
            raw_waypoints,
            preview_waypoints,
            tracking_validation=preview_tracking_validation,
            lookahead_guard=preview_lookahead_guard,
        )
        return 0

    if not require_motion_confirmation(args, preview_waypoints):
        print("Waypoint following cancelled.")
        return 130

    if rclpy is None:
        print(
            "ROS 2 Python modules are unavailable. Source ROS 2 Humble before running.",
            file=stderr,
        )
        return 2

    rclpy.init()
    node = WaypointFollower(args)
    status = "failed"
    notes = args.notes
    reached_count = 0
    executable_waypoints = preview_waypoints
    tracking_points = preview_tracking_points
    tracking_validation = preview_tracking_validation
    start_pose = None
    final_pose = None
    blocked_waypoint = None
    timeout_waypoint = None
    scan_safety = None
    amcl_health = None
    return_code = 1

    try:
        node.wait_for_startup_gate()
        start_pose, _frame, amcl_health = node.check_health_or_recover()
        start_selection = select_executable_waypoints(
            raw_waypoints,
            start_pose,
            args.start_selection,
            args.start_on_path_tolerance_m,
            args.waypoint_tolerance_m,
            args.goal_tolerance_m,
            args.min_waypoint_spacing_m,
            skip_first=not args.no_skip_first_waypoint,
        )
        executable_waypoints = start_selection.waypoints
        tracking_points, tracking_validation = prepare_tracking_setup(
            args,
            raw_waypoints,
            current_pose=start_pose,
            logger=node.get_logger(),
        )
        node.diagnostics.selected_start_segment_index = (
            start_selection.selected_segment_index
        )
        node.diagnostics.selected_start_waypoint_index = (
            start_selection.selected_waypoint_index
        )
        node.diagnostics.distance_to_path_m = start_selection.distance_to_path_m
        if args.verbose:
            node.get_logger().info(
                "Selected executable route: "
                f"segment={start_selection.selected_segment_index}, "
                f"first_waypoint={start_selection.selected_waypoint_index}, "
                f"distance_to_path={start_selection.distance_to_path_m}"
            )
        node.publish_rviz_route(executable_waypoints, current_pose=start_pose)
        node.publish_rviz_obstacles()
        if not wait_before_follow_confirmation(
            args,
            start_pose,
            executable_waypoints,
        ):
            status = "interrupted"
            notes = f"{args.notes};wait_before_follow_cancelled"
            notes = notes_with_velocity_scheduler_metadata(notes, args)
            notes = notes_with_smoothing_metadata(notes, args)
            notes = notes_with_route_projection_metadata(notes, args, node)
            notes = notes_with_post_replan_recovery_metadata(notes, args, node)
            node.diagnostics.final_status_reason = "wait_before_follow_cancelled"
            print("Waypoint following cancelled before custom follower start.")
            return_code = 130
        else:
            if args.wait_before_follow:
                node.refresh_after_operator_wait(time_now_sec())
            result = node.follow_waypoints(
                executable_waypoints,
                tracking_points=tracking_points,
                tracking_validation=tracking_validation,
            )
            reached_count = result["reached_count"]
            start_pose = result["start_pose"]
            final_pose = result["final_pose"]
            scan_safety = result["scan_safety"]
            amcl_health = result["amcl_health"]
            status = result.get("status", "completed")
            notes = notes_with_tracking_metadata(notes, args, tracking_validation)
            notes = notes_with_velocity_scheduler_metadata(notes, args)
            notes = notes_with_smoothing_metadata(notes, args)
            notes = notes_with_route_projection_metadata(notes, args, node)
            notes = notes_with_guard_metadata(
                notes,
                args,
                getattr(node, "last_lookahead_guard_result", None),
            )
            notes = notes_with_post_replan_recovery_metadata(notes, args, node)
            node.diagnostics.final_status_reason = status
            return_code = 0

    except KeyboardInterrupt:
        status = "interrupted"
        notes = f"{args.notes};keyboard_interrupt"
        notes = notes_with_velocity_scheduler_metadata(notes, args)
        notes = notes_with_smoothing_metadata(notes, args)
        notes = notes_with_route_projection_metadata(notes, args, node)
        notes = notes_with_guard_metadata(
            notes,
            args,
            getattr(node, "last_lookahead_guard_result", None),
        )
        notes = notes_with_post_replan_recovery_metadata(notes, args, node)
        node.diagnostics.final_status_reason = "keyboard_interrupt"
        print("Interrupted. Sending stop command...")
        return_code = 130

    except BlockedByScanError as exc:
        status = "blocked"
        notes = f"{args.notes};{exc}"
        notes = notes_with_velocity_scheduler_metadata(notes, args)
        notes = notes_with_smoothing_metadata(notes, args)
        notes = notes_with_route_projection_metadata(notes, args, node)
        notes = notes_with_guard_metadata(
            notes,
            args,
            getattr(node, "last_lookahead_guard_result", None),
        )
        notes = notes_with_post_replan_recovery_metadata(notes, args, node)
        node.diagnostics.final_status_reason = str(exc)
        reached_count = node.reached_count
        start_pose = node.start_pose
        final_pose = node.final_pose
        amcl_health = node.last_amcl_health
        scan_safety = exc.scan_safety
        blocked_waypoint = exc.waypoint
        node.get_logger().error(str(exc))
        return_code = 1

    except WaypointTimeoutError as exc:
        status = "timeout"
        notes = f"{args.notes};{exc}"
        notes = notes_with_velocity_scheduler_metadata(notes, args)
        notes = notes_with_smoothing_metadata(notes, args)
        notes = notes_with_route_projection_metadata(notes, args, node)
        notes = notes_with_guard_metadata(
            notes,
            args,
            getattr(node, "last_lookahead_guard_result", None),
        )
        notes = notes_with_post_replan_recovery_metadata(notes, args, node)
        node.diagnostics.final_status_reason = str(exc)
        reached_count = node.reached_count
        start_pose = node.start_pose
        final_pose = node.final_pose
        scan_safety = node.last_scan_safety
        amcl_health = node.last_amcl_health
        timeout_waypoint = exc.waypoint
        node.get_logger().error(str(exc))
        return_code = 1

    except Exception as exc:
        status = "failed"
        notes = f"{args.notes};{exc}"
        notes = notes_with_velocity_scheduler_metadata(notes, args)
        notes = notes_with_smoothing_metadata(notes, args)
        notes = notes_with_route_projection_metadata(notes, args, node)
        notes = notes_with_guard_metadata(
            notes,
            args,
            getattr(node, "last_lookahead_guard_result", None),
        )
        notes = notes_with_post_replan_recovery_metadata(notes, args, node)
        node.diagnostics.final_status_reason = str(exc)
        reached_count = node.reached_count
        start_pose = node.start_pose
        final_pose = node.final_pose
        scan_safety = node.last_scan_safety
        amcl_health = node.last_amcl_health
        node.get_logger().error(str(exc))
        return_code = 1

    finally:
        try:
            node.stop_repeatedly()
            if final_pose is None:
                try:
                    final_pose, _frame = node.lookup_pose()
                except Exception:
                    final_pose = None
        finally:
            if not args.no_log:
                try:
                    row = build_log_row(
                        args,
                        len(executable_waypoints),
                        reached_count,
                        status,
                        notes,
                        start_pose=start_pose,
                        final_pose=final_pose,
                        blocked_waypoint=blocked_waypoint,
                        timeout_waypoint=timeout_waypoint,
                        base_frame_used=node.base_frame_used,
                        scan_safety=scan_safety,
                        amcl_health=amcl_health,
                        diagnostics=node.diagnostics,
                    )
                    append_csv_row(args.results_csv, CSV_HEADER, row)
                    node.get_logger().info(f"Saved run log to {args.results_csv}")
                except Exception as log_exc:
                    print(
                        f"Could not write waypoint-follow log: {log_exc}",
                        file=stderr,
                    )
            node.destroy_node()
            rclpy.shutdown()

    return return_code
