"""Fixed eight-scan head acquisition, separate from stored-tour scan evidence."""
from scripts.aufgabe04.perception.lidar_scan_metadata import LidarScanMetadata
from scripts.aufgabe04.real_robot.readiness.tour_scan_contract import _stationary_capture_payload
from scripts.aufgabe04.real_robot.readiness.tour_scan_capture import _TourScanNode, _capture_stationary_scan


HASH_FIELD = "candidate_head_scan_capture_sha256"
SCAN_COUNT = 8
MAX_WINDOW_SEC = 1.5  # Also accommodates eight scans at the LDS-02's 5 Hz floor.


def head_capture_payload(scans, *, tour_id, odom_frame, base_frame, scan_frame,
                         captured_at_unix_sec):
    payload = _stationary_capture_payload(scans, tour_id=tour_id, odom_frame=odom_frame,
        base_frame=base_frame, scan_frame=scan_frame, captured_at_unix_sec=captured_at_unix_sec,
        required_scan_count=SCAN_COUNT, maximum_window_sec=MAX_WINDOW_SEC)
    for scan in scans:
        LidarScanMetadata.from_mapping(scan.get("scan_metadata")).validate(scan["ranges"])
    payload.update(artifact_kind="candidate_head_scan_capture", viewpoint_id=payload.pop("tour_id"),
                   scan_count=SCAN_COUNT)
    return payload


class _HeadScanNode(_TourScanNode):  # pragma: no cover - uses the same passive ROS adapter.
    sample_count = SCAN_COUNT
    payload_builder = staticmethod(head_capture_payload)

    def __init__(self, profile, viewpoint_id, observation_not_before_sec, *, topology_profile):
        self.metadata_profile = topology_profile
        super().__init__(profile, viewpoint_id, observation_not_before_sec)


def capture_head_scan(profile, *, viewpoint_id, output_path, observation_not_before_sec,
                      topology_profile, timeout_sec=3.):
    return _capture_stationary_scan(profile, tour_id=viewpoint_id, output_path=output_path,
        hash_field=HASH_FIELD, timeout_sec=timeout_sec,
        observation_not_before_sec=observation_not_before_sec,
        node_factory=lambda p, uid, floor: _HeadScanNode(p, uid, floor, topology_profile=topology_profile))
