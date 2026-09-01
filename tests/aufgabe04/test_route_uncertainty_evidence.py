import tempfile
import unittest
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    load_content_hashed_json,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
    evaluate_route_uncertainty_admission,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_evidence import (
    ROUTE_UNCERTAINTY_ARTIFACT_HASH_FIELD,
    RouteUncertaintyAdmissionRejected,
    publish_route_uncertainty_budget,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import (
    CELL_FREE,
    MapMetadata,
    OccupancyGrid,
)
from scripts.aufgabe04.navigation.station_segment.reporting import (
    build_odom_execution_admission_stop_details,
)


def open_costmap() -> Costmap:
    metadata = MapMetadata(
        yaml_path=Path("map.yaml"),
        image_path=Path("map.pgm"),
        resolution=1.0,
        origin=(0.0, 0.0, 0.0),
        negate=0,
        occupied_thresh=0.65,
        free_thresh=0.20,
        mode="trinary",
    )
    return Costmap.from_occupancy_grid(
        OccupancyGrid(
            metadata=metadata,
            width=20,
            height=20,
            cells=tuple(tuple([CELL_FREE] * 20) for _ in range(20)),
        )
    )


def admission(*, robot_radius_m: float):
    return evaluate_route_uncertainty_admission(
        open_costmap(),
        (Pose2D(5.0, 5.0), Pose2D(6.0, 5.0)),
        PlanarCovariance(0.0, 0.0, 0.0),
        RouteUncertaintyAdmissionConfig(
            robot_radius_m=robot_radius_m,
            collision_margin_m=0.0,
            fixed_odom_tracking_bound_m=0.0,
            empirical_odom_drift_bound_m=0.0,
            braking_latency_distance_m=0.0,
            localization_sigma_multiplier=2.0,
            heading_sigma_rad=0.0,
            heading_lever_arm_m=0.0,
            sampling_spacing_m=0.25,
        ),
    )


def payload(result) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": "route_uncertainty_admission",
        "admission": result.to_evidence_dict(),
    }


class RouteUncertaintyEvidenceTest(unittest.TestCase):
    def test_rejection_is_persisted_before_typed_error_is_raised(self):
        result = admission(robot_radius_m=10.0)
        self.assertFalse(result.decision.accepted)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "uncertainty.json"
            with self.assertRaises(RouteUncertaintyAdmissionRejected) as caught:
                publish_route_uncertainty_budget(
                    path,
                    payload=payload(result),
                    admission=result,
                )

            stored = load_content_hashed_json(
                path,
                hash_field=ROUTE_UNCERTAINTY_ARTIFACT_HASH_FIELD,
            )
            error = caught.exception
            details = error.to_stop_details()
            self.assertFalse(
                stored["admission"]["decision"]["decision"]["accepted"]
            )
            self.assertEqual(error.uncertainty_budget_json, path)
            self.assertEqual(
                details["uncertainty_budget_sha256"],
                error.uncertainty_budget_sha256,
            )
            self.assertEqual(details["uncertainty_budget_json"], str(path))
            self.assertFalse(details["uncertainty_budget_accepted"])
            self.assertIn(
                "route uncertainty budget exhausted:",
                str(error),
            )
            runtime_details = build_odom_execution_admission_stop_details(error)
            self.assertEqual(
                runtime_details["uncertainty_budget_sha256"],
                error.uncertainty_budget_sha256,
            )
            self.assertEqual(
                runtime_details["uncertainty_budget_json"],
                str(path),
            )
            self.assertFalse(runtime_details["motion_published"])
            self.assertTrue(runtime_details["fail_closed"])

    def test_accepted_budget_returns_durable_hash(self):
        result = admission(robot_radius_m=0.1)
        self.assertTrue(result.decision.accepted)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "uncertainty.json"
            digest = publish_route_uncertainty_budget(
                path,
                payload=payload(result),
                admission=result,
            )
            stored = load_content_hashed_json(
                path,
                hash_field=ROUTE_UNCERTAINTY_ARTIFACT_HASH_FIELD,
            )

        self.assertEqual(len(digest), 64)
        self.assertTrue(
            stored["admission"]["decision"]["decision"]["accepted"]
        )

    def test_payload_mismatch_fails_before_publication(self):
        result = admission(robot_radius_m=0.1)
        mismatched = payload(result)
        mismatched["admission"] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "uncertainty.json"
            with self.assertRaisesRegex(ValueError, "payload differs"):
                publish_route_uncertainty_budget(
                    path,
                    payload=mismatched,
                    admission=result,
                )
            self.assertFalse(path.exists())

    def test_immutable_conflict_is_not_misreported_as_admission_rejection(self):
        accepted = admission(robot_radius_m=0.1)
        rejected = admission(robot_radius_m=10.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "uncertainty.json"
            publish_route_uncertainty_budget(
                path,
                payload=payload(accepted),
                admission=accepted,
            )
            with self.assertRaises(ContentStoreError):
                publish_route_uncertainty_budget(
                    path,
                    payload=payload(rejected),
                    admission=rejected,
                )

    def test_generic_admission_failure_does_not_fabricate_evidence_identity(self):
        details = build_odom_execution_admission_stop_details(
            ValueError("map certificate unavailable")
        )

        self.assertNotIn("uncertainty_budget_json", details)
        self.assertNotIn("uncertainty_budget_sha256", details)
        self.assertEqual(details["fault_code"], "odom_execution_admission_failed")
        self.assertFalse(details["motion_published"])


if __name__ == "__main__":
    unittest.main()
