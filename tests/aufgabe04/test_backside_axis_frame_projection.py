import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import (
    load_backside_axis_frame_projection,
    write_backside_axis_frame_projection,
)
from scripts.aufgabe04.artifacts.backside_axis_observation import (
    load_backside_axis_observation,
)
from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    map_pose_to_odom,
    odom_pose_to_map,
)
from tests.aufgabe04.backside_axis_fixture import (
    backside_axis_payload,
    write_candidate_frame_projection_fixture,
)
from tests.aufgabe04.test_candidate_uncertainty_handoff import (
    planning_frame as recorded_planning_frame,
    recorded_fixture,
)
from tests.aufgabe04.test_exact_two_camera_decision import (
    _camera_frame_projection,
    _fixture,
)


class BacksideAxisFrameProjectionTest(unittest.TestCase):
    def test_real_preflight_frame_projects_and_reloads_backside_receipt(self):
        recorded = recorded_fixture()
        frame = recorded_planning_frame(recorded["preflight"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root, with_frame_provenance=True)
            uid = "survey_candidate_0003"
            source = _camera_frame_projection(fixture, root / "source")
            target = _camera_frame_projection(
                fixture, root / "target", planning_frame=frame
            )
            observed = source.projection.projected_snapshot.candidate_for(uid)
            expected = target.projection.projected_snapshot.candidate_for(uid)
            robot = Pose2D(observed.geometry.x_m, observed.geometry.y_m + 0.7)
            axis_path = root / "axis.json"
            axis_path.write_text(json.dumps(backside_axis_payload(
                stand_id=uid,
                stand_x_m=observed.geometry.x_m,
                stand_y_m=observed.geometry.y_m,
                robot_x_m=robot.x_m,
                robot_y_m=robot.y_m,
            )))
            output = root / "axis_projection.json"

            write_backside_axis_frame_projection(
                output,
                axis_evidence_path=axis_path,
                source_candidate_projection_path=source.evidence_path,
                source_candidate_projection_sha256=source.evidence_sha256,
                target_candidate_projection_path=target.evidence_path,
                target_candidate_projection_sha256=target.evidence_sha256,
                target_candidate_x_m=expected.geometry.x_m,
                target_candidate_y_m=expected.geometry.y_m,
            )
            projected = load_backside_axis_frame_projection(output)

            expected_robot = odom_pose_to_map(
                map_pose_to_odom(robot, source.projection.planning_frame.map_from_odom),
                frame.map_from_odom,
            )
            self.assertAlmostEqual(projected.stand_x_m, expected.geometry.x_m)
            self.assertAlmostEqual(projected.stand_y_m, expected.geometry.y_m)
            self.assertAlmostEqual(projected.robot_x_m, expected_robot.x_m)
            self.assertAlmostEqual(projected.robot_y_m, expected_robot.y_m)
            self.assertAlmostEqual(projected.stand_axis_rad, frame.map_from_odom.yaw_rad)
            self.assertEqual(
                projected.target_candidate_projection_sha256, target.evidence_sha256
            )
            self.assertEqual(
                json.loads(target.evidence_path.read_text())["planning_frame_admission"],
                recorded["expected_planning_frame"],
            )

    def _artifacts(self, root: Path):
        axis_path = root / "axis.json"
        axis_path.write_text(
            json.dumps(
                backside_axis_payload(
                    stand_x_m=1.0,
                    robot_x_m=1.0,
                    robot_y_m=0.7,
                )
            ),
            encoding="utf-8",
        )
        source_path = root / "source_projection.json"
        source_sha256, _, _ = write_candidate_frame_projection_fixture(
            source_path,
            candidate_uid="candidate_1",
            canonical_x_m=1.0,
            canonical_y_m=0.0,
            transform_x_m=0.0,
            transform_y_m=0.0,
            transform_yaw_rad=0.0,
        )
        target_path = root / "target_projection.json"
        target_sha256, _, _ = write_candidate_frame_projection_fixture(
            target_path,
            candidate_uid="candidate_1",
            canonical_x_m=1.0,
            canonical_y_m=0.0,
            transform_x_m=0.2,
            transform_y_m=-0.1,
            transform_yaw_rad=math.pi / 2.0,
        )
        return axis_path, source_path, source_sha256, target_path, target_sha256

    def test_yaw_and_translation_projection_is_distinct_and_authenticated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis, source, source_sha, target, target_sha = self._artifacts(root)
            output = root / "axis_projection.json"

            write_backside_axis_frame_projection(
                output,
                axis_evidence_path=axis,
                source_candidate_projection_path=source,
                source_candidate_projection_sha256=source_sha,
                target_candidate_projection_path=target,
                target_candidate_projection_sha256=target_sha,
                target_candidate_x_m=0.2,
                target_candidate_y_m=0.9,
            )
            projected = load_backside_axis_frame_projection(output)

            self.assertAlmostEqual(projected.stand_x_m, 0.2)
            self.assertAlmostEqual(projected.stand_y_m, 0.9)
            self.assertAlmostEqual(projected.robot_x_m, -0.5)
            self.assertAlmostEqual(projected.robot_y_m, 0.9)
            self.assertAlmostEqual(projected.stand_axis_rad, math.pi / 2.0)
            self.assertAlmostEqual(projected.opposite_face_normal_rad, 0.0)
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_backside_axis_observation(output)

    def test_source_receipt_edit_invalidates_derived_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis, source, source_sha, target, target_sha = self._artifacts(root)
            output = root / "axis_projection.json"
            write_backside_axis_frame_projection(
                output,
                axis_evidence_path=axis,
                source_candidate_projection_path=source,
                source_candidate_projection_sha256=source_sha,
                target_candidate_projection_path=target,
                target_candidate_projection_sha256=target_sha,
                target_candidate_x_m=0.2,
                target_candidate_y_m=0.9,
            )
            payload = json.loads(axis.read_text(encoding="utf-8"))
            payload["axis_confidence"] = 0.91
            axis.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_backside_axis_frame_projection(output)

    def test_projection_lineage_mismatch_fails_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis, source, source_sha, _, _ = self._artifacts(root)
            target = root / "other_target_projection.json"
            target_sha, _, _ = write_candidate_frame_projection_fixture(
                target,
                candidate_uid="candidate_1",
                canonical_x_m=1.0,
                canonical_y_m=0.0,
                transform_x_m=0.2,
                transform_y_m=-0.1,
                transform_yaw_rad=math.pi / 2.0,
                source_registry_sha256="d" * 64,
                source_snapshot_path=root / "other_source_snapshot.json",
            )
            output = root / "must_not_exist.json"

            with self.assertRaisesRegex(ValueError, "different lineage"):
                write_backside_axis_frame_projection(
                    output,
                    axis_evidence_path=axis,
                    source_candidate_projection_path=source,
                    source_candidate_projection_sha256=source_sha,
                    target_candidate_projection_path=target,
                    target_candidate_projection_sha256=target_sha,
                    target_candidate_x_m=0.2,
                    target_candidate_y_m=0.9,
                )
            self.assertFalse(output.exists())

    def test_rehashed_invalid_reprojection_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis, source, _, target, target_sha = self._artifacts(root)
            payload = load_content_hashed_json(
                source, hash_field="candidate_frame_projection_sha256"
            )
            payload["candidate_reprojections"]["candidate_1"][
                "diagnostics"
            ] = {}
            tampered = root / "rehashed_invalid_projection.json"
            tampered_sha = write_content_hashed_json(
                tampered,
                payload,
                hash_field="candidate_frame_projection_sha256",
            )

            with self.assertRaisesRegex(ValueError, "fields mismatch"):
                write_backside_axis_frame_projection(
                    root / "must_not_exist.json",
                    axis_evidence_path=axis,
                    source_candidate_projection_path=tampered,
                    source_candidate_projection_sha256=tampered_sha,
                    target_candidate_projection_path=target,
                    target_candidate_projection_sha256=target_sha,
                    target_candidate_x_m=0.2,
                    target_candidate_y_m=0.9,
                )

    def test_rehashed_boolean_projection_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis, source, _, target, target_sha = self._artifacts(root)
            payload = load_content_hashed_json(
                source, hash_field="candidate_frame_projection_sha256"
            )
            payload["schema_version"] = True
            tampered = root / "rehashed_boolean_schema.json"
            tampered_sha = write_content_hashed_json(
                tampered,
                payload,
                hash_field="candidate_frame_projection_sha256",
            )

            with self.assertRaisesRegex(ValueError, "unsupported"):
                write_backside_axis_frame_projection(
                    root / "must_not_exist.json",
                    axis_evidence_path=axis,
                    source_candidate_projection_path=tampered,
                    source_candidate_projection_sha256=tampered_sha,
                    target_candidate_projection_path=target,
                    target_candidate_projection_sha256=target_sha,
                    target_candidate_x_m=0.2,
                    target_candidate_y_m=0.9,
                )


if __name__ == "__main__":
    unittest.main()
