"""Fixed-model failure classification on original current-frame pixels.

Recorded QR text/corners are explicit geometry inputs here; decoder recovery
has separate tests. The recorded automatic head proposal is not a manually
selected border, and the raw-border refinement still runs on the source JPEG.
"""

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover
    cv2 = None
    numpy = None

from scripts.aufgabe04.perception.camera_calibration import rectify_bgr_frame
from scripts.aufgabe04.perception.stand_axis.geometry_contract import (
    HEAD_QR_GEOMETRY_MISMATCH,
    classify_joint_geometry_contract,
)
from scripts.aufgabe04.perception.stand_axis.model_diagnostics import metric_fit_diagnostics_payload
from scripts.aufgabe04.perception.stand_axis.model_pipeline import estimate_stand_axis_from_metric_model
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.pose_fit_diagnostics import collect_metric_model_diagnostics
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import (
    RectifiedCameraMatrix,
    estimate_planar_pose_ippe,
)
from scripts.aufgabe04.perception.stand_axis_consensus import axis_conditioning
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.configuration.geometry import intrinsics_from_camera_info
from scripts.aufgabe04.real_robot.observer.axis_sample_policy import admit_axis_sample

REPO = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures/qr_recovery_20260911"
PIPELINE = "scripts.aufgabe04.perception.stand_axis.model_pipeline."
DIAGNOSTICS = "scripts.aufgabe04.perception.stand_axis.pose_fit_diagnostics."


@unittest.skipIf(cv2 is None or numpy is None, "OpenCV and numpy are required")
class GeometryContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile_path = REPO / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"
        cls.profile_bytes = cls.profile_path.read_bytes()
        cls.profile = load_measured_physical_stand_model(cls.profile_path)
        cls.frame = next(
            frame for frame in json.loads((FIXTURE / "inputs.json").read_text())["frames"]
            if frame["frame_index"] == 24
        )
        image_bytes = (FIXTURE / cls.frame["image_file"]).read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != cls.frame["image_sha256"]:
            raise AssertionError("source JPEG changed")
        if cls.profile.sha256 != cls.frame["model_profile_sha256"]:
            raise AssertionError("physical profile changed")
        info = SimpleNamespace(**cls.frame["camera_info"])
        intrinsics = intrinsics_from_camera_info(info)
        raw = cv2.imdecode(numpy.frombuffer(image_bytes, numpy.uint8), cv2.IMREAD_COLOR)
        rectified = rectify_bgr_frame(raw, info, cv2, numpy)
        attempt = cls.frame["geometry_strict_attempt"]
        roi = attempt["roi"]
        cls.crop = rectified[roi["y0"]:roi["y1"], roi["x0"]:roi["x1"]]
        cls.camera = RectifiedCameraMatrix(
            intrinsics.fx_px, intrinsics.fy_px,
            intrinsics.cx_px - roi["x0"], intrinsics.cy_px - roi["y0"],
        )
        cls.qr_corners = tuple(ImagePoint(**p) for p in cls.frame["geometry_qr_corners"])
        detector = cls.frame["stand_axis_profile"]
        cls.options = dict(
            model_profile=cls.profile,
            camera_fx_px=cls.camera.fx_px, camera_fy_px=cls.camera.fy_px,
            camera_cx_px=cls.camera.cx_px, camera_cy_px=cls.camera.cy_px,
            qr_observations=(DecodedQrObservation(
                "QR_003", tuple((p.u_px, p.v_px) for p in cls.qr_corners),
                "recorded_same_frame_geometry_diagnostic",
            ),),
            edge_preprocess=detector["edge_preprocess"],
            canny_low=detector["canny_low"], canny_high=detector["canny_high"],
            min_edge_height_px=detector["min_edge_height_px"],
            expected_head_center_u_px=attempt["expected_center_u_px"] - roi["x0"],
            expected_head_center_v_px=attempt["expected_center_v_px"] - roi["y0"],
            expected_head_height_px=attempt["expected_head_height_px"],
            current_head_proposal_corners=tuple(
                ImagePoint(**p) for p in cls.frame["geometry_head_proposal_corners"]
            ),
        )
        cls.estimate, cls.debug = estimate_stand_axis_from_metric_model(cv2, cls.crop, **cls.options)
        # Preserve the audit's original inner-looking border contract as an
        # explicit diagnostic. Production now checks its neck junction and may
        # recover a different complete outer border from the same raw pixels.
        cls.original_corners = tuple(ImagePoint(**p) for p in cls.frame["geometry_refined_corners"])
        head = estimate_planar_pose_ippe(cv2, cls.original_corners, cls.profile.head_corners, cls.camera)
        qr = estimate_planar_pose_ippe(cv2, cls.qr_corners, cls.profile.qr_corners, cls.camera)
        cls.original_joint_pose = estimate_planar_pose_ippe(
            cv2, cls.original_corners + cls.qr_corners,
            cls.profile.head_corners + cls.profile.qr_corners, cls.camera,
        )
        cls.original_diagnostics = collect_metric_model_diagnostics(
            cv2, profile=cls.profile, camera=cls.camera,
            head_corners=cls.original_corners, qr_corners=cls.qr_corners,
            diagnostic_pose=cls.original_joint_pose.hypotheses[0], qr_pose=qr, head_pose=head,
        )

    def classify(self, **overrides):
        inputs = dict(
            profile=self.profile, diagnostics=self.original_diagnostics,
            joint_reason="reprojection_error_too_high",
            joint_reprojection_rmse_px=self.frame["geometry_joint_rmse_px"],
            max_reprojection_rmse_px=2.0, qr_marker_verified=True,
        )
        inputs.update(overrides)
        return classify_joint_geometry_contract(**inputs)

    def test_original_inner_border_stays_diagnostic_and_raw_outer_recovery_is_explicit(self):
        estimate, debug = self.estimate, self.debug
        self.assertEqual(estimate.reason, "axis_estimated_current_measured_head")
        self.assertTrue(estimate.usable)
        self.assertEqual(estimate.evidence_state, "fresh_refined")
        self.assertAlmostEqual(estimate.yaw_deg, 30.909475167, places=7)
        self.assertIsNotNone(estimate.camera_face_normal_xyz)
        self.assertIsNotNone(debug.model_pose)
        self.assertTrue(debug.qr_marker_verified)
        self.assertTrue(debug.corner_arm_support.accepted)
        self.assertEqual(debug.model_pose_fit_source, "model_current_measured_head")
        self.assertEqual(debug.head_outer_recovery.original_corners, self.original_corners)
        self.assertTrue(debug.head_outer_recovery.accepted)
        self.assertEqual(debug.head_outer_recovery.original_neck_start_gap_px, 4)
        self.assertEqual(debug.head_neck_junction.start_gap_px, 0)
        self.assertEqual(len(debug.head_outer_recovery.attempted_growth_factors), 2)
        self.assertNotEqual(debug.refined_corners, self.original_corners)
        self.assertAlmostEqual(
            self.original_joint_pose.hypotheses[0].reprojection_rmse_px,
            self.frame["geometry_joint_rmse_px"], places=8,
        )
        self.assertAlmostEqual(estimate.pose_reprojection_rmse_px, 0.346305284154, places=8)

    def test_serialized_evidence_separates_independent_fits_and_dimension_sources(self):
        payload = metric_fit_diagnostics_payload(self.debug)
        contract = asdict(self.classify())
        self.assertEqual(contract["underlying_joint_reason"], "reprojection_error_too_high")
        self.assertEqual(contract["maximum_reprojection_rmse_px"], 2.0)
        self.assertAlmostEqual(contract["independent_head_rmse_px"], 0.456818578, places=7)
        self.assertAlmostEqual(contract["independent_qr_rmse_px"], 0.335953513, places=7)
        self.assertAlmostEqual(contract["diagnostic_head_yaw_deg"], 37.814968012, places=7)
        self.assertAlmostEqual(contract["diagnostic_qr_yaw_deg"], 34.104196074, places=7)
        self.assertAlmostEqual(contract["diagnostic_axis_separation_deg"], 3.710771938, places=7)
        self.assertEqual(contract["head_size_m"], (0.078, 0.078))
        self.assertEqual(contract["qr_symbol_size_m"], (0.062, 0.062))
        self.assertEqual(contract["qr_panel_size_m"], (0.071, 0.071))
        self.assertEqual(contract["expected_head_qr_ratio"], (78 / 62, 78 / 62))
        self.assertEqual(contract["expected_head_panel_ratio"], (78 / 71, 78 / 71))
        self.assertAlmostEqual(contract["observed_head_qr_ratio"][0], 1.079444160, places=7)
        self.assertAlmostEqual(contract["observed_head_qr_ratio"][1], 1.129973123, places=7)
        self.assertEqual(contract["profile_dimension_source"], self.profile.source)
        self.assertIn("estimated", contract["profile_dimension_source"].lower())
        self.assertFalse(contract["motion_authorized"])
        self.assertFalse(contract["measurement_authorized"])
        self.assertFalse(contract["profile_changed"])
        self.assertAlmostEqual(payload["model_pose"]["reprojection_rmse_px"], 0.346305284154, places=8)
        self.assertTrue(payload["head_neck_junction"]["accepted"])
        self.assertTrue(payload["head_outer_recovery"]["accepted"])
        self.assertEqual(self.profile_path.read_bytes(), self.profile_bytes)

    def test_classification_does_not_add_pose_solves(self):
        with (
            patch("scripts.aufgabe04.perception.stand_axis.head_model_fit.estimate_planar_pose_ippe",
                  wraps=estimate_planar_pose_ippe) as current_fit,
            patch(DIAGNOSTICS + "estimate_planar_pose_ippe", wraps=estimate_planar_pose_ippe) as diagnostic_fit,
        ):
            estimate, debug = estimate_stand_axis_from_metric_model(cv2, self.crop, **self.options)
        self.assertEqual(estimate.reason, "axis_estimated_current_measured_head")
        self.assertEqual(current_fit.call_count, 3)  # Head, QR and diagnostic joint.
        self.assertEqual(diagnostic_fit.call_count, 0)  # Reuse independent head.
        self.assertTrue(debug.head_model_quality.neck_junction_verified)

    def test_incomplete_or_bad_independent_evidence_keeps_original_reason(self):
        diagnostics = self.original_diagnostics
        cases = [
            {"qr_marker_verified": False},
            {"joint_reason": "pose_estimated"},
            {"joint_reason": "planar_pose_axis_ambiguous"},
            {"joint_reprojection_rmse_px": 2.0},
            {"joint_reprojection_rmse_px": None},
            {"joint_reprojection_rmse_px": math.nan},
            {"max_reprojection_rmse_px": 0.0},
            {"max_reprojection_rmse_px": math.inf},
            {"profile": replace(self.profile, measurement_status="provisional")},
        ]
        for name in ("head_only", "qr_only"):
            original = getattr(diagnostics, name)
            cases.append({"diagnostics": replace(diagnostics, **{name: None})})
            for changes in (
                {"accepted": False}, {"axis_ambiguous": True},
                {"reprojection_rmse_px": 2.001}, {"reprojection_rmse_px": -0.1},
                {"reprojection_rmse_px": None}, {"reprojection_rmse_px": math.nan},
                {"yaw_deg": None}, {"yaw_deg": math.inf},
            ):
                cases.append({"diagnostics": replace(diagnostics, **{name: replace(original, **changes)})})
        for name in ("observed_head_qr_width_ratio", "observed_head_qr_height_ratio"):
            for value in (None, math.nan, 0.0, -1.0):
                cases.append({"diagnostics": replace(diagnostics, **{name: value})})
        for overrides in cases:
            with self.subTest(overrides=overrides):
                self.assertIsNone(self.classify(**overrides))

    def test_profile_tolerance_never_relaxes_pixel_gate(self):
        contract = self.classify(profile=replace(self.profile, tolerance_m=0.020))
        self.assertEqual(contract.maximum_reprojection_rmse_px, 2.0)
        self.assertEqual(contract.qr_symbol_size_m, (0.062, 0.062))
        self.assertFalse(contract.measurement_authorized)
        self.assertFalse(contract.profile_changed)

    def test_counterfactual_paper_sized_symbol_still_fails_obliqueness_gate(self):
        # Sensitivity diagnostic only: synthetic points do not edit the physical
        # profile or imply that image fitting measures a real 71 mm symbol.
        model = self.profile
        synthetic_qr = tuple(type(p)(
            model.qr_center_x_m + (p.x_m - model.qr_center_x_m) * 0.071 / model.qr_symbol_width_m,
            model.qr_center_y_m + (p.y_m - model.qr_center_y_m) * 0.071 / model.qr_symbol_height_m,
            p.z_m,
        ) for p in model.qr_corners)
        fit = estimate_planar_pose_ippe(
            cv2, self.original_corners + self.qr_corners,
            model.head_corners + synthetic_qr, self.camera,
        )
        self.assertTrue(fit.accepted)
        self.assertAlmostEqual(fit.best.reprojection_rmse_px, 1.460511798, places=7)
        self.assertAlmostEqual(fit.best.yaw_deg, 36.133127462, places=7)
        yaw = math.radians(fit.best.yaw_deg)
        counterfactual = replace(
            self.estimate, usable=True, evidence_state="fresh_refined",
            reason="axis_estimated_model_current_frame_refined",
            source="model_current_frame_refined", yaw_deg=fit.best.yaw_deg,
            camera_face_normal_xyz=fit.best.face_normal_xyz,
            pose_reprojection_rmse_px=fit.best.reprojection_rmse_px,
            pose_ambiguity_gap_px=fit.ambiguity_gap_px,
        )
        debug = replace(
            self.debug, evidence_state="fresh_refined", model_pose=fit.best,
            model_pose_fit_source="joint_qr_head", head_model_quality=None,
            head_backside_classification=None,
            pose_reprojection_rmse_px=fit.best.reprojection_rmse_px,
            pose_ambiguity_gap_px=fit.ambiguity_gap_px,
        )
        admitted = admit_axis_sample(
            estimate=counterfactual, debug=debug,
            conditioning=axis_conditioning(yaw, max_obliqueness_rad=math.radians(30.0)),
            yaw_rad=yaw, qr_texts=("QR_003",), lidar_target_associated=True,
        )
        self.assertFalse(admitted.accepted)
        self.assertEqual(admitted.reason, "oblique_silhouette")
        self.assertIsNone(admitted.source)
        self.assertEqual(self.profile_path.read_bytes(), self.profile_bytes)


if __name__ == "__main__":
    unittest.main()
