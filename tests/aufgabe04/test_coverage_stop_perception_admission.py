from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
import unittest

from scripts.aufgabe04.navigation.content_hashed_evidence import payload_sha256
from scripts.aufgabe04.navigation.coverage_stop_perception_admission import (
    ContentHashedAdmissionArtifact,
    CoverageEpochPerceptionAdmission,
    CoverageVisibilityReconciliationAdmission,
)


ROOT = Path(__file__).resolve().parents[2]
ADMISSION_MODULE = (
    ROOT
    / "scripts/aufgabe04/navigation/coverage_stop_perception_admission.py"
)
RECORD_MODULE = ROOT / "scripts/aufgabe04/navigation/record_stand_coverage_stop.py"


class CoverageStopPerceptionAdmissionBoundaryTest(unittest.TestCase):
    def test_admission_state_is_frozen_and_content_bound(self):
        self.assertTrue(
            CoverageEpochPerceptionAdmission.__dataclass_params__.frozen
        )
        self.assertTrue(
            CoverageVisibilityReconciliationAdmission.__dataclass_params__.frozen
        )
        payload = {"schema_version": 1, "motion_authorized": False}
        digest = payload_sha256(payload)
        artifact = ContentHashedAdmissionArtifact(
            kind="test_admission",
            path=Path(f"test_admission_{digest}.json"),
            payload=payload,
            sha256=digest,
            hash_field="test_admission_sha256",
        )
        self.assertIs(artifact.validated(), artifact)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            artifact.sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            dataclasses.replace(artifact, payload={"schema_version": 2}).validated()

    def test_module_is_ros_free_and_parent_uses_only_public_facade(self):
        admission_tree = ast.parse(ADMISSION_MODULE.read_text())
        admission_imports = _imported_modules(admission_tree)
        self.assertFalse(
            any(
                name == "rclpy" or name.startswith("rclpy.")
                for name in admission_imports
            )
        )

        parent_tree = ast.parse(RECORD_MODULE.read_text())
        facade_imports = [
            node
            for node in ast.walk(parent_tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            == "scripts.aufgabe04.navigation.coverage_stop_perception_admission"
        ]
        self.assertEqual(len(facade_imports), 1)
        self.assertTrue(
            all(not alias.name.startswith("_") for alias in facade_imports[0].names)
        )
        prohibited_direct_imports = {
            "scripts.aufgabe04.navigation.coverage_candidate_reconciliation",
            "scripts.aufgabe04.navigation.coverage_candidate_reconciliation_report",
            "scripts.aufgabe04.navigation.coverage_visibility_reporting",
            "scripts.aufgabe04.navigation.stand_candidate_static_map_admission",
            "scripts.aufgabe04.perception.lidar_stand_morphology",
            "scripts.aufgabe04.perception.lidar_visibility_evidence",
        }
        self.assertTrue(
            prohibited_direct_imports.isdisjoint(_imported_modules(parent_tree))
        )


def _imported_modules(tree: ast.AST) -> set[str]:
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    return names


if __name__ == "__main__":
    unittest.main()
