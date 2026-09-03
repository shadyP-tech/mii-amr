import unittest

from scripts.aufgabe04.navigation.localization.ros_preflight import (
    RosPreflightResult,
)
from scripts.aufgabe04.navigation.localization.ros_preflight_evidence_contract import (
    ROS_PREFLIGHT_EVIDENCE_FIELDS,
    ros_preflight_requirements_evidence,
    validate_ros_preflight_evidence_fields,
    validate_ros_preflight_requirements_evidence,
)


class RosPreflightEvidenceContractTest(unittest.TestCase):
    def _payload(self):
        payload = {name: None for name in ROS_PREFLIGHT_EVIDENCE_FIELDS}
        payload["preflight_requirements"] = (
            ros_preflight_requirements_evidence(
                stationary_map_from_odom_pairing_requested=False,
                stationary_map_from_odom_pairing_required=False,
            )
        )
        return payload

    def test_result_serializer_uses_shared_exact_field_contract(self):
        payload = RosPreflightResult(
            ok=True,
            failures=[],
            observations=[],
            runtime_config={},
        ).to_json_dict()

        self.assertEqual(frozenset(payload), ROS_PREFLIGHT_EVIDENCE_FIELDS)
        validate_ros_preflight_evidence_fields(payload)
        validate_ros_preflight_requirements_evidence(
            payload["preflight_requirements"]
        )

    def test_outer_contract_rejects_missing_and_unknown_fields(self):
        missing = self._payload()
        missing.pop("preflight_requirements")
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            validate_ros_preflight_evidence_fields(missing)

        unknown = self._payload()
        unknown["unexpected"] = None
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            validate_ros_preflight_evidence_fields(unknown)

    def test_requirements_contract_accepts_all_possible_flag_pairs(self):
        for requested, required in (
            (False, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(requested=requested, required=required):
                evidence = ros_preflight_requirements_evidence(
                    stationary_map_from_odom_pairing_requested=requested,
                    stationary_map_from_odom_pairing_required=required,
                )
                validate_ros_preflight_requirements_evidence(evidence)

    def test_requirements_contract_rejects_shape_type_and_invariant_drift(self):
        invalid = (
            (
                {"stationary_map_from_odom_pairing_requested": False},
                "fields mismatch",
            ),
            (
                {
                    "stationary_map_from_odom_pairing_requested": False,
                    "stationary_map_from_odom_pairing_required": False,
                    "unexpected": False,
                },
                "fields mismatch",
            ),
            (
                {
                    "stationary_map_from_odom_pairing_requested": 0,
                    "stationary_map_from_odom_pairing_required": False,
                },
                "flags must be booleans",
            ),
            (
                {
                    "stationary_map_from_odom_pairing_requested": True,
                    "stationary_map_from_odom_pairing_required": False,
                },
                "flags are inconsistent",
            ),
        )
        for evidence, message in invalid:
            with self.subTest(evidence=evidence):
                with self.assertRaisesRegex(ValueError, message):
                    validate_ros_preflight_requirements_evidence(evidence)

    def test_builder_rejects_impossible_requirement_combination(self):
        with self.assertRaisesRegex(ValueError, "cannot be requested"):
            ros_preflight_requirements_evidence(
                stationary_map_from_odom_pairing_requested=True,
                stationary_map_from_odom_pairing_required=False,
            )

    def test_explicit_pairing_policy_rejects_odom_owner_only_evidence(self):
        odom_owner_only = ros_preflight_requirements_evidence(
            stationary_map_from_odom_pairing_requested=False,
            stationary_map_from_odom_pairing_required=True,
        )
        with self.assertRaisesRegex(
            ValueError,
            "did not require stationary map-from-odom pairing",
        ):
            validate_ros_preflight_requirements_evidence(
                odom_owner_only,
                require_explicit_stationary_map_from_odom_pairing=True,
            )

    def test_validator_rejects_non_boolean_requirement_policy(self):
        with self.assertRaisesRegex(TypeError, "must be a bool"):
            validate_ros_preflight_requirements_evidence(
                self._payload()["preflight_requirements"],
                require_explicit_stationary_map_from_odom_pairing=1,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
