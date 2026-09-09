"""Admission boundaries for the second-candidate replacement startup failure."""

from copy import deepcopy
import unittest

from scripts.aufgabe04.navigation.localization.prestart_localization_reseal import (
    evaluate_prestart_localization_reseal,
)
from tests.aufgabe04.test_prestart_localization_reseal import _stop_details


def initial_map_tf_stop():
    """Typed equivalent of the saved valid-odom / missing-global-edge stop."""
    failed = {
        "source": "tf_lookup", "reason": "lookup_exception",
        "target_frame": "map", "source_frame": "odom",
        "exception_type": "LookupException", "available": False,
        "validation_passed": False, "age_sec": None, "stamp_sec": None,
        "max_age_sec": 1.0, "max_future_sec": 1.1,
    }
    ready = {
        "source": "tf_lookup", "reason": "fresh_transform",
        "target_frame": "odom", "source_frame": "base_footprint",
        "available": True, "validation_passed": True, "stamp_sec": 100.0,
        "age_sec": 0.02, "max_age_sec": 1.0, "max_future_sec": 0.02,
    }
    context = {
        "certificate_sha256": "a" * 64,
        "map_frame": "map", "odom_frame": "odom", "base_frame": "base_footprint",
        "frozen_map_from_odom": {"x_m": -1.6105, "y_m": -0.0843, "yaw_rad": -0.4870},
        "max_map_from_odom_translation_drift_m": 0.14034,
        "max_map_from_odom_yaw_drift_rad": 0.09207,
    }
    state = {
        "schema_version": 2, "phase": "cold_tf_acquisition", "extension_used": True,
        "failed_edge_role": "global_consistency", "sensor_inputs_fresh": True,
        "admission_failure_seen": False, "deadline_exhausted": True,
        "denial_reason": "cold_tf_acquisition_deadline_exhausted",
        "initial_sensor_wait_sec": 2.0, "initial_tf_acquisition_wait_sec": 3.0,
        "maximum_startup_wait_sec": 5.0, "elapsed_sec": 5.01,
        "required_edges": ["execution_pose", "global_consistency"],
        "execution_context": context, "motion_authorized": False,
        "fresh_sensor_and_localization_admission_required": True,
        "executor_health": {
            "ready": True, "thread_alive": True, "heartbeat_count": 41,
            "heartbeat_age_sec": 0.000658, "heartbeat_max_age_sec": 0.5,
        },
        "edges": {},
    }
    for role, sample, acquired in (("execution_pose", ready, True), ("global_consistency", failed, False)):
        state["edges"][role] = {
            "target_frame": sample["target_frame"], "source_frame": sample["source_frame"],
            "attempt_count": 4, "successful_sample_count": 4 if acquired else 0,
            "non_acquisition_failure_seen": False, "current_ready": acquired,
            "last_sample": deepcopy(sample),
        }
    return {
        **failed, "execution_phase": "before_motion", "phase": "initial_runtime_input_wait",
        "motion_published": False, "fail_closed": True, "initial_tf_acquisition": state,
    }


def decide(details, **kwargs):
    return evaluate_prestart_localization_reseal(
        status=kwargs.get("status", "stopped"), motion_published=kwargs.get("motion", False),
        stop_details=details,
    )


def initial_tf_drift_stop():
    details = _stop_details()
    state = initial_map_tf_stop()["initial_tf_acquisition"]
    state.update(admission_failure_seen=True, denial_reason="continuity_admission_failed",
                 phase="initial_sensor_wait", extension_used=False, elapsed_sec=2.02,
                 deadline_exhausted=False)
    continuity = details["continuity"]
    state["execution_context"].update(
        frozen_map_from_odom=deepcopy(continuity["frozen_map_from_odom"]),
        max_map_from_odom_translation_drift_m=continuity["max_translation_drift_m"],
        max_map_from_odom_yaw_drift_rad=continuity["max_yaw_drift_rad"],
    )
    global_edge = state["edges"]["global_consistency"]
    global_edge.update(successful_sample_count=4, current_ready=True)
    global_edge["last_sample"] = {
        **state["edges"]["execution_pose"]["last_sample"],
        "target_frame": "map", "source_frame": "odom", "max_future_sec": 1.1,
    }
    details["initial_tf_acquisition"] = state
    return details


class InitialMapTfRecoveryTests(unittest.TestCase):
    def test_missing_global_edge_requests_new_preparation_without_motion_authority(self):
        for exception in ("LookupException", "ConnectivityException"):
            with self.subTest(exception=exception):
                details = initial_map_tf_stop()
                details["exception_type"] = exception
                details["initial_tf_acquisition"]["edges"]["global_consistency"]["last_sample"]["exception_type"] = exception
                decision = decide(details)
                self.assertTrue(decision.eligible, decision.reason)
                self.assertEqual(decision.recovery_action, "tf_warmup_retry")
                self.assertTrue(decision.requires_fresh_localization)
                self.assertTrue(decision.requires_new_route_certificate)
                self.assertFalse(decision.automatic_motion_authorized)

    def test_unsafe_or_incomplete_evidence_cannot_enter_startup_recovery(self):
        mutations = [
            (("source",), "global_consistency_monitor"),
            (("exception_type",), "ExtrapolationException"),
            (("reason",), "stale_transform"),
            (("source_frame",), "base_footprint"),
            (("execution_phase",), "after_motion"),
            (("motion_published",), True),
            (("fail_closed",), 1),
            (("age_sec",), 99), (("stamp_sec",), 1),
            (("max_future_sec",), 2), (("continuity",), _stop_details()["continuity"]),
        ]
        for path, value in (
            (("schema_version",), 3), (("schema_version",), True),
            (("deadline_exhausted",), False), (("elapsed_sec",), 4.99),
            (("initial_tf_acquisition_wait_sec",), 3.1), (("initial_tf_acquisition_wait_sec",), 0),
            (("maximum_startup_wait_sec",), 8), (("sensor_inputs_fresh",), False),
            (("admission_failure_seen",), True), (("motion_authorized",), True),
            (("failed_edge_role",), "execution_pose"), (("required_edges",), [[], {}]),
            (("executor_health", "ready"), False),
            (("executor_health", "heartbeat_age_sec"), 0.6),
            (("executor_health", "heartbeat_count"), True),
            (("execution_context", "certificate_sha256"), ""),
            (("execution_context", "odom_frame"), "different_odom"),
            (("edges", "global_consistency", "successful_sample_count"), 1),
            (("edges", "global_consistency", "successful_sample_count"), False),
            (("edges", "global_consistency", "non_acquisition_failure_seen"), True),
            (("edges", "global_consistency", "last_sample", "exception_type"), "ExtrapolationException"),
            (("edges", "global_consistency", "last_sample", "stamp_sec"), 100.0),
            (("edges", "execution_pose", "current_ready"), False),
            (("edges", "execution_pose", "last_sample", "age_sec"), 1.1),
            (("edges", "execution_pose", "last_sample", "age_sec"), -0.03),
            (("edges", "execution_pose", "last_sample", "age_sec"), float("nan")),
            (("edges", "execution_pose", "last_sample", "reason"), "stale_transform"),
            (("edges", "execution_pose", "last_sample", "source"), "not_tf"),
        ):
            mutations.append((("initial_tf_acquisition", *path), value))
        for path, value in mutations:
            with self.subTest(path=path, value=value):
                details = initial_map_tf_stop()
                target = details
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertFalse(decide(details).eligible)
        for field in initial_map_tf_stop()["initial_tf_acquisition"]:
            with self.subTest(missing_field=field):
                details = initial_map_tf_stop()
                del details["initial_tf_acquisition"][field]
                self.assertFalse(decide(details).eligible)
        for field in ("max_age_sec", "max_future_sec"):
            details = initial_map_tf_stop()
            del details[field]
            del details["initial_tf_acquisition"]["edges"]["global_consistency"]["last_sample"][field]
            self.assertFalse(decide(details).eligible)

    def test_typed_failure_cannot_fall_back_to_legacy_warning_admission(self):
        for warning in ("stale_map_from_odom", "future_map_from_odom", "map_from_odom_lookup_failed: buffer warming"):
            details = _stop_details(reason="map_from_odom_missing", warning=warning)
            self.assertTrue(decide(details).eligible)  # Existing historical contract.
            details["initial_tf_acquisition"] = initial_map_tf_stop()["initial_tf_acquisition"]
            self.assertFalse(decide(details).eligible)
        for schema in (None, {}, {"schema_version": 3}, {"schema_version": True}):
            details = _stop_details(reason="map_from_odom_missing", warning="stale_map_from_odom")
            details["initial_tf_acquisition"] = schema
            self.assertFalse(decide(details).eligible)

    def test_measured_drift_keeps_existing_geometric_reseal_contract(self):
        decision = decide(initial_tf_drift_stop())
        self.assertTrue(decision.eligible, decision.reason)
        self.assertEqual(decision.recovery_action, "fresh_localization_reseal")

    def test_optional_acquisition_counters_cannot_conceal_stale_history(self):
        counters = ("non_acquisition_failure_count", "waitable_stale_sample_count")
        for make_details in (initial_map_tf_stop, initial_tf_drift_stop):
            for role in ("execution_pose", "global_consistency"):
                with self.subTest(kind=make_details.__name__, role=role):
                    details = make_details()
                    self.assertTrue(decide(details).eligible)  # Older schema 2.
                    edge = details["initial_tf_acquisition"]["edges"][role]
                    edge.update(dict.fromkeys(counters, 0))
                    self.assertTrue(decide(details).eligible)
                    for counter in counters:
                        for bad_value in (1, -1, True, 0.0, None):
                            changed = deepcopy(details)
                            changed["initial_tf_acquisition"]["edges"][role][counter] = bad_value
                            decision = decide(changed)
                            self.assertFalse(decision.eligible)
                            self.assertEqual(decision.reason, "invalid_initial_map_tf_acquisition_counters")
                        changed = deepcopy(details)
                        del changed["initial_tf_acquisition"]["edges"][role][counter]
                        self.assertFalse(decide(changed).eligible)

    def test_typed_drift_requires_complete_matching_context_and_acquired_global_edge(self):
        for path, value in (
            (("execution_context", "certificate_sha256"), "b" * 64),
            (("execution_context", "map_frame"), "another_map"),
            (("execution_context", "frozen_map_from_odom", "x_m"), 1),
            (("execution_context", "max_map_from_odom_translation_drift_m"), 0.11),
            (("edges", "global_consistency", "successful_sample_count"), 0),
            (("edges", "global_consistency", "current_ready"), False),
            (("edges", "global_consistency", "last_sample", "reason"), "lookup_exception"),
            (("edges", "global_consistency", "last_sample", "age_sec"), 1.1),
            (("admission_failure_seen",), False),
        ):
            with self.subTest(path=path):
                details = initial_tf_drift_stop()
                target = details["initial_tf_acquisition"]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertFalse(decide(details).eligible)
        for state in ({"schema_version": 2}, initial_map_tf_stop()["initial_tf_acquisition"]):
            details = _stop_details()
            details["initial_tf_acquisition"] = state
            self.assertFalse(decide(details).eligible)


if __name__ == "__main__":
    unittest.main()
