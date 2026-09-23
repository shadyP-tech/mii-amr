"""Only an explicitly bound Start heading turn admits zero route length."""

import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.control.safety_checks import validate_route_diagnostics_json
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg


class StationaryStartRouteAdmissionTest(unittest.TestCase):
    def rows(self):
        return [
            dict(leg_index=0, point_index=index, world_x_m=1., world_y_m=2.,
                 cumulative_length_m=0., yaw_rad="" if index == 0 else 1.57,
                 stationary_turn="true", route_kind="admitted_candidate_pose",
                 simulation_only="false")
            for index in range(2)
        ]

    def write(self, directory, rows):
        path = Path(directory) / "route.csv"
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return path

    def diagnostics(self):
        return {
            "metadata": {"route_kind": "admitted_candidate_pose", "stationary_turn": True},
            "legs": [{"diagnostics": {"status": "ok"}, "failure": None,
                      "route_point_count": 2, "route_length_m": 0.}],
        }

    def validate(self, leg, payload):
        return validate_route_diagnostics_json(
            Path("unused.json"), 0, csv_point_count=2,
            diagnostics_payload=payload, route_leg=leg,
        )

    def test_scoped_stationary_csv_preserves_two_points_and_matching_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            leg = load_route_leg(self.write(directory, self.rows()), 0, thinning_min_spacing_m=.15)
            self.assertTrue(leg.stationary_turn)
            self.assertEqual(leg.route_length_m, 0.)
            self.assertEqual(len(leg.executable_waypoints), 2)
            self.assertTrue(self.validate(leg, self.diagnostics()).ok)

    def test_missing_or_wrong_stationary_provenance_does_not_admit_zero_length(self):
        mutations = (
            {"stationary_turn": "false"}, {"stationary_turn": ""},
            {"stationary_turn": "garbage"}, {"route_kind": "stand_discovery_corridor"},
            {"route_kind": ""}, {"simulation_only": "true"},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                rows = self.rows()
                for row in rows:
                    row.update(mutation)
                with self.assertRaises(ValueError):
                    load_route_leg(self.write(directory, rows), 0)

    def test_stationary_geometry_cannot_hide_translation_or_missing_heading(self):
        mutations = ({"world_x_m": 1.001}, {"world_y_m": 2.001},
                     {"cumulative_length_m": .001}, {"yaw_rad": ""},
                     {"stationary_turn": "false"})
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                rows = self.rows()
                rows[1].update(mutation)
                with self.assertRaises(ValueError):
                    load_route_leg(self.write(directory, rows), 0)
        for size in (1, 3):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                rows = self.rows()
                if size == 1:
                    rows.pop(0)
                    rows[0]["point_index"] = 0
                else:
                    rows.append({**rows[-1], "point_index": 2})
                with self.assertRaises(ValueError):
                    load_route_leg(self.write(directory, rows), 0)

    def test_diagnostics_flag_alone_does_not_relax_generic_motion_length_check(self):
        status = validate_route_diagnostics_json(
            Path("unused.json"), 0, csv_point_count=2, diagnostics_payload=self.diagnostics(),
        )
        self.assertFalse(status.ok)
        self.assertIn("positive for motion", status.failures[0])

    def test_stationary_diagnostics_must_match_csv_flag_kind_and_length(self):
        with tempfile.TemporaryDirectory() as directory:
            leg = load_route_leg(self.write(directory, self.rows()), 0)
            for value in (False, "true", None):
                with self.subTest(stationary_turn=value):
                    payload = self.diagnostics()
                    payload["metadata"]["stationary_turn"] = value
                    self.assertFalse(self.validate(leg, payload).ok)
            payload = self.diagnostics()
            payload["metadata"]["route_kind"] = "stand_discovery_corridor"
            self.assertFalse(self.validate(leg, payload).ok)
            payload = self.diagnostics()
            payload["legs"][0]["route_length_m"] = .01
            self.assertFalse(self.validate(leg, payload).ok)
            self.assertFalse(self.validate(replace(leg, stationary_turn=False), self.diagnostics()).ok)


if __name__ == "__main__":
    unittest.main()
