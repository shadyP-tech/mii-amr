import io
import json
import unittest
from http.client import IncompleteRead
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from scripts.aufgabe04.task_client.station_tour_client import StationTourClient, StationTourHttpError


class StationTourClientTests(unittest.TestCase):
    def test_actual_paths_methods_and_json_bodies(self):
        client = StationTourClient(robot_id="robot/name", base_url="http://10.42.0.1:8000/")
        with patch("scripts.aufgabe04.task_client.station_tour_client.urlopen",
                   side_effect=lambda *args, **kwargs: io.BytesIO(b'{}')) as request:
            client.randomize_plan(qr_count=4, stations=3)
            client.get_plan()
            client.get_qr_mappings()
            client.report_arrival("QR_001", client_event_id="event-1")
        requests = [call.args[0] for call in request.call_args_list]
        self.assertEqual([item.method for item in requests], ["POST", "GET", "GET", "POST"])
        self.assertEqual([item.full_url for item in requests], [
            "http://10.42.0.1:8000/api/v1/robots/robot%2Fname/plan/randomize",
            "http://10.42.0.1:8000/api/v1/robots/robot%2Fname/plan",
            "http://10.42.0.1:8000/api/v1/robots/robot%2Fname/qr-mappings",
            "http://10.42.0.1:8000/api/v1/qr/QR_001/scan",
        ])
        self.assertEqual(json.loads(requests[0].data), {"qr_count": 4, "stations": 3})
        self.assertEqual(json.loads(requests[3].data), {"robot_id": "robot/name", "client_event_id": "event-1"})

    def test_lost_mutation_response_is_ambiguous_and_never_retried(self):
        client = StationTourClient(robot_id="robot")
        with patch("scripts.aufgabe04.task_client.station_tour_client.urlopen", side_effect=URLError("timeout")) as request:
            with self.assertRaises(StationTourHttpError) as error:
                client.report_arrival("Start", client_event_id="persisted-event")
        self.assertTrue(error.exception.write_outcome_unknown)
        self.assertEqual(request.call_count, 1)

    def test_http_error_classifies_write_uncertainty(self):
        for status, uncertain in ((422, False), (500, True)):
            with self.subTest(status=status):
                failure = HTTPError("http://server", status, "error", {}, io.BytesIO(b'{"detail":"failed"}'))
                with patch("scripts.aufgabe04.task_client.station_tour_client.urlopen", side_effect=failure):
                    with self.assertRaises(StationTourHttpError) as error:
                        StationTourClient(robot_id="robot").randomize_plan(qr_count=4, stations=3)
                self.assertEqual(error.exception.write_outcome_unknown, uncertain)
                self.assertEqual(error.exception.status_code, status)

    def test_truncated_post_response_is_ambiguous_without_retry(self):
        class TruncatedResponse(io.BytesIO):
            def read(self, size=-1):
                raise IncompleteRead(b'{"accepted":', 30)
        with patch("scripts.aufgabe04.task_client.station_tour_client.urlopen",
                   return_value=TruncatedResponse()) as request:
            with self.assertRaises(StationTourHttpError) as error:
                StationTourClient(robot_id="robot").report_arrival("Start", client_event_id="original-event")
        self.assertTrue(error.exception.write_outcome_unknown)
        self.assertEqual(request.call_count, 1)

    def test_invalid_mutation_json_is_ambiguous(self):
        for data in (b"not json", b'{"value":NaN}', b'{"accepted":false,"accepted":true}'):
            with self.subTest(data=data), patch("scripts.aufgabe04.task_client.station_tour_client.urlopen",
                                               return_value=io.BytesIO(data)):
                with self.assertRaises(StationTourHttpError) as error:
                    StationTourClient(robot_id="robot").report_arrival("Start", client_event_id="id")
                self.assertTrue(error.exception.write_outcome_unknown)

    def test_invalid_local_request_has_no_http_effect(self):
        with patch("scripts.aufgabe04.task_client.station_tour_client.urlopen") as request:
            for count, stations in ((3, 3), (11, 3), (4, 2), (True, 3)):
                with self.subTest(count=count, stations=stations), self.assertRaises(ValueError):
                    StationTourClient(robot_id="robot").randomize_plan(qr_count=count, stations=stations)
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
