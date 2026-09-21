"""Recorded three-pass search keeps identical decisions with one extraction."""

from collections import Counter
from unittest.mock import patch

import pytest

from scripts.aufgabe04.perception.stand_axis.head_search_inputs import HeadSearchInputs


def test_cache_rejects_another_frame_edges_or_region():
    context = [object(), object(), object()]
    cache = HeadSearchInputs(*context)
    for index in range(3):
        changed = context.copy()
        changed[index] = object()
        with pytest.raises(ValueError):
            cache.check_context(*changed)


def test_three_search_orderings_share_extraction_without_changing_decisions():
    from tests.aufgabe04.test_stopped_target_search import (
        recorded_options, FIXTURE, cv2, np, rectify_bgr_frame,
    )
    from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import acquire_cold_head_proposal
    from scripts.aufgabe04.perception.stand_axis.candidate_head_search import CandidateHeadSearch

    options, _, calibration, _ = recorded_options(4)
    frame = rectify_bgr_frame(cv2.imread(str(FIXTURE / "frame_000004.jpg")), calibration, cv2, np)
    projection = options["original_projection"]
    height = options["intrinsics"].fy_px * options["model_profile"].head_height_m / projection.depth_m
    search = CandidateHeadSearch.optional(projection.u_px, projection.v_px, height,
                                          max_center_offset_ratio=1.5)
    original = HeadSearchInputs.get
    results, counts = [], []
    for cached in (False, True):
        extractions = Counter()
        def get(self, name, extract):
            def counted():
                extractions[name] += 1
                return extract()
            return original(self, name, counted) if cached else counted()
        with patch.object(HeadSearchInputs, "get", get):
            results.append(acquire_cold_head_proposal(cv2, frame,
                model_profile=options["model_profile"], candidate_search=search))
        counts.append(extractions)
    assert results[0] == results[1]
    for name in ("contours", "gray", "lsd", "hough", "distance"):
        assert counts[0][name] == 3
        assert counts[1][name] == 1
