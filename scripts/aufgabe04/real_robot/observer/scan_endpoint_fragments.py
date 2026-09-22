"""Eligibility for witnessed endpoint fragments, never circular scan authority."""

from scripts.aufgabe04.perception.scan_endpoint_fragments import bounded_endpoint_fragments


ENDPOINT_WITNESS_KIND = "scan_endpoint_fragments_witnessed"
INTERNAL_WITNESS_KIND = "one_internal_missing_beam_witnessed"


def endpoint_fragments(scan, clusters):
    """Bound the original scan gap; a caller must still prove real continuity.

    The LDS receipts report a near-one-step header seam while their indexed
    seam varies between one and two steps. Do not repair their angles or join
    their groups here. Only three independently valid scans can witness this
    limited missing interval, with the unchanged spatial and freshness gates.
    """
    return bounded_endpoint_fragments(scan,
        tuple(tuple(sample.index for sample in cluster.samples) for cluster in clusters))
