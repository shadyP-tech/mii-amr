"""Color-agnostic Canny preprocessing and topology-only morphology."""

from __future__ import annotations


def _largest_external_bounding_area(cv2, edges) -> float:
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = 0.0
    for contour in contours:
        _x, _y, width, height = cv2.boundingRect(contour)
        largest = max(largest, float(width * height))
    return largest


def _edge_topology_hypotheses(
    cv2,
    topology_seed,
    *,
    close_kernel: int,
    close_iterations: int,
    include_gap_recovery: bool,
    edge_exclusion_mask=None,
):
    """Return bounded morphology variants used only to locate edge topology.

    The configured variant stays first. In silhouette mode, two conservative
    alternatives bridge the one- or two-pixel head-border gaps seen in Gazebo
    without ever changing the raw Canny evidence used for rectangle support.
    """

    specifications = [(close_kernel, close_iterations)]
    if include_gap_recovery:
        specifications.extend(((3, 2), (5, 1)))

    hypotheses = []
    seen = set()
    for kernel_size, iterations in specifications:
        specification = (
            (int(kernel_size), int(iterations))
            if kernel_size > 1 and iterations > 0
            else (1, 0)
        )
        if specification in seen:
            continue
        seen.add(specification)

        edges = topology_seed.copy()
        if specification[1] > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (specification[0], specification[0]),
            )
            edges = cv2.morphologyEx(
                edges,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=specification[1],
            )
        # Apply the synchronized wall mask after closing as well. Otherwise a
        # removed wall line can be painted back across the exclusion band by a
        # neighbouring foreground edge.
        if edge_exclusion_mask is not None:
            edges = cv2.bitwise_and(
                edges,
                cv2.bitwise_not(edge_exclusion_mask),
            )
        hypotheses.append(edges)
    return hypotheses


def _canny_edges_from_frame(
    cv2,
    frame,
    *,
    edge_preprocess: str,
    blur_kernel: int,
    canny_low: int,
    canny_high: int,
):
    """Extract edges without assigning semantic meaning to any color.

    ``channel_union`` applies identical blur/Canny operations independently to
    B, G, and R and takes their logical union. It is invariant to channel
    permutation and therefore remains color-agnostic, while retaining borders
    whose foreground/background luminance happens to be almost identical.
    """

    if edge_preprocess == "channel_union":
        edge_frame = frame
        if blur_kernel > 1:
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            edge_frame = cv2.GaussianBlur(
                edge_frame,
                (blur_kernel, blur_kernel),
                0,
            )
        channels = cv2.split(edge_frame)
        if not channels:
            raise ValueError("frame must contain at least one image channel")
        edges = cv2.Canny(channels[0], canny_low, canny_high)
        for channel in channels[1:]:
            edges = cv2.bitwise_or(
                edges,
                cv2.Canny(channel, canny_low, canny_high),
            )
        return edges

    edge_input = _edge_input_image(
        cv2,
        frame,
        edge_preprocess=edge_preprocess,
        blur_kernel=blur_kernel,
    )
    return cv2.Canny(edge_input, canny_low, canny_high)


def _topology_edges_from_frame(
    cv2,
    frame,
    *,
    edge_preprocess: str,
    canny_low: int,
    canny_high: int,
    fallback_edges,
):
    """Build a texture-suppressed locator without changing measurement edges."""

    if edge_preprocess != "channel_union":
        return fallback_edges.copy()

    # Apply the existing low-pass outer-border preparation to each channel
    # independently. The operation is symmetric under any B/G/R permutation,
    # but QR modules are attenuated before the small morphology hypotheses can
    # accidentally connect them to the head outline.
    channels = cv2.split(frame)
    if not channels:
        raise ValueError("frame must contain at least one image channel")
    edges = cv2.Canny(
        _outer_border_edge_input(cv2, channels[0]),
        canny_low,
        canny_high,
    )
    for channel in channels[1:]:
        edges = cv2.bitwise_or(
            edges,
            cv2.Canny(
                _outer_border_edge_input(cv2, channel),
                canny_low,
                canny_high,
            ),
        )
    return edges


def _topology_supported_measurement_edges(
    cv2,
    raw_edges,
    topology_edges,
    *,
    min_edge_height_px: float,
):
    """Keep precise Canny pixels only near the gated outer-border topology.

    The adaptive foreground gate and the low-frequency topology image remove
    background clutter and most QR texture.  The final fit must still use
    actual pre-morphology Canny pixels, but unrestricted raw Canny can escape
    the proposal and snap a side to a radiator rail or an interior QR edge.
    This corridor preserves raw-pixel precision while enforcing the same
    foreground/topology ownership shown in the stand-axis edge window.
    """

    if raw_edges.shape[:2] != topology_edges.shape[:2]:
        raise ValueError("raw_edges and topology_edges must have matching shapes")
    corridor_radius_px = max(
        2,
        min(5, int(round(0.25 * float(min_edge_height_px)))),
    )
    kernel_size = 2 * corridor_radius_px + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    topology_corridor = cv2.dilate(topology_edges, kernel, iterations=1)
    return cv2.bitwise_and(raw_edges, topology_corridor)


def _edge_input_image(cv2, frame, *, edge_preprocess: str, blur_kernel: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if edge_preprocess == "outer_border":
        return _outer_border_edge_input(cv2, gray)
    if edge_preprocess == "gray":
        if blur_kernel > 1:
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        return gray
    raise ValueError(f"unsupported edge preprocess mode: {edge_preprocess}")


def _outer_border_edge_input(cv2, gray):
    # Suppress QR-code texture before Canny. The square outline and stem are
    # low-frequency structure; QR modules are high-frequency interior texture.
    smoothed = cv2.GaussianBlur(gray, (9, 9), 0)
    smoothed = cv2.medianBlur(smoothed, 7)
    return cv2.bilateralFilter(smoothed, 9, 50, 50)
