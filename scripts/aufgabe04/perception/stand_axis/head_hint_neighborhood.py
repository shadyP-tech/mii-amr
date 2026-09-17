"""Small current-edge locator neighborhoods, never measured head borders.

Offset peaks repair an observed quadrilateral's search location. The caller
retains the original hint and must compare and strictly fit returned variants;
this helper neither selects a physical frame nor supplies admission evidence.
"""

import math

from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import check_head_acquisition_deadline
from scripts.aufgabe04.perception.stand_axis.head_border_seed import validate_current_head_proposal
from scripts.aufgabe04.perception.stand_axis.metric_edge_association import metric_corner_arm_support
from scripts.aufgabe04.perception.stand_axis.model_refinement import model_corridor_half_width_px
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint


MAX_HINT_NEIGHBORS = 8
MIN_SIDE_SUPPORT = .75


def observed_head_hint_neighborhood(cv2, raw_edges, corners, *, model_profile,
                                    deadline_monotonic_sec=None, diagnostics=None):
    """Return at most eight coherent offsets supported by current raw pixels.

    Each side retains its observed direction. Integer normal offsets within the
    ordinary model corridor locate current support peaks; a subpixel sampling band
    only tolerates rasterization. Intersections remain unverified search hints.
    Missing sides or corner arms cannot be supplied by the model or other rails.
    """
    import numpy as np

    def check(stage):
        check_head_acquisition_deadline(deadline_monotonic_sec, stage)

    check("head_hint_neighborhood")
    if diagnostics is not None:
        diagnostics.update(max_variants=MAX_HINT_NEIGHBORS, returned_variants=0,
                           supplies_measurement=False, raw_pixels_changed=False)
    if raw_edges is None or raw_edges.ndim != 2:
        return ()
    try:
        original = validate_current_head_proposal(corners, frame_shape=raw_edges.shape)
    except (TypeError, ValueError):
        return ()
    if original is None:
        return ()
    corridor = model_corridor_half_width_px(original, model_profile=model_profile,
                                             pose_reprojection_rmse_px=0.)
    points = np.asarray([(p.u_px, p.v_px) for p in original], float)
    delta = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(delta, axis=1)
    tangent = delta / lengths[:, None]
    normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    # Orient all normals inward, independent of the polygon's winding.
    normals *= np.where(np.sum(normals * (points.mean(axis=0) - points), axis=1) >= 0.,
                        1., -1.)[:, None]
    offsets = np.arange(-math.floor(corridor), math.floor(corridor) + 1, dtype=float)
    fractions = np.linspace(.10, .90, 48)
    rows, cols = raw_edges.shape
    modes = []
    for point, direction, normal in zip(points, delta, normals):
        check("head_hint_offset_support")
        samples = (point + fractions[None, :, None, None] * direction
                   + (offsets[:, None, None, None]
                      + np.asarray((-.65, 0., .65))[None, None, :, None]) * normal)
        pixels = np.rint(samples).astype(np.int32)
        xs, ys = pixels[..., 0], pixels[..., 1]
        valid = (xs >= 0) & (xs < cols) & (ys >= 0) & (ys < rows)
        occupied = np.zeros(valid.shape, bool)
        occupied[valid] = raw_edges[ys[valid], xs[valid]] > 0
        support = occupied.any(axis=2).mean(axis=1)
        central = occupied[:, :, 1].mean(axis=1)
        eligible = np.flatnonzero(support >= MIN_SIDE_SUPPORT)
        if not len(eligible):
            return ()
        # One contiguous supported band is one offset mode. Raster aliases of
        # the same rail must not fill the bounded neighborhood with duplicates.
        runs = np.split(eligible, np.flatnonzero(np.diff(eligible) > 1) + 1)
        side = []
        for run in runs:
            index = max(run, key=lambda i: (central[i], support[i], -abs(offsets[i])))
            side.append((float(offsets[index]), float(support[index]), float(central[index])))
        modes.append(side)
    if diagnostics is not None:
        diagnostics.update(corridor_half_width_px=corridor,
                           observed_modes_per_side=tuple(len(side) for side in modes))

    def choose(side, policy):
        if policy in ("inward", "inward_far", "outward", "outward_far"):
            sign = 1 if policy.startswith("inward") else -1
            eligible = [item for item in side if sign * item[0] >= 0.]
            if not eligible:
                return None
            return max(eligible, key=lambda item: ((1 if policy.endswith("far") else -1)
                                                   * abs(item[0]), item[2], item[1]))[0]
        return max(side, key=lambda item: ((item[2], item[1], -abs(item[0]))
                    if policy == "strongest" else (-abs(item[0]), item[2], item[1])))[0]

    policies = [(name,) * 4 for name in
                ("nearest", "strongest", "inward", "inward_far", "outward", "outward_far")]
    policies += [("inward", "outward", "inward", "outward"),
                 ("outward", "inward", "outward", "inward")]
    variants, seen = [], {tuple(original)}
    for policy in policies:
        check("head_hint_offset_intersections")
        shift = [choose(side, name) for side, name in zip(modes, policy)]
        if any(value is None for value in shift):
            continue
        moved = points + np.asarray(shift)[:, None] * normals
        proposed = []
        for index in range(4):
            previous = (index - 1) % 4
            matrix = np.stack((tangent[previous], -tangent[index]), axis=1)
            if abs(np.linalg.det(matrix)) < 1.e-6:
                break
            along = np.linalg.solve(matrix, moved[index] - moved[previous])[0]
            corner = moved[previous] + along * tangent[previous]
            proposed.append(ImagePoint(*map(float, corner)))
        try:
            variant = validate_current_head_proposal(proposed, frame_shape=raw_edges.shape)
        except (TypeError, ValueError):
            continue
        if variant in seen:
            continue
        seen.add(variant)
        check("head_hint_corner_support")
        if metric_corner_arm_support(cv2, raw_edges, variant).accepted:
            variants.append(variant)
    check("head_hint_neighborhood_complete")
    if diagnostics is not None:
        diagnostics["returned_variants"] = len(variants)
    return tuple(variants)
