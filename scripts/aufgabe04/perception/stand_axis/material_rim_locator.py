"""Current material components locate heads; untouched raw pixels measure them.

Palette labels are never station identities. Each material is kept separate so
paper from another palette cannot close a broken rim. This optional cold-search
path retains competing components and falls back when no complete rim exists.
"""
from dataclasses import dataclass

from scripts.aufgabe04.perception.stand_color_support import STAND_EDGE_PALETTE
from scripts.aufgabe04.perception.mask_processing import build_mask_for_ranges
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import check_head_acquisition_deadline

MAX_RIM_COMPONENTS = 8


@dataclass(frozen=True)
class MaterialRim:
    label: str
    support: object
    holes: tuple


def _box(x, y, w, h):
    return tuple(ImagePoint(float(u), float(v)) for u, v in
                 ((x,y),(x+w,y),(x+w,y+h),(x,y+h)))


def material_rims(cv2, frame, search, *, deadline=None):
    """Require an enclosed, head-scale hole in one chromatic material.

    Hole coordinates are only locators. No contour/morphology coordinate is
    returned as a physical measurement. Achromatic or broken rims use the
    ordinary raw-edge acquisition path.
    """
    import numpy as np
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    palette = {}
    for item in STAND_EDGE_PALETTE:
        if item.label != 'light-grey':
            palette.setdefault(item.label, []).append(item)
    result = []
    for label, ranges in palette.items():
        check_head_acquisition_deadline(deadline, 'material_rim_components')
        mask = np.zeros(frame.shape[:2], np.uint8)
        for item in ranges:
            # ColorRange's field names are shared with build_mask_for_ranges.
            mask |= build_mask_for_ranges(cv2, np, hsv, (item,))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if len(contours) > 1024:
            raise ValueError('material rim contour budget exceeded')
        if hierarchy is None:
            continue
        parents = {}
        for index, contour in enumerate(contours):
            check_head_acquisition_deadline(deadline, 'material_rim_holes')
            parent = int(hierarchy[0,index,3])
            if parent < 0:
                continue
            x,y,w,h = cv2.boundingRect(contour)
            if min(w,h) < 24 or not search.accepts_hint(_box(x,y,w,h)):
                continue
            if search.edge_region is not None and not search.edge_region.contains(_box(x,y,w,h)):
                continue
            parents.setdefault(parent, []).append((x,y,w,h))
        if not parents:
            continue
        if sum(len(r.holes) for r in result)+sum(map(len,parents.values())) > MAX_RIM_COMPONENTS:
            raise ValueError('material rim component budget exceeded')
        exterior = np.zeros_like(mask)
        selected = np.zeros_like(mask)
        for parent in parents:
            cv2.drawContours(exterior, contours, parent, 255, 1)
            cv2.drawContours(selected, contours, parent, 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        support = cv2.bitwise_and(cv2.dilate(exterior,kernel),cv2.dilate(mask & selected,kernel))
        result.append(MaterialRim(label, support, tuple(hole for holes in parents.values() for hole in holes)))
    return tuple(result)


def unexplained_raw_rectangle(cv2, raw, selected, search, *, frame, model_profile,
                              proposal_filter=None, deadline=None):
    """Retain complete raw contours outside the selected rim, including gray.

    This is a conservative competitor veto, not a geometry producer. Printed
    contours inside this complete material-owned head cannot donate corners to
    the selected measurement. Independent material components are compared by
    the caller before this check.
    """
    from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import _bounded_quad
    from scripts.aufgabe04.perception.stand_axis.head_outer_border import _encloses
    from scripts.aufgabe04.perception.stand_axis.raw_support import _quadrilateral_edge_support
    source = raw if search.edge_region is None else search.edge_region.locator_edges(raw)
    contours = cv2.findContours(source,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours) > 1024:
        return True
    for contour in contours:
        check_head_acquisition_deadline(deadline, 'material_rim_raw_competitors')
        length = cv2.arcLength(contour,True)
        if length < 96:
            continue
        for fraction in (.015,.025,.04):
            quad = cv2.approxPolyDP(contour,fraction*length,True)
            if len(quad)!=4 or not cv2.isContourConvex(quad):
                continue
            corners = _bounded_quad(quad.reshape(-1,2),raw.shape)
            if corners is None or not search.accepts_hint(corners):
                continue
            source_support = getattr(proposal_filter, 'source_support', None)
            if source_support is not None and not source_support.accepts(corners):
                continue
            if _encloses(selected,corners,tolerance=10.):
                continue
            if _quadrilateral_edge_support(cv2,raw,corners).accepted:
                return True
    # Open outer contours (a stem may join the rim) still have paired raw
    # rails. Run the ordinary bounded locator with zero verification slots:
    # any unowned hypothesis vetoes the shortcut, including achromatic heads.
    from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import acquire_cold_head_proposal
    class Unowned:
        source_support = getattr(proposal_filter, 'source_support', None)

        def __call__(self, proposal):
            return (not _encloses(selected, proposal.corners, tolerance=10.)
                    and (proposal_filter is None or
                         getattr(proposal_filter,'preview',proposal_filter)(proposal)))
        preview = __call__
    # The locator expects source_support to be absent rather than None.
    gate = Unowned()
    if gate.source_support is None:
        del Unowned.source_support
    competitors = acquire_cold_head_proposal(cv2,frame,raw_edges=raw,
        candidate_search=search,model_profile=model_profile,proposal_filter=gate,
        _verification_limit=0,deadline_monotonic_sec=deadline)
    return ('budget' in competitors.reason or 'deadline' in competitors.reason
            or 'ambiguous' in competitors.reason or competitors.proposal is not None)
