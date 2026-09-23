"""Bounded rim search followed by independent full-raw border certification."""
from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import (
    bounded_head_acquisition, check_head_acquisition_deadline)
from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import acquire_cold_head_proposal, MAX_RAW_VERIFICATIONS, MAX_IMAGE_PIXELS
from scripts.aufgabe04.perception.stand_axis.head_proposal import HeadProposalResult, _proposal, _extent
from scripts.aufgabe04.perception.stand_axis.head_border_families import CurrentBorderFamilies
from scripts.aufgabe04.perception.stand_axis.current_head_refinement import refine_current_physical_head
from scripts.aufgabe04.perception.stand_axis.current_head_refinement_proof import capture_current_head_refinement
from scripts.aufgabe04.perception.stand_axis.material_rim_locator import material_rims, unexplained_raw_rectangle


@bounded_head_acquisition
def acquire_material_rim(cv2, frame, *, raw_edges, candidate_search, model_profile,
                         proposal_filter=None, refinement_out=None, budget_out=None, deadline_monotonic_sec=None):
    if candidate_search is None or candidate_search.pixel_size is None:
        return None
    deadline = deadline_monotonic_sec
    diagnostic = dict(policy='material_rim_then_full_raw', materials=[],
        raw_measurement_unchanged=True, color_is_identity=False, comparison_complete=False)
    used = 0
    if budget_out is not None:
        budget_out["used"] = 0
    def result(reason, proposal=None):
        diagnostic['raw_verifications'] = used
        return HeadProposalResult(proposal,reason,0,used,'material_rim',diagnostic)
    if frame.shape[0]*frame.shape[1] > MAX_IMAGE_PIXELS:
        return result("head_material_rim_budget_exceeded")
    try:
        rims = material_rims(cv2,frame,candidate_search,deadline=deadline)
    except ValueError:
        return result('head_material_rim_budget_exceeded')
    if not rims:
        return None
    verified = []
    families = CurrentBorderFamilies(raw_edges,frame)
    for rim in rims:
        check_head_acquisition_deadline(deadline,'material_rim_acquisition')
        remaining = MAX_RAW_VERIFICATIONS-used-1  # Reserve one full-raw validation.
        if remaining < 1:
            return result('head_material_rim_budget_exceeded')
        located = acquire_cold_head_proposal(cv2,frame,
            raw_edges=cv2.bitwise_and(raw_edges,rim.support), candidate_search=candidate_search,
            model_profile=model_profile,proposal_filter=proposal_filter,
            color_support_mask=rim.support,_verification_limit=remaining,
            deadline_monotonic_sec=deadline)
        used += located.raw_verifications
        if budget_out is not None:
            budget_out["used"] = used
        diagnostic['materials'].append(dict(label=rim.label,reason=located.reason,
            strict_verifications=located.raw_verifications))
        if located.proposal is None:
            if 'ambiguous' in located.reason or 'budget' in located.reason or 'deadline' in located.reason:
                return result(located.reason)
            continue
        # Never reuse the masked acquisition's transient refinement proof.
        check_head_acquisition_deadline(deadline,'material_rim_full_raw_validation')
        measured,outer,seed = refine_current_physical_head(cv2,raw_edges,
            model_profile=model_profile,proposal_corners=located.proposal.corners,
            deadline_monotonic_sec=deadline)
        used += 1
        if budget_out is not None:
            budget_out["used"] = used
        if (not measured.accepted or not outer.accepted or measured.corners is None
                or not families.same(located.proposal.corners,measured.corners)
                or not candidate_search.accepts_measurement(measured.corners)):
            return result('head_material_rim_full_raw_disagrees')
        _,height,center = _extent(measured.corners)
        proposal = _proposal(measured,raw_edges.shape,expected_height=height,expected_center=center)
        if proposal_filter is not None and not proposal_filter(proposal):
            return result('head_proposal_candidate_association_rejected')
        verified.append((proposal,measured,outer,seed))
    if not verified:
        return None
    # Pairwise evidence, not transitive grouping or color priority.
    if any(not families.same(a[0].corners,b[0].corners)
           for i,a in enumerate(verified) for b in verified[i+1:]):
        return result('head_proposal_ambiguous')
    proposal,measured,outer,seed = verified[0]
    if unexplained_raw_rectangle(cv2,raw_edges,proposal.corners,candidate_search,
            frame=frame,model_profile=model_profile,proposal_filter=proposal_filter,deadline=deadline):
        return result('head_proposal_ambiguous')
    check_head_acquisition_deadline(deadline,'material_rim_complete')
    diagnostic['comparison_complete'] = True
    if refinement_out is not None:
        refinement_out['selected'] = capture_current_head_refinement(frame,raw_edges,
            model_profile=model_profile,refinement=measured,outer_recovery=outer,seed=seed)
    return result('current_head_proposal',proposal)
