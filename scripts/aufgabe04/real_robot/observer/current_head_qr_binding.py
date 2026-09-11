"""Keep head-enabled camera registration tied to that head's own QR symbol."""

from dataclasses import replace

from scripts.aufgabe04.perception.stand_axis.geometry import order_corners


def bind_qr_to_current_head(binding, observations, *, head_corners, head_association):
    """Require an enclosed symbol and overlapping samples of the same scan cluster.

    The independent QR-ray association already checked shape, range, freshness
    and uniqueness. Its narrow cone may include a different subset of the
    head's cluster, so require overlap rather than identical sample sets.
    Both quadrilaterals use the same selected crop coordinates.
    """
    if not binding.accepted:
        return binding
    if not head_association.accepted:
        reason = "qr_current_head_unassociated"
        shared = ()
    else:
        # These corners have passed current-head and decoded-QR validation.
        quad = observations[0].corners
        head_corners = order_corners(head_corners)
        enclosed = all(
            all((b.u_px - a.u_px) * (y - a.v_px)
                - (b.v_px - a.v_px) * (x - a.u_px) >= -1.e-6
                for a, b in zip(head_corners, head_corners[1:] + head_corners[:1]))
            for x, y in quad
        )
        head = head_association.lidar_association.search_association
        qr = (binding.association or {}).get("search_association") or binding.association or {}
        same_scan = (qr.get("scan_frame_id") == head.scan_frame_id
                     and qr.get("scan_stamp_sec") == head.scan_stamp_sec)
        shared = tuple(sorted(set(head.selected_cluster_source_indices)
                              & set(qr.get("selected_cluster_source_indices", ()))))
        reason = ("qr_outside_current_head" if not enclosed else
                  "qr_current_head_cluster_mismatch" if not same_scan or not shared else
                  "qr_inside_current_head_same_lidar_cluster")
    accepted = reason == "qr_inside_current_head_same_lidar_cluster"
    return replace(
        binding, accepted=accepted,
        reason=binding.reason if accepted else reason,
        qr_texts_for_evidence=binding.qr_texts_for_evidence if accepted else (),
        current_head_binding={"accepted": accepted, "reason": reason,
                              "shared_scan_source_indices": list(shared),
                              "motion_authorized": False},
    )
