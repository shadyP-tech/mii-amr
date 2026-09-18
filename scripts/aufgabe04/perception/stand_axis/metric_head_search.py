"""Distance/model priors for pixel search, never synthetic measured borders."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ProjectedHeadSize:
    depth_m: float
    depth_uncertainty_m: float
    height_px: float
    frontal_width_px: float
    min_height_px: float
    max_height_px: float
    max_width_px: float

    def accepts(self, width, height, *, refinement_allowance=False):
        # Rough locators can move during the existing raw-border refinement.
        low = .85 * height if refinement_allowance else height
        high = 1.15 * 1.25 * height if refinement_allowance else height
        width_low = .85 * width if refinement_allowance else width
        return (low <= self.max_height_px and high >= self.min_height_px
                and width_low <= self.max_width_px)

    def diagnostics(self):
        return {**self.__dict__, "policy": "metric_depth_pixel_size",
                "supplies_corners": False, "supplies_yaw": False}


def projected_head_size(*, model_profile, depth_m, fx, fy,
                        depth_uncertainty_m=.02, center_normalized=(0., 0.),
                        camera_vertical=(0., 1., 0.), pixel_margin=4.):
    """Bound projected upright side lengths using calibrated camera depth.

    The vertical vector is the stand's world/base up direction expressed in
    optical-camera coordinates. Unknown yaw is covered by the head radius;
    distance/position error and model tolerance expand the search interval.
    A frontal width is reported, but no lower width is imposed on oblique heads.
    """
    values = (depth_m, fx, fy, depth_uncertainty_m, pixel_margin,
              *center_normalized, *camera_vertical)
    if (any(not math.isfinite(v) for v in values) or min(depth_m, fx, fy) <= 0
            or min(depth_uncertainty_m, pixel_margin) < 0
            or not model_profile.committable or model_profile.environment != "physical"):
        raise ValueError("metric head search requires finite physical camera depth and calibration")
    norm = math.sqrt(sum(v*v for v in camera_vertical))
    if not .99 <= norm <= 1.01:
        raise ValueError("camera vertical must be a unit vector")
    ux, uy, uz = (v/norm for v in camera_vertical)
    x, y = (v*depth_m for v in center_normalized)
    h, w, d, tol = (model_profile.head_height_m, model_profile.head_width_m,
                    model_profile.head_depth_m, model_profile.tolerance_m)
    radius = math.hypot(w/2., d) + depth_uncertainty_m + tol
    near, far = depth_m-radius, depth_m+radius
    hmin, hmax = h-tol, h+tol
    if hmin <= 0 or near <= .5*hmax*abs(uz) + .01:
        raise ValueError("possible head volume crosses the camera plane")
    # Exact projected length of a vertical segment at its nominal centre:
    # H ||(fx(ux Z-X uz), fy(uy Z-Y uz))|| / (Z²-(H uz/2)²).
    numerator = math.hypot(fx*(ux*depth_m-x*uz), fy*(uy*depth_m-y*uz))
    error = radius*math.hypot(fx*(abs(ux)+abs(uz)), fy*(abs(uy)+abs(uz)))
    nominal = h*numerator/(depth_m**2-(h*uz/2.)**2)
    lower = max(0., hmin*max(0., numerator-error)/far**2-pixel_margin)
    upper = hmax*(numerator+error)/(near**2-(hmax*uz/2.)**2)+pixel_margin
    if abs(uz) < 1.e-8:
        # With an upright camera the exact side scale is f/Z. Avoid losing
        # that correlation to independent numerator/denominator bounds.
        vertical_focal = math.hypot(fx*ux, fy*uy)
        lower = max(0., hmin*vertical_focal/far-pixel_margin)
        upper = hmax*vertical_focal/near+pixel_margin
    # Bound every horizontal yaw using the maximum projective magnification.
    max_ray = (math.hypot(x, y)+radius+.5*hmax)/near
    ray_scale = math.sqrt(1.+max_ray**2)
    max_width = max(fx, fy)*math.hypot(w+tol, d+tol)*ray_scale/near+pixel_margin
    return ProjectedHeadSize(depth_m, depth_uncertainty_m, nominal, fx*w/depth_m,
                             lower, upper, max_width)


def metric_head_search(*, model_profile, depth_m, fx, fy, cx, cy, image_shape,
                       center=None, depth_uncertainty_m=.02,
                       camera_vertical=(0., 1., 0.), max_center_offset_ratio=1.5,
                       edge_region=None):
    from scripts.aufgabe04.perception.stand_axis.candidate_head_search import CandidateHeadSearch

    location_known = center is not None
    center = (cx, cy) if center is None else center
    size = projected_head_size(model_profile=model_profile, depth_m=depth_m,
        fx=fx, fy=fy, depth_uncertainty_m=depth_uncertainty_m,
        center_normalized=((center[0]-cx)/fx, (center[1]-cy)/fy),
        camera_vertical=camera_vertical)
    return CandidateHeadSearch(center, size.height_px, max_center_offset_ratio,
        edge_region=edge_region, pixel_size=size,
        center_tolerance_px=(None if location_known else math.hypot(*image_shape[:2])))
