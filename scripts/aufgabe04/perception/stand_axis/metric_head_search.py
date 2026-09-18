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
                        camera_vertical=(0., 1., 0.), pixel_margin=4., position_uncertainty_m=None):
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
    if position_uncertainty_m is not None and (
            not math.isfinite(position_uncertainty_m) or position_uncertainty_m < 0.):
        raise ValueError("position uncertainty must be finite and nonnegative")
    if not .99 <= norm <= 1.01:
        raise ValueError("camera vertical must be a unit vector")
    ux, uy, uz = (v/norm for v in camera_vertical)
    x, y = (v*depth_m for v in center_normalized)
    h, w, d, tol = (model_profile.head_height_m, model_profile.head_width_m,
                    model_profile.head_depth_m, model_profile.tolerance_m)
    radius = math.hypot(w/2., d) + depth_uncertainty_m + tol
    xy_radius = math.hypot(w/2., d) + max(depth_uncertainty_m, position_uncertainty_m or 0.) + tol
    near, far = depth_m-radius, depth_m+radius
    hmin, hmax = h-tol, h+tol
    if hmin <= 0 or near <= .5*hmax*abs(uz) + .01:
        raise ValueError("possible head volume crosses the camera plane")
    # Exact projected length of a vertical segment at its nominal centre:
    # H ||(fx(ux Z-X uz), fy(uy Z-Y uz))|| / (Z²-(H uz/2)²).
    numerator = math.hypot(fx*(ux*depth_m-x*uz), fy*(uy*depth_m-y*uz))
    nominal = h*numerator/(depth_m**2-(h*uz/2.)**2)
    # Divide numerator and denominator by Z before bounding. Bounding their
    # shared Z independently makes a slightly tilted camera admit ~75--245 px
    # for a 137 px head. For |dX|, |dY|, |dZ| <= radius, the normalized-ray
    # error is <= radius * (1 + |X/Z|) / near. The triangle inequality then
    # bounds the projected vertical vector for every allowed side centre.
    xn, yn = center_normalized
    vertical_scale = math.hypot(fx*(ux-xn*uz), fy*(uy-yn*uz))
    scale_error = abs(uz)/near*math.hypot(fx*(xy_radius+radius*abs(xn)), fy*(xy_radius+radius*abs(yn)))
    lower = max(0., hmin*max(0., vertical_scale-scale_error)/far-pixel_margin)
    upper = hmax*(vertical_scale+scale_error)/(near*(1.-(hmax*uz/(2.*near))**2))+pixel_margin
    # Bound every horizontal yaw using the maximum projective magnification.
    max_ray = (math.hypot(x, y)+xy_radius+.5*hmax)/near
    ray_scale = math.sqrt(1.+max_ray**2)
    max_width = max(fx, fy)*math.hypot(w+tol, d+tol)*ray_scale/near+pixel_margin
    return ProjectedHeadSize(depth_m, depth_uncertainty_m, nominal, fx*w/depth_m,
                             lower, upper, max_width)


def metric_head_search(*, model_profile, depth_m, fx, fy, cx, cy, image_shape,
                       center=None, depth_uncertainty_m=.02,
                       camera_vertical=(0., 1., 0.), max_center_offset_ratio=1.5,
                       edge_region=None, position_uncertainty_m=None):
    """Build size bounds and, with declared position error, a head-volume prior.

    Position error bounds the camera X/Y centre independently of optical depth.
    Callers without that registration contract retain the broad legacy centre
    allowance; a depth measurement alone never establishes lateral accuracy.
    """
    from scripts.aufgabe04.perception.stand_axis.candidate_head_search import CandidateHeadSearch

    location_known = center is not None
    center = (cx, cy) if center is None else center
    size = projected_head_size(model_profile=model_profile, depth_m=depth_m,
        fx=fx, fy=fy, depth_uncertainty_m=depth_uncertainty_m,
        center_normalized=((center[0]-cx)/fx, (center[1]-cy)/fy),
        camera_vertical=camera_vertical, position_uncertainty_m=position_uncertainty_m)
    center_bounds = None
    if position_uncertainty_m is not None and (
            not math.isfinite(position_uncertainty_m) or position_uncertainty_m < 0.):
        raise ValueError("position uncertainty must be finite and nonnegative")
    if location_known and position_uncertainty_m is not None:
        # A circular horizontal envelope covers every unknown head yaw. Project
        # its conservative camera-axis box, including floor/registration error.
        # Unlike a scalar centre-radius-plus-full-height window, this uses the
        # known upright construction to bound both image axes before rail quotas.
        from itertools import product
        from scripts.aufgabe04.perception.stand_axis.lidar_head_edge_region import LidarHeadEdgeRegion
        norm = math.sqrt(sum(v*v for v in camera_vertical))
        vertical = tuple(v/norm for v in camera_vertical)
        radius = math.hypot(model_profile.head_width_m/2., model_profile.head_depth_m)
        radius += max(depth_uncertainty_m, position_uncertainty_m) + model_profile.tolerance_m
        half_height = model_profile.head_height_m/2. + position_uncertainty_m + model_profile.tolerance_m
        extents = tuple(radius*math.sqrt(max(0., 1.-v*v))+half_height*abs(v) for v in vertical)
        point = ((center[0]-cx)*depth_m/fx, (center[1]-cy)*depth_m/fy, depth_m)
        # Centre uncertainty is independent of the head's half-width/height.
        # Adding the object's extent to centre tolerance admits another object
        # immediately beside the selected stand. Preserve projective depth error.
        errors = (position_uncertainty_m+model_profile.tolerance_m,)*2 + (
            depth_uncertainty_m+model_profile.tolerance_m,)
        centres = tuple(product(*((p-e, p+e) for p,e in zip(point,errors))))
        center_us, center_vs = zip(*((fx*x/z+cx, fy*y/z+cy) for x, y, z in centres))
        center_bounds = (min(center_us)-4., min(center_vs)-4., max(center_us)+4., max(center_vs)+4.)
        vertices = tuple(product(*( (p-e, p+e) for p, e in zip(point, extents))))
        if min(p[2] for p in vertices) <= .01:
            raise ValueError("possible upright head volume crosses the camera plane")
        us, vs = zip(*((fx*x/z+cx, fy*y/z+cy) for x, y, z in vertices))
        rows, cols = image_shape[:2]
        bounds = (max(0, math.floor(min(us)-6.)), max(0, math.floor(min(vs)-6.)),
                  min(cols, math.ceil(max(us)+6.)+1), min(rows, math.ceil(max(vs)+6.)+1))
        if edge_region is not None:
            a, b, c, d = edge_region.bounds
            bounds = (max(a, bounds[0]), max(b, bounds[1]), min(c, bounds[2]), min(d, bounds[3]))
        edge_region = LidarHeadEdgeRegion((rows, cols), bounds)
    return CandidateHeadSearch(center, size.height_px, max_center_offset_ratio,
        edge_region=edge_region, pixel_size=size,
        center_tolerance_px=(None if location_known else math.hypot(*image_shape[:2])),
        center_bounds_px=center_bounds)
