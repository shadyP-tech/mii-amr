"""Stopped, current-image stand inspection without persisted proof protocols.

All authority comes from the immutable synchronized bundle, measured startup
calibration, current raw head borders and a unique current LiDAR association.
Vision never publishes motion. Seven independent completed empty decodes can
bound an unidentified head; an absent or skipped decoder cannot.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from collections import deque, OrderedDict
from types import SimpleNamespace
import math
import threading
import time
from time import monotonic
import numpy as np
from .model import Candidate, InspectionResult, Pose2D, StoppedState, wrap_angle
from .config import RigidTransform
from .lidar import Scan, TargetMatch, match_target
from . import vision as v


@dataclass(frozen=True)
class FrameBundle:
    image: object
    stamp_sec: float
    received_monotonic: float
    scan: Scan
    map_from_camera: RigidTransform
    pose_map: Pose2D
    pose_odom: Pose2D
    epoch: int
    now_ros_sec: float
    selected_monotonic: float

    def __post_init__(self):
        if (not isinstance(self.image,np.ndarray) or self.image.dtype!=np.uint8
                or self.image.ndim!=3 or self.image.shape[2]!=3
                or self.image.flags.writeable
                or not isinstance(self.scan,Scan) or self.scan.epoch!=self.epoch
                or not isinstance(self.map_from_camera,RigidTransform)
                or not all(math.isfinite(x) for x in (self.stamp_sec,self.received_monotonic,self.now_ros_sec,self.selected_monotonic))
                or self.stamp_sec<=0 or self.received_monotonic<0):
            raise ValueError("camera bundle requires immutable current pixels and exact transforms")


@dataclass(frozen=True)
class HeadFit:
    corners: tuple
    poses: tuple
    yaw_hypotheses: tuple
    interval: tuple[float,float] | None
    expected_center: tuple[float,float]
    expected_height: float
    depth_m: float
    alternative_corners: tuple = ()


def _inverse(point, transform):
    x,y,z,w=transform.rotation_xyzw
    return v.rotate_vector(tuple(a-b for a,b in zip(point,transform.translation_xyz_m)),(-x,-y,-z,w))


def _rectified_point(point_map,bundle,rectification):
    return rectification @ np.asarray(_inverse(point_map,bundle.map_from_camera))


def _project(point,camera):
    if point[2]<=.01:raise ValueError("target is behind the camera")
    return camera.fx_px*point[0]/point[2]+camera.cx_px,camera.fy_px*point[1]/point[2]+camera.cy_px


def _roi_boundary(cv2, frame, *, model, camera, search, source_support, deadline, budget):
    """Retain the bounded opposite-view endpoint locator on current raw pixels.

    Its cropped LSD endpoints only seed raw rail fitting. All candidate fits
    must agree pairwise, and every accepted border is checked for a physical
    outer frame before it supplies geometry or identity.
    """
    height=camera.fy_px*model.head_height_m/search.pixel_size.depth_m
    u,y=search.center
    x0,y0=max(0,math.floor(u-1.2*height)),max(0,math.floor(y-1.2*height))
    x1,y1=min(frame.shape[1],math.ceil(u+1.2*height)+1),min(frame.shape[0],math.ceil(y+1.2*height)+1)
    pixels=frame[y0:y1,x0:x1]
    if not pixels.size:return None
    edges=v._canny_edges_from_frame(cv2,pixels,edge_preprocess="channel_union",blur_kernel=5,canny_low=20,canny_high=60)
    v.check_head_acquisition_deadline(deadline,"local_head_lines")
    lines=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY))[0]
    center=(u-x0,y-y0)
    rough=v._rough_proposals(v._line_groups(lines,expected_height=height,expected_center=center),
        expected_height=height,expected_center=center,max_center_offset=.75,deadline_monotonic_sec=deadline)
    if len(rough)>budget:return None
    accepted=[];families=v.CurrentBorderFamilies(edges,pixels)
    def shifted(corners):
        return None if corners is None else tuple(v.ImagePoint(p.u_px+x0,p.v_px+y0) for p in corners)
    for _,corners in rough:
        v.check_head_acquisition_deadline(deadline,"local_head_raw_fit")
        measured=v.refine_projected_head_border(cv2,edges,corners,corridor_half_width_px=4.)
        measured,outer=v.select_current_outer_head_border(cv2,edges,model_profile=model,
            refinement=measured,corridor_half_width_px=4.,neutral_proposal_corners=corners,
            deadline_monotonic_sec=deadline)
        full=shifted(measured.corners)
        if (measured.accepted and outer.accepted and source_support.accepts(full)
                and search.accepts_measurement(full)):
            accepted.append((measured,outer,corners))
    v.check_head_acquisition_deadline(deadline,"local_head_comparison")
    # Independent cropped fits may choose opposite sides of the same thick
    # physical rail. Keep all such pixel hypotheses (maximum four pixels),
    # never collapse them into a falsely precise angle or enlarge the QR mask.
    if not accepted or any(max(v._distance(p,q) for p,q in zip(a[0].corners,b[0].corners))>4.
            for i,a in enumerate(accepted) for b in accepted[i+1:]):return None
    alternatives=tuple(shifted(row[0].corners) for row in accepted)
    measured,outer,seed=max(accepted,key=lambda row:v._polygon_area(row[0].corners))
    measured=replace(measured,corners=shifted(measured.corners),candidate_corners=shifted(measured.candidate_corners))
    outer=replace(outer,original_corners=shifted(outer.original_corners),
        recovered_corners=shifted(outer.recovered_corners),neutral_proposal_corners=shifted(outer.neutral_proposal_corners),
        current_raw_alternatives=tuple(shifted(q) for q in outer.current_raw_alternatives))
    return v.CurrentHeadRefinement(measured,outer,v.HeadBorderSeed(shifted(seed),4.,"current_endpoint_locator"),alternatives)


def fit_head(cv2, frame, *, model, camera, search, source_support, deadline,
             proposal_filter=None):
    """One private call: raw acquisition → complete border → all plausible IPPE poses."""
    edges=v._canny_edges_from_frame(cv2,frame,edge_preprocess="channel_union",blur_kernel=5,canny_low=20,canny_high=60)
    colors=v.color_edge_support(cv2,np,frame)
    boundary={};budget={"used":0};screen=source_support.filter(proposal_filter)
    acquisition=v.acquire_material_rim(cv2,frame,raw_edges=edges,candidate_search=search,
        model_profile=model,proposal_filter=screen,refinement_out=boundary,budget_out=budget,
        deadline_monotonic_sec=deadline)
    if acquisition is None:
        acquisition=v.acquire_cold_head_proposal(cv2,frame,raw_edges=edges,model_profile=model,
            proposal_filter=screen,candidate_search=search,color_support_mask=colors,
            refinement_out=boundary,_verification_limit=max(0,v.MAX_RAW_VERIFICATIONS-budget['used']),
            deadline_monotonic_sec=deadline)
    v.check_head_acquisition_deadline(deadline,"head_pose")
    current=boundary.get('selected')
    if current is None and not any(word in acquisition.reason for word in ("ambiguous","budget","deadline")):
        current=_roi_boundary(cv2,frame,model=model,camera=camera,search=search,
            source_support=source_support,deadline=deadline,
            budget=max(0,v.MAX_RAW_VERIFICATIONS-budget['used']-acquisition.raw_verifications))
    if current is None:return None
    measured,outer=current.refinement,current.outer_recovery
    corners=measured.corners
    if (not measured.accepted or not outer.accepted or corners is None
            or measured.support is None or not measured.support.accepted
            or measured.corner_arm_support is None or not measured.corner_arm_support.accepted
            or not v.validated_current_head_boundary(outer,corners=corners)
            or (acquisition.proposal is not None and not v.same_current_border_family(
                corners,acquisition.proposal.corners,raw_edges=edges,frame_bgr=frame))
            or not source_support.accepts(corners) or not search.accepts_measurement(corners)
            or not v._complete_border(corners,frame.shape)
            or min(v._distance(a,b) for a,b in zip(corners,corners[1:]+corners[:1]))<24.):
        return None
    all_poses=[];plausible=[];bounded=True
    alternatives=current.alternative_corners or (corners,)
    for quad in alternatives:
        poses=v.estimate_planar_pose_ippe(cv2,quad,model.head_corners,camera,max_reprojection_rmse_px=2.)
        if not poses.accepted:
            bounded=False
            continue
        all_poses.extend(poses.hypotheses)
        best=poses.best.reprojection_rmse_px
        for pose in poses.hypotheses:
            if not pose.positive_depth or pose.reprojection_rmse_px>best+max(.75,best):continue
            sigma=max(.75,pose.reprojection_rmse_px)
            try:std,condition,depths=v._local_yaw_uncertainty(cv2,profile=model,camera=camera,pose=pose,sigma=sigma)
            except (ArithmeticError,ValueError,TypeError):std,condition,depths=None,None,False
            if (not depths or std is None or condition is None or not math.isfinite(std)
                    or std<0 or not 1.<=condition<=1.e6):
                bounded=False
                continue
            plausible.append(v.HeadOrientationHypothesis(math.radians(pose.yaw_deg),math.radians(std),
                pose.reprojection_rmse_px,sigma,condition,depths,pose.rotation_vector,
                pose.translation_xyz_m,pose.face_normal_xyz))
    interval=v.enclosing_axial_interval(plausible) if bounded else None
    v.check_head_acquisition_deadline(deadline,"head_complete")
    return HeadFit(corners,tuple(all_poses),tuple(plausible) if bounded else (),interval,
        search.center,search.height,search.pixel_size.depth_m,alternatives)



class InspectionSession:
    """One stopped target epoch. No previous frame can become new evidence."""
    def __init__(self,target,stopped,config,*,neighbors=(),retained_axis=None,arrival_anchor=None):
        import cv2
        if not isinstance(target,Candidate) or not isinstance(stopped,StoppedState):
            raise ValueError("inspection requires a typed target and stopped epoch")
        self.target,self.stopped,self.config=target,stopped,config
        self.neighbors=tuple(neighbors);self.retained_axis=retained_axis;self.arrival_anchor=arrival_anchor
        self.cv2=cv2;self.last_stamp=-math.inf;self.last_scan_stamp=-math.inf
        self.samples=deque(maxlen=7);self.matches=deque(maxlen=3);self.poisoned=False
        self.last_reason="awaiting_frame"
        c=config.calibration;m=config.model
        self.model=v.HeadModel(m.head_width_m,m.head_height_m,m.head_depth_m,m.qr_panel_width_m,
                              m.qr_panel_height_m,m.tolerance_m,m.head_top_height_m)
        self.camera=v.RectifiedCameraMatrix(c.projection_matrix[0],c.projection_matrix[5],
                                           c.projection_matrix[2],c.projection_matrix[6])
        self.rectification=np.asarray(c.rectification_matrix,float).reshape(3,3)
        self.maps=cv2.initUndistortRectifyMap(np.asarray(c.camera_matrix,float).reshape(3,3),
            np.asarray(c.distortion_coefficients,float),self.rectification,
            np.asarray(c.projection_matrix,float).reshape(3,4)[:,:3],
            (c.width_px,c.height_px),cv2.CV_32FC1)
        x,y=self.maps
        valid=((x>=0)&(y>=0)&(x<c.width_px-1)&(y<c.height_px-1)).astype(np.uint8)
        self.source_support=v.ImageSourceSupport(cv2,valid)
        self.resources=QrDecoderResources(cv2)
        if retained_axis is not None and (retained_axis.target_id!=target.uid
                or retained_axis.status!="unidentified" or retained_axis.accepted_frames<7
                or retained_axis.axis_normal_rad is None or retained_axis.axis_half_width_rad>math.radians(15.)):
            raise ValueError("opposite inspection requires the same target's bounded first view")

    def _reject(self,reason,*,poison=False):
        self.last_reason=reason;self.samples.clear()
        if poison:
            self.poisoned=True
            raise RuntimeError("inspection evidence invalid: "+reason)
        return None

    def _current(self,bundle,started):
        now=bundle.now_ros_sec+max(0.,time.monotonic()-bundle.selected_monotonic)
        return (0<=time.monotonic()-bundle.received_monotonic<=.5
            and 0<=time.monotonic()-bundle.scan.received_monotonic<=.5
            and bundle.epoch==self.stopped.epoch and bundle.scan.epoch==self.stopped.epoch
            and 0<=now-bundle.stamp_sec<=.5 and 0<=now-bundle.scan.stamp_sec<=.5
            and abs(bundle.stamp_sec-bundle.scan.stamp_sec)<=.10
            and math.hypot(bundle.pose_map.x_m-self.stopped.pose_map.x_m,
                           bundle.pose_map.y_m-self.stopped.pose_map.y_m)<=.03
            and abs(wrap_angle(bundle.pose_map.yaw_rad-self.stopped.pose_map.yaw_rad))<=math.radians(2.)
            and math.hypot(bundle.pose_odom.x_m-self.stopped.pose_odom.x_m,
                           bundle.pose_odom.y_m-self.stopped.pose_odom.y_m)<=.03
            and abs(wrap_angle(bundle.pose_odom.yaw_rad-self.stopped.pose_odom.yaw_rad))<=math.radians(2.))

    def _anchor_target(self):
        if self.arrival_anchor is None:return self.target
        source,target=self.arrival_anchor
        if (not isinstance(source,StoppedState) or not isinstance(target,Candidate)
                or target.uid!=self.target.uid or source.map_id!=self.stopped.map_id
                or not 0<=self.stopped.captured_monotonic-source.captured_monotonic<=self.config.mission_timeout_sec
                or source.epoch>self.stopped.epoch):
            return None
        local=source.map_from_odom.inverse_apply(Pose2D(target.x_m,target.y_m))
        current=self.stopped.map_from_odom.apply(local)
        if math.hypot(current.x_m-self.target.x_m,current.y_m-self.target.y_m)>.35:return None
        return self.target

    def _crop(self,frame,fit,bundle,target):
        corners=fit.corners;c=self.config.calibration;m=self.config.model
        box=(math.floor(min(p.u_px for p in corners)),math.floor(min(p.v_px for p in corners)),
             math.ceil(max(p.u_px for p in corners))+1,math.ceil(max(p.v_px for p in corners))+1)
        if not 0<=box[0]<box[2]<=c.width_px or not 0<=box[1]<box[3]<=c.height_px:return None
        target_far=fit.depth_m+self.config.survey_candidate_radius_m+target.uncertainty_m
        for other in self.neighbors:
            if other.uid==target.uid:continue
            point=_rectified_point((other.x_m,other.y_m,m.head_center_height_m),bundle,self.rectification)
            if point[2]<=0:continue
            u,y=_project(point,self.camera)
            margin=max(4.,self.camera.fx_px*other.uncertainty_m/point[2])
            hx=self.camera.fx_px*m.head_width_m/(2*point[2])+margin
            hy=self.camera.fy_px*m.head_height_m/(2*point[2])+margin
            neighbor=(u-hx,y-hy,u+hx,y+hy)
            overlap=box[0]<neighbor[2] and neighbor[0]<box[2] and box[1]<neighbor[3] and neighbor[1]<box[3]
            if overlap and point[2]-other.uncertainty_m-m.head_width_m/2<=target_far:return None
        pixels=frame[box[1]:box[3],box[0]:box[2]].copy()
        mask=np.full(pixels.shape[:2],255,np.uint8)
        for quad in fit.alternative_corners or (corners,):
            points=np.asarray([(p.u_px-box[0],p.v_px-box[1]) for p in quad],np.float32)
            current=np.zeros(pixels.shape[:2],np.uint8)
            self.cv2.fillConvexPoly(current,np.rint(points).astype(np.int32),255)
            mask=self.cv2.bitwise_and(mask,current)
        mask=self.cv2.erode(mask,np.ones((3,3),np.uint8));pixels[mask==0]=255
        return pixels

    def _map_interval(self,fit,bundle):
        if fit.interval is None:return None
        hypotheses=[]
        for h in fit.yaw_hypotheses:
            optical=self.rectification.T@np.asarray(h.face_normal_xyz)
            normal=v.rotate_vector(tuple(optical),bundle.map_from_camera.rotation_xyzw)
            if math.hypot(normal[0],normal[1])<.5:return None
            hypotheses.append(SimpleNamespace(yaw_rad=math.atan2(normal[1],normal[0]),yaw_std_rad=h.yaw_std_rad))
        return v.enclosing_axial_interval(hypotheses)

    def observe(self,bundle):
        started=time.monotonic()
        if self.poisoned:raise RuntimeError("inspection epoch was poisoned")
        if not self._current(bundle,started):return self._reject("stale_or_moving",poison=True)
        if bundle.stamp_sec<=self.last_stamp:return self._reject("repeated_image")
        self.last_stamp=bundle.stamp_sec
        target=self._anchor_target()
        if target is None:return self._reject("invalid_arrival_anchor",poison=True)
        match=match_target(bundle.scan,target,neighbors=self.neighbors)
        if match is None:return self._reject("current_scan_not_unique")
        if bundle.scan.stamp_sec>self.last_scan_stamp:
            self.matches.append((bundle.scan.stamp_sec,match));self.last_scan_stamp=bundle.scan.stamp_sec
        while self.matches and bundle.scan.stamp_sec-self.matches[0][0]>1.5:self.matches.popleft()
        if len(self.matches)<3:return self._reject("collecting_current_scans")
        if any(math.hypot(a.x_m-match.x_m,a.y_m-match.y_m)>.05 for _,a in self.matches):
            return self._reject("inconsistent_current_scans")
        calibration=self.config.calibration
        if bundle.image.shape[:2]!=(calibration.height_px,calibration.width_px):
            return self._reject("image_calibration_mismatch",poison=True)
        frame=self.cv2.remap(bundle.image,*self.maps,interpolation=self.cv2.INTER_LINEAR,borderMode=self.cv2.BORDER_CONSTANT)
        point=_rectified_point((match.x_m,match.y_m,self.config.model.head_center_height_m),bundle,self.rectification)
        try:
            center=_project(point,self.camera)
            vertical=self.rectification@np.asarray(v.rotate_vector((0.,0.,1.),
                tuple(-a if i<3 else a for i,a in enumerate(bundle.map_from_camera.rotation_xyzw))))
            search=v.metric_head_search(model_profile=self.model,depth_m=float(point[2]),
                fx=self.camera.fx_px,fy=self.camera.fy_px,cx=self.camera.cx_px,cy=self.camera.cy_px,
                image_shape=frame.shape,center=center,camera_vertical=tuple(vertical),
                depth_uncertainty_m=max(.02,target.uncertainty_m),position_uncertainty_m=max(.02,target.uncertainty_m))
            if search.height<18:return self._reject("head_too_small")
            remaining=.5-(bundle.now_ros_sec-min(bundle.stamp_sec,bundle.scan.stamp_sec)+started-bundle.selected_monotonic)
            deadline=started+max(0.,remaining-.05)
            fit=fit_head(self.cv2,frame,model=self.model,camera=self.camera,search=search,
                source_support=self.source_support,deadline=deadline)
        except (ValueError,v.HeadAcquisitionDeadlineExceeded):
            return self._reject("head_unobservable_or_deadline")
        if fit is None:return self._reject("head_unobservable")
        # The fitted diagonal center supplies a finite physical ray, while the
        # raw scan supplies its range. Reassociation still requires uniqueness.
        u,y=v.CandidateHeadSearch.projected_center(fit.corners)
        ray=self.rectification.T@np.asarray(((u-self.camera.cx_px)/self.camera.fx_px,
                                            (y-self.camera.cy_px)/self.camera.fy_px,1.))
        world=v.rotate_vector(tuple(ray),bundle.map_from_camera.rotation_xyzw)
        camera_origin=bundle.map_from_camera.translation_xyz_m
        horizontal=math.hypot(world[0],world[1])
        if horizontal<1e-9:return self._reject("invalid_head_ray")
        distance=math.hypot(match.x_m-camera_origin[0],match.y_m-camera_origin[1])
        hit=(camera_origin[0]+world[0]*distance/horizontal,camera_origin[1]+world[1]*distance/horizontal)
        in_scan=bundle.scan.map_from_scan.inverse_apply(Pose2D(*hit))
        registered=match_target(bundle.scan,target,camera_bearing=math.atan2(in_scan.y_m,in_scan.x_m),
                                cone_half_angle=math.radians(3.),neighbors=self.neighbors)
        if registered is None or not set(registered.source_indices).issubset(match.source_indices):
            return self._reject("head_ray_not_current_target")
        pixels=self._crop(frame,fit,bundle,target)
        if pixels is None:return self._reject("target_crop_overlap")
        if not self._current(bundle,started):return self._reject("expired_geometry")
        remaining=.5-(bundle.now_ros_sec-min(bundle.stamp_sec,bundle.scan.stamp_sec)+time.monotonic()-bundle.selected_monotonic)
        if remaining<=.05:return self._reject("decoder_skipped")
        details={}
        observations=detect_qr_observations_bgr(pixels,self.cv2,diagnostics=details,
            max_elapsed_sec=min(.12,remaining-.05),resources=self.resources,preferred_scale=4)
        if not self._current(bundle,started):return self._reject("expired_decode")
        if len(observations)>1:return self._reject("ambiguous_qr_payloads",poison=True)
        interval=self._map_interval(fit,bundle)
        if self.retained_axis is not None and interval is not None:
            normal=self.retained_axis.axis_normal_rad
            if self.arrival_anchor is not None:
                source,_=self.arrival_anchor
                normal+=self.stopped.map_from_odom.yaw_rad-source.map_from_odom.yaw_rad
            if abs((interval[0]-normal+math.pi/2)%math.pi-math.pi/2)>interval[1]+self.retained_axis.axis_half_width_rad+math.radians(3.):
                return self._reject("opposite_orientation_disagrees")
        def result(status,qr_id=None,count=1,bound=interval):
            self.last_reason=status
            return InspectionResult(target.uid,status,self.stopped.epoch,bundle.pose_map,bundle.stamp_sec,
                match.x_m,match.y_m,max(target.uncertainty_m,match.uncertainty_m),qr_id,
                None if bound is None else bound[0],None if bound is None else bound[1],count)
        if observations:return result("identified",observations[0].text)
        if details.get('attempted') is not True:return self._reject("decoder_incomplete")
        if interval is None or interval[1]>math.radians(15.):return self._reject("orientation_unbounded")
        normalized=tuple(((p.u_px-fit.expected_center[0])/fit.expected_height,
                          (p.v_px-fit.expected_center[1])/fit.expected_height) for p in fit.corners)
        self.samples.append((bundle.stamp_sec,interval,normalized))
        while self.samples and bundle.stamp_sec-self.samples[0][0]>5.:self.samples.popleft()
        if any(max(math.dist(a,b) for a,b in zip(first[2],second[2]))>.04
                for i,first in enumerate(self.samples) for second in tuple(self.samples)[i+1:]):
            return self._reject("current_corner_drift")
        combined=v.enclosing_axial_interval(tuple(SimpleNamespace(yaw_rad=sample[1][0],
            yaw_std_rad=sample[1][1]/3.) for sample in self.samples))
        if combined is None or combined[1]>math.radians(15.):return self._reject("inconsistent_orientation")
        if len(self.samples)==7:return result("unidentified",count=7,bound=combined)
        self.last_reason="collecting_empty_current_frames"
        return None


@dataclass(frozen=True)
class DecodedQrObservation:
    text: str
    corners: tuple[tuple[float, float], ...] | None
    detector: str
    scale: float = 1.0

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("decoded QR observation requires a nonblank identity")
        if not isinstance(self.detector, str) or not self.detector:
            raise ValueError("decoded QR observation requires detector provenance")
        if not math.isfinite(self.scale) or self.scale < 1:
            raise ValueError("decoded QR observation scale must be finite and >= 1")
        if self.corners is not None:
            raise ValueError("QR identity decoding does not provide geometry")


class QrDecoderRuntime:
    """New evidence per call, with optional single-owner reusable backends."""

    def __init__(self, cv2, diagnostics=None, *, resources=None, work_budget=None):
        self.cv2 = cv2
        self.resources = resources
        self.work_budget = work_budget
        if resources is not None:
            resources.check_owner(cv2)
        self.diagnostics = diagnostics
        self._decoders = {}
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics["attempted"] = False

    def decoder(self, name):
        if name not in self._decoders:
            factory_name = ("QRCodeDetector" if name == "native"
                            else "wechat_qrcode_WeChatQRCode")
            try:
                factory = getattr(self.cv2, factory_name, None)
                self._decoders[name] = (self.resources.decoder(name) if self.resources is not None
                                        else factory() if factory is not None else None)
            except Exception:
                self._decoders[name] = None
        return self._decoders[name]

    def allow_work(self, stage, frame):
        return self.work_budget is None or self.work_budget.allow(stage, image_pixels(frame))

    def run_work(self, stage, frame, operation):
        return (operation() if self.work_budget is None else
                self.work_budget.measure(stage, image_pixels(frame), operation))


    def completed(self):
        """A backend returned a normalized batch, including a genuine miss."""
        if self.diagnostics is not None:
            self.diagnostics["attempted"] = True


class QrWorkHistory:
    """Bounded recent peak costs, partitioned by operation and pixel count."""

    def __init__(self):
        self._costs = OrderedDict()
        self._generation = 0
        self._seen = {}

    def begin_frame(self):
        self._generation += 1
        for key in tuple(self._costs):
            if self._generation - self._seen[key] >= 8:
                del self._costs[key]
                del self._seen[key]

    def estimate(self, stage, pixels):
        costs = [(size, elapsed) for (name, size), elapsed in self._costs.items()
                 if name == stage]
        if not costs:
            return 0.  # First observation remains a cooperative, measured probe.
        size, elapsed = min(costs, key=lambda item: abs(math.log(pixels / item[0])))
        if pixels > 2 * size and self._generation % 8 == 0:
            # Extrapolation must not permanently exclude a never-measured
            # enlarged view that may be the only decodable one. Periodically
            # measure it under the same cooperative source-age protection.
            return 0.
        return .002 + 1.2 * elapsed * max(1., pixels / size)

    def record(self, stage, pixels, elapsed):
        key = stage, pixels
        # Slow calls remain conservative, but do not permanently set the cost
        # after a transient scheduling stall. No data-dependent result survives.
        self._costs[key] = max(elapsed, .95 * self._costs.get(key, 0.))
        self._seen[key] = self._generation
        self._costs.move_to_end(key)
        while len(self._costs) > 32:
            old, _ = self._costs.popitem(last=False)
            del self._seen[old]


def image_pixels(image):
    return max(1, int(image.shape[0]) * int(image.shape[1]))


class QrWorkBudget:
    """Admit atomic decoder work against this frame's cooperative deadline."""

    def __init__(self, *, deadline, history, clock):
        self.deadline, self.history, self.clock = deadline, history, clock
        history.begin_frame()

    def allow(self, stage, pixels):
        remaining = None if self.deadline is None else self.deadline - self.clock()
        estimate = self.history.estimate(stage, pixels)
        allowed = remaining is None or remaining > estimate
        return allowed

    def measure(self, stage, pixels, operation):
        started = self.clock()
        try:
            return operation()
        finally:
            elapsed = max(0., self.clock() - started)
            self.history.record(stage, pixels, elapsed)


class QrDecoderResources:
    """Single-owner backend objects and cost history; never frame evidence."""

    def __init__(self, cv2):
        self.cv2 = cv2
        self.history = QrWorkHistory()
        self._owner = threading.get_ident()
        self._decoders = {}

    def check_owner(self, cv2):
        if cv2 is not self.cv2 or threading.get_ident() != self._owner:
            raise RuntimeError("QR resources require their original backend and owner thread")

    def decoder(self, name):
        self.check_owner(self.cv2)
        if name not in {"native", "wechat"}:
            raise ValueError("unknown QR decoder")
        if name not in self._decoders:
            factory = getattr(self.cv2, "QRCodeDetector" if name == "native"
                              else "wechat_qrcode_WeChatQRCode", None)
            try:
                self._decoders[name] = factory() if factory is not None else None
            except Exception:
                # A transient initialization failure must not disable this
                # backend for every subsequent image in the observer.
                return None
        return self._decoders[name]


def detect_qr_observations_bgr(
    frame, cv2, *, diagnostics: dict | None = None,
    max_elapsed_sec: float | None = None,
    resources=None,
    preferred_scale: int | None = None,
) -> tuple[DecodedQrObservation, ...]:
    """Decode identities in an already bound current-head crop.

    The first nonempty batch preserves every payload, including duplicates.
    Spatial association comes from the caller's measured head and LiDAR proof.
    Budgets stop between backend calls and image sizes; callers still enforce
    source freshness after an atomic backend call. ``diagnostics['attempted']``
    distinguishes completed empty decoding from skipped or failed work.
    """
    if max_elapsed_sec is not None and (
        type(max_elapsed_sec) not in (int, float)
        or not math.isfinite(max_elapsed_sec) or max_elapsed_sec <= 0.0
    ):
        raise ValueError("QR processing budget must be finite and positive")
    if preferred_scale is not None and (type(preferred_scale) is not int or preferred_scale not in (2, 4)):
        raise ValueError("preferred QR scale must be 2 or 4")
    if resources is not None:
        resources.check_owner(cv2)
    started = monotonic()
    deadline = None if max_elapsed_sec is None else started + max_elapsed_sec
    work_budget = (None if deadline is None and resources is None else
        QrWorkBudget(deadline=deadline, history=QrWorkHistory() if resources is None
                     else resources.history, clock=monotonic))
    runtime = QrDecoderRuntime(cv2, diagnostics, resources=resources, work_budget=work_budget)

    def exhausted():
        return deadline is not None and monotonic() >= deadline

    candidates = iter(_qr_decode_candidates(frame, cv2,
        **({"preferred_scale": preferred_scale} if preferred_scale is not None else {}),
        **({"work_budget": work_budget} if resources is not None else {})))
    while not exhausted():
        # Check before advancing the generator: next() may resize the crop.
        try:
            candidate, scale = next(candidates)
        except StopIteration:
            break
        if exhausted():
            break
        for observations in _identity_observations(
            candidate, scale=scale, runtime=runtime, budget_exhausted=exhausted,
        ):
            if observations:
                return observations
            if exhausted():
                return ()
    return ()


def _identity_observations(candidate, *, scale, runtime, budget_exhausted):
    # Payload-capable calls only: never seek native QR outlines, inspect their
    # corners, or warp a quad. Stop at the first successful backend via the
    # caller's generator loop, preserving every payload returned by that call.
    for backend, stage, method in (
        ("wechat", "wechat", "detectAndDecode"),
        ("native", "opencv_multi", "detectAndDecodeMulti"),
        ("native", "opencv_single", "detectAndDecode"),
    ):
        if budget_exhausted is not None and budget_exhausted():
            return
        if not runtime.allow_work(stage, candidate):
            continue
        decoder = runtime.decoder(backend)
        if decoder is None:
            continue
        try:
            result = runtime.run_work(stage, candidate, lambda: getattr(decoder, method)(candidate))
            if stage == "opencv_multi":
                decoded = result[1] if len(result) > 1 and result[0] else ()
            else:
                decoded = result[0]
            texts = ((decoded,) if isinstance(decoded, str)
                     else tuple(decoded) if decoded is not None else ())
            observations = tuple(
                DecodedQrObservation(str(raw).strip(), None, stage, float(scale))
                for raw in texts if raw is not None and str(raw).strip()
            )
            runtime.completed()
            yield observations
        except Exception:
            pass


def _qr_decode_candidates(frame, cv2, *, work_budget=None, preferred_scale=None):
    try:
        height, width = frame.shape[:2]
    except (AttributeError, ValueError):
        yield frame, 1.0
        return

    def prepare(stage, pixels, operation):
        if work_budget is None:
            return operation()
        if not work_budget.allow(stage, pixels):
            return None
        return work_budget.measure(stage, pixels, operation)

    # A bounded scan crop already localized a small head. Try enlarged pixels
    # first. Spatial binding already comes from the measured current head.
    # Respect the existing measured cost forecast and a fixed allocation cap.
    preferred_pixels = height*width*(preferred_scale or 1)**2
    if (preferred_scale is not None and preferred_pixels <= 1_500_000
            and (work_budget is None or any(work_budget.allow(stage, preferred_pixels)
                 for stage in ("opencv_multi", "opencv_single", "wechat")))):
        enlarged = prepare("resize", preferred_pixels,
                           lambda: _resize_for_qr(cv2, frame, scale=preferred_scale))
        if enlarged is not None:
            yield enlarged, float(preferred_scale)
    yield frame, 1.0


def _resize_for_qr(cv2, frame, *, scale: int):
    if scale <= 1:
        return frame
    try:
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    except Exception:
        return None
