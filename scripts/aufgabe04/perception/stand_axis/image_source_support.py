"""Exclude remapping/canvas boundaries from current physical head evidence."""

import math


class ImageSourceSupport:
    """Immutable current image-domain mask, independent of colour or target.

    The complete four borders must lie inside real source pixels after allowing
    for Gaussian blur, Canny's gradient and nonmaximum-suppression footprint.
    Neither image pixels nor raw edges are altered or supplied by this filter.
    """

    def __init__(self, cv2, valid_pixels, *, blur_kernel=5):
        import numpy as np
        if (not isinstance(valid_pixels, np.ndarray) or valid_pixels.ndim != 2
                or not valid_pixels.size or valid_pixels.dtype not in (np.uint8, np.bool_)
                or type(blur_kernel) is not int or blur_kernel < 0):
            raise ValueError("source support requires a nonempty camera-domain mask")
        self.radius = (blur_kernel // 2 if blur_kernel > 1 else 0) + 2
        self._valid = cv2.erode((valid_pixels != 0).astype(np.uint8),
            np.ones((2*self.radius+1, 2*self.radius+1), np.uint8),
            borderType=cv2.BORDER_CONSTANT, borderValue=0)
        self._valid.flags.writeable = False
        self.shape = self._valid.shape

    def accepts(self, corners):
        import numpy as np
        if corners is None or len(corners) != 4:
            return False
        points = np.asarray([(p.u_px, p.v_px) for p in corners], dtype=float)
        height, width = self.shape
        if (not np.isfinite(points).all() or np.any(points < 0.)
                or np.any(points[:, 0] >= width - 1.) or np.any(points[:, 1] >= height - 1.)):
            return False
        return all(self.segment(a, b) for a, b in zip(points, np.roll(points, -1, axis=0)))

    def segment(self, a, b):
        import numpy as np
        a, b = np.asarray(a, float), np.asarray(b, float)
        height, width = self.shape
        if (not np.isfinite((a, b)).all() or np.any(a < 0.) or np.any(b < 0.)
                or max(a[0], b[0]) >= width-1. or max(a[1], b[1]) >= height-1.):
            return False
        count = int(math.ceil(float(np.max(np.abs(b-a))))) + 1
        pixels = np.rint(np.linspace(a, b, count)).astype(np.int32)
        return bool(self._valid[pixels[:, 1], pixels[:, 0]].all())

    def filter(self, other=None):
        domain = self
        class Filter:
            source_support = domain

            def preview(self, proposal):
                return domain.accepts(proposal.corners) and (
                    other is None or getattr(other, "preview", other)(proposal))

            def __call__(self, proposal):
                return domain.accepts(proposal.corners) and (other is None or other(proposal))
        return Filter()

    def diagnostics(self):
        return {"policy": "original_camera_pixels_only", "footprint_radius_px": self.radius,
                "image_shape": self.shape, "supplies_corners": False}
