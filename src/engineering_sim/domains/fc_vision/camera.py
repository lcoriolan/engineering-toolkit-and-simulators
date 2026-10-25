#!/usr/bin/env python3
# DESCRIPTION: The ONE geometry bridge between physical motion on the unified core and the legacy pixel-space
#              vision models (legacy vision reference, vision_tracker sim_edge_node.Scene). Those legacy
#              models take pixel truth (cx, cy, side_px / bbox); this module produces that truth from 3-D poses.
#              Conventions: world ENU (x E, y N, z up); body FLU (x fwd, y left, z up); image OpenCV
#              (origin top-left, x right, y down). Ideal pinhole, no distortion (ASSUMED).
import math

import numpy as np


def focal_px(width_px, hfov_deg):
    """Focal length [px] from image width and horizontal field of view."""
    return 0.5 * width_px / math.tan(math.radians(hfov_deg) / 2.0)


def quat_to_R(q):
    """(w,x,y,z) body->world rotation matrix."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class DownCamera:
    """Body-fixed nadir camera: optical axis = body -Z, image up = body +X (nose at top of frame)."""

    def __init__(self, width=640, height=480, hfov_deg=90.0):
        self.W, self.H, self.f = width, height, focal_px(width, hfov_deg)

    def project(self, p_world, cam_pos, q_body):
        d = quat_to_R(q_body).T @ (np.asarray(p_world, float) - np.asarray(cam_pos, float))   # body frame
        depth = -d[2]
        if depth <= 1e-6:
            return None
        cx = self.W / 2 + self.f * (-d[1]) / depth          # image right = body -Y (right side)
        cy = self.H / 2 + self.f * (-d[0]) / depth          # image down  = body -X (aft)
        return cx, cy, depth

    def tag_truth(self, tag_xyz, tag_side_m, cam_pos, q_body):
        """Pixel truth in the legacy VisionSim convention: (cx, cy, side_px, in_view)."""
        pr = self.project(tag_xyz, cam_pos, q_body)
        if pr is None:
            return None, None, 0.0, False
        cx, cy, depth = pr
        side = self.f * tag_side_m / depth
        in_view = (0 <= cx < self.W) and (0 <= cy < self.H)
        return cx, cy, side, in_view


class FixedCamera:
    """Ground camera at `pos` looking along world +Y (level), image up = world +Z.
    Used for the vision_tracker edge-node scene (objects crossing the field of view)."""

    def __init__(self, pos, width=320, height=240, hfov_deg=60.0):
        self.pos = np.asarray(pos, float)
        self.W, self.H, self.f = width, height, focal_px(width, hfov_deg)

    def project(self, p_world):
        d = np.asarray(p_world, float) - self.pos
        depth = d[1]
        if depth <= 1e-6:
            return None
        return self.W / 2 + self.f * d[0] / depth, self.H / 2 - self.f * d[2] / depth, depth

    def unproject(self, cx, cy, depth):
        """Inverse of project (for building legacy-equivalent kinematic tracks)."""
        return self.pos + np.array([(cx - self.W / 2) * depth / self.f, depth, -(cy - self.H / 2) * depth / self.f])
