"""
pose_smoothing.py
-----------------
Temporal smoothing of 6-DoF poses for stabilise_aruco.py.

- Rotation is smoothed with SLERP on unit quaternions (exponential moving
  average on the rotation manifold), translation with a plain EMA.
- During marker dropouts the last motion can be extrapolated for a few
  frames (constant angular + linear velocity), after which the pose is
  held, instead of always freezing the last warp.

Online (causal) filter: it only uses past frames, so it can run in the
same single pass as the stabiliser. The price is lag: with alpha < 1 the
smoothed pose trails fast real motion.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Quaternion helpers, (w, x, y, z) convention
# ---------------------------------------------------------------------------

def quat_from_matrix(R):
    """Unit quaternion (w, x, y, z) from a 3x3 rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s,
             (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s,
             (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
             0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def matrix_from_quat(q):
    """3x3 rotation matrix from a quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def quat_mul(a, b):
    """Hamilton product a * b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def slerp(q0, q1, t):
    """Spherical linear interpolation from q0 (t=0) to q1 (t=1), shortest path."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:          # q and -q are the same rotation: take the short way
        q1, dot = -q1, -dot
    if dot > 0.9995:       # nearly identical: lerp is accurate and stable
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / s


# ---------------------------------------------------------------------------
# Smoother
# ---------------------------------------------------------------------------

class PoseSmoother:
    """
    Parameters
    ----------
    alpha : float in (0, 1]
        Weight of each new measurement. 1.0 = no smoothing.
    max_extrapolate : int
        Number of consecutive missing frames over which the last
        velocity is extrapolated before the pose is held. 0 = hold at once.
    """

    def __init__(self, alpha=1.0, max_extrapolate=0):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.max_extrapolate = max(0, int(max_extrapolate))
        self.q = None          # smoothed rotation
        self.t = None          # smoothed translation (3,)
        self.dq = np.array([1.0, 0, 0, 0])   # per-frame rotation step
        self.dt = np.zeros(3)                # per-frame translation step
        self.missed = 0

    def _out(self):
        rvec, _ = cv2.Rodrigues(matrix_from_quat(self.q))
        return rvec, self.t.reshape(3, 1).copy()

    def update(self, rvec, tvec):
        """Feed a measured pose; returns the smoothed (rvec, tvec)."""
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        q_meas = quat_from_matrix(R)
        t_meas = np.asarray(tvec, dtype=np.float64).reshape(3)

        if self.q is None:
            self.q, self.t = q_meas, t_meas
            self.missed = 0
            return self._out()

        q_prev, t_prev = self.q, self.t
        q_new = slerp(q_prev, q_meas, self.alpha)
        t_new = t_prev + self.alpha * (t_meas - t_prev)

        # Velocity is only meaningful between consecutive tracked frames
        if self.missed == 0:
            dq = quat_mul(q_new, quat_conj(q_prev))
            self.dq = dq if dq[0] >= 0 else -dq
            self.dt = t_new - t_prev
        else:
            self.dq = np.array([1.0, 0, 0, 0])
            self.dt = np.zeros(3)

        self.q, self.t = q_new, t_new
        self.missed = 0
        return self._out()

    def predict(self):
        """
        Pose for a frame without a measurement, or None before the first
        measurement. Extrapolates for up to max_extrapolate frames, then holds.
        """
        if self.q is None:
            return None
        if self.missed < self.max_extrapolate:
            self.q = quat_mul(self.dq, self.q)
            self.q /= np.linalg.norm(self.q)
            self.t = self.t + self.dt
        self.missed += 1
        return self._out()
