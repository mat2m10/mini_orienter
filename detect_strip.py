"""
detect_strip.py
---------------
Core De Bruijn strip detector.

Given a frame, finds the strip on the base rim, reads the binary sequence,
matches it against the known De Bruijn sequence to get absolute rotation,
then returns the 3D pose via solvePnP.

The strip geometry assumed:
  - Cells lie on a circle of radius BASE_RADIUS_MM at height 0 (flat approximation)
  - Two solid sync lines (top + bottom borders) frame the cells
  - Binary pattern: filled cell = 1 (dark), empty = 0 (light)
"""

import cv2
import numpy as np
from itertools import product


# ---------------------------------------------------------------------------
# De Bruijn sequence generation
# ---------------------------------------------------------------------------

def de_bruijn(n, k=2):
    """Generate a binary De Bruijn sequence of order n (length = k^n = 2^n)."""
    alphabet = list(range(k))
    a = [0] * k * n
    seq = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                seq.extend(a[1:p+1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    # Pad to exactly 2^n
    while len(seq) < 2**n:
        seq.append(0)
    return seq


def build_lookup(seq, window):
    """
    Build a dict: tuple(window_bits) -> start_index around the ring.
    Every window of length `window` in the (wrapped) sequence maps to
    exactly one position — that's the De Bruijn guarantee.
    """
    n = len(seq)
    lookup = {}
    ring = seq + seq  # wrap
    for i in range(n):
        key = tuple(ring[i:i+window])
        lookup[key] = i
    return lookup


# ---------------------------------------------------------------------------
# Strip detector
# ---------------------------------------------------------------------------

class DeBruijnDetector:
    """
    Detects the De Bruijn strip on the base rim and returns pose.

    Parameters
    ----------
    base_radius_mm  : float  — radius of the dot circle on the base (mm)
    strip_height_mm : float  — physical height of the strip (mm)
    n_window        : int    — De Bruijn window size (default 4)
    sequence_rotation: int   — rotation applied when the strip was generated
    camera_matrix   : np.ndarray (3x3) or None
    dist_coeffs     : np.ndarray or None
    debug           : bool   — draw intermediate visuals
    """

    def __init__(self,
                 base_radius_mm,
                 strip_height_mm,
                 n_window=4,
                 sequence_rotation=0,
                 camera_matrix=None,
                 dist_coeffs=None,
                 debug=False):

        self.R = base_radius_mm
        self.H = strip_height_mm
        self.n_window = n_window
        self.debug = debug
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4,1))

        # Build sequence and lookup
        raw = de_bruijn(n_window)
        n = len(raw)
        rot = sequence_rotation % n
        self.sequence = raw[rot:] + raw[:rot]
        self.n_cells = n
        self.lookup = build_lookup(self.sequence, n_window)

        # Precompute 3D cell centre positions on the circle (flat approximation)
        # Cell i centre is at angle: i * 2pi/n_cells  (starting from angle 0 = right)
        self.obj_points_3d = self._make_3d_points()

        print(f"[DeBruijn] {n_cells} cells, window={n_window}, radius={base_radius_mm}mm")
        print(f"[DeBruijn] Sequence: {''.join(map(str, self.sequence))}")

    def _make_3d_points(self):
        """3D positions of cell centres on the circle, in the base's local frame."""
        pts = []
        for i in range(self.n_cells):
            angle = 2 * np.pi * i / self.n_cells
            x = self.R * np.cos(angle)
            y = self.R * np.sin(angle)
            z = 0.0
            pts.append([x, y, z])
        return np.array(pts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect(self, frame):
        """
        Process one frame.

        Returns
        -------
        result : dict with keys:
            'found'       : bool
            'rvec'        : rotation vector (solvePnP) or None
            'tvec'        : translation vector or None
            'cell_angle'  : absolute angle of cell 0 in image (radians) or None
            'debug_frame' : annotated frame if debug=True
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        debug_frame = frame.copy() if self.debug else None

        # Step 1: Find the strip ellipse (inner + outer sync rings)
        ellipse = self._find_strip_ellipse(gray, debug_frame)
        if ellipse is None:
            return {'found': False, 'rvec': None, 'tvec': None,
                    'cell_angle': None, 'debug_frame': debug_frame}

        (cx, cy), (ma, mi), angle_deg = ellipse  # OpenCV ellipse format

        # Step 2: Sample cells along the ellipse midline
        bits, sample_pts_2d = self._sample_cells(gray, ellipse, debug_frame)
        if bits is None:
            return {'found': False, 'rvec': None, 'tvec': None,
                    'cell_angle': None, 'debug_frame': debug_frame}

        # Step 3: Decode position using De Bruijn lookup
        start_idx, confidence = self._decode(bits)
        if start_idx is None:
            return {'found': False, 'rvec': None, 'tvec': None,
                    'cell_angle': None, 'debug_frame': debug_frame}

        # Step 4: Build 2D <-> 3D correspondences for solvePnP
        # sample_pts_2d[i] corresponds to cell (start_idx + i) % n_cells
        image_points = []
        object_points = []
        n_visible = len(sample_pts_2d)

        for i, pt2d in enumerate(sample_pts_2d):
            cell_idx = (start_idx + i) % self.n_cells
            image_points.append(pt2d)
            object_points.append(self.obj_points_3d[cell_idx])

        image_points = np.array(image_points, dtype=np.float32)
        object_points = np.array(object_points, dtype=np.float32)

        # Step 5: solvePnP
        rvec, tvec = self._solve_pose(object_points, image_points)

        if self.debug and rvec is not None:
            self._draw_debug(debug_frame, ellipse, sample_pts_2d,
                             bits, start_idx, rvec, tvec)

        return {
            'found': rvec is not None,
            'rvec': rvec,
            'tvec': tvec,
            'ellipse': ellipse,
            'start_idx': start_idx,
            'confidence': confidence,
            'debug_frame': debug_frame
        }

    # ------------------------------------------------------------------
    # Step 1: Find the strip ellipse
    # ------------------------------------------------------------------

    def _find_strip_ellipse(self, gray, debug_frame=None):
        """
        Find the ellipse corresponding to the strip's sync border lines.
        Strategy:
          1. Threshold for dark regions
          2. Find contours
          3. Filter by elliptical shape and reasonable size
          4. Return the best ellipse
        """
        h, w = gray.shape

        # Adaptive threshold — robust to varying lighting
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=8
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        best_ellipse = None
        best_score = 0

        for cnt in contours:
            if len(cnt) < 5:
                continue

            area = cv2.contourArea(cnt)
            # Rough size filter — strip should be a medium-sized ring
            if area < 500 or area > (w * h * 0.5):
                continue

            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (ma, mi), ang = ellipse

            # Aspect ratio — shouldn't be too elongated (very tilted camera)
            if mi < 1 or ma / mi > 6:
                continue

            # Should be roughly centred in frame
            if cx < w * 0.1 or cx > w * 0.9:
                continue
            if cy < h * 0.1 or cy > h * 0.9:
                continue

            # Score by how well the contour fits the ellipse
            # (perimeter vs ellipse perimeter)
            score = area / (ma * mi + 1e-5)
            if score > best_score:
                best_score = score
                best_ellipse = ellipse

        if debug_frame is not None and best_ellipse is not None:
            cv2.ellipse(debug_frame, best_ellipse, (0,255,255), 2)

        return best_ellipse

    # ------------------------------------------------------------------
    # Step 2: Sample cells along the ellipse
    # ------------------------------------------------------------------

    def _sample_cells(self, gray, ellipse, debug_frame=None):
        """
        Walk around the ellipse midline, sample brightness at n_cells positions.
        Returns binary bit list and 2D sample point coordinates.
        """
        (cx, cy), (ma, mi), angle_deg = ellipse
        angle_rad = np.deg2rad(angle_deg)

        # Sample at more points than cells for robustness, then bin
        n_samples = self.n_cells * 8
        samples = []
        pts_2d = []

        for i in range(n_samples):
            theta = 2 * np.pi * i / n_samples
            # Parametric point on ellipse midline
            x = cx + (ma/2) * np.cos(theta) * np.cos(angle_rad) \
                   - (mi/2) * np.sin(theta) * np.sin(angle_rad)
            y = cy + (ma/2) * np.cos(theta) * np.sin(angle_rad) \
                   + (mi/2) * np.sin(theta) * np.cos(angle_rad)

            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < gray.shape[1] and 0 <= yi < gray.shape[0]:
                val = int(gray[yi, xi])
                samples.append((val, (x, y)))
            else:
                samples.append((255, (x, y)))  # out of frame = white

        # Bin into n_cells buckets, average brightness per cell
        cell_values = []
        cell_centres = []
        bucket = n_samples // self.n_cells

        for c in range(self.n_cells):
            bucket_samples = samples[c*bucket:(c+1)*bucket]
            vals = [s[0] for s in bucket_samples]
            pts  = [s[1] for s in bucket_samples]
            cell_values.append(np.mean(vals))
            mx = np.mean([p[0] for p in pts])
            my = np.mean([p[1] for p in pts])
            cell_centres.append((mx, my))

        # Threshold to binary using Otsu-like midpoint
        arr = np.array(cell_values)
        thresh = (arr.min() + arr.max()) / 2
        bits = [1 if v < thresh else 0 for v in arr]

        if debug_frame is not None:
            for i, (pt, bit) in enumerate(zip(cell_centres, bits)):
                col = (50,50,200) if bit else (200,200,50)
                cv2.circle(debug_frame, (int(pt[0]), int(pt[1])), 4, col, -1)

        return bits, cell_centres

    # ------------------------------------------------------------------
    # Step 3: Decode De Bruijn position
    # ------------------------------------------------------------------

    def _decode(self, bits):
        """
        Try every window of length n_window around the ring.
        Return (start_index, confidence) or (None, 0).
        """
        n = len(bits)
        ring = bits + bits
        best_idx = None
        best_conf = 0

        for start in range(n):
            window = tuple(ring[start:start+self.n_window])
            if window in self.lookup:
                # Verify: extend the match as far as possible
                db_start = self.lookup[window]
                matches = 0
                for k in range(n):
                    expected = self.sequence[(db_start + k) % n]
                    actual   = bits[(start + k) % n]
                    if expected == actual:
                        matches += 1
                conf = matches / n
                if conf > best_conf:
                    best_conf = conf
                    # start_idx = which cell in the sequence is at position 0 of our sample ring
                    best_idx = (n - start + db_start) % n

        if best_conf < 0.6:  # < 60% match = unreliable
            return None, best_conf

        return best_idx, best_conf

    # ------------------------------------------------------------------
    # Step 4: solvePnP
    # ------------------------------------------------------------------

    def _solve_pose(self, obj_pts, img_pts):
        if self.camera_matrix is None:
            # No calibration: return None — caller can still use 2D stabilisation
            return None, None

        if len(obj_pts) < 4:
            return None, None

        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return (rvec, tvec) if success else (None, None)

    # ------------------------------------------------------------------
    # Debug drawing
    # ------------------------------------------------------------------

    def _draw_debug(self, frame, ellipse, sample_pts, bits, start_idx, rvec, tvec):
        if rvec is not None and self.camera_matrix is not None:
            # Draw axes on the base
            axis_len = self.R
            axes = np.float32([[axis_len,0,0],[0,axis_len,0],[0,0,-axis_len],[0,0,0]])
            imgpts, _ = cv2.projectPoints(axes, rvec, tvec,
                                          self.camera_matrix, self.dist_coeffs)
            origin = tuple(imgpts[3].ravel().astype(int))
            cv2.line(frame, origin, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 2)  # X red
            cv2.line(frame, origin, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 2)  # Y green
            cv2.line(frame, origin, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 2)  # Z blue

        (cx, cy), _, _ = ellipse
        cv2.putText(frame, f"cell0={start_idx}", (int(cx)-40, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
