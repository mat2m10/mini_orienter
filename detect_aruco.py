"""
detect_aruco.py
---------------
Find ArUco markers in an image and show what was detected.

Usage:
    python detect_aruco.py --image photo.jpg
    python detect_aruco.py --image photo.jpg --dict 4X4_50
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import argparse

DICTS = {
    "4X4_50":     aruco.DICT_4X4_50,
    "4X4_100":    aruco.DICT_4X4_100,
    "5X5_50":     aruco.DICT_5X5_50,
    "6X6_50":     aruco.DICT_6X6_50,
}

def detect(image_path, dict_name="4X4_50"):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Cannot open: {image_path}")
        return

    dictionary = aruco.getPredefinedDictionary(DICTS[dict_name])
    params = aruco.DetectorParameters()

    # Looser params — helps with hand-drawn markers
    params.adaptiveThreshWinSizeMin  = 3
    params.adaptiveThreshWinSizeMax  = 53
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate    = 0.02
    params.polygonalApproxAccuracyRate = 0.08   # more forgiving of wobbly edges

    detector = aruco.ArucoDetector(dictionary, params)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    vis = frame.copy()

    if ids is not None:
        aruco.drawDetectedMarkers(vis, corners, ids)
        print(f"Found {len(ids)} marker(s): IDs = {ids.ravel().tolist()}")
        for i, (corner, mid) in enumerate(zip(corners, ids.ravel())):
            c = corner[0]
            cx, cy = c.mean(axis=0).astype(int)
            cv2.putText(vis, f"ID {mid}", (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        print("No markers found.")
        print(f"Rejected candidates: {len(rejected)}")
        # Draw rejected candidates in red so you can see what it almost detected
        aruco.drawDetectedMarkers(vis, rejected, borderColor=(0, 0, 255))
        print("(Red boxes = shapes that looked like markers but failed decoding)")

    # Save result
    out_path = image_path.rsplit(".", 1)[0] + "_detected.jpg"
    cv2.imwrite(out_path, vis)
    print(f"Saved: {out_path}")

    # Show
    cv2.imshow("ArUco detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--dict",  default="4X4_50", choices=DICTS.keys())
    args = parser.parse_args()
    detect(args.image, args.dict)