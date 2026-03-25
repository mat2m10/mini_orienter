"""
diagnose.py
-----------
Dumps frame N from your video showing every detection step.
Saves several debug images so you can see exactly where it's failing.

Usage:
    python diagnose.py --video data/knight.mp4 --frame 0
    python diagnose.py --video data/knight.mp4 --frame 10 --all-frames
"""

import cv2
import numpy as np
import argparse
import os

def diagnose(video_path, frame_num, out_dir="debug_frames"):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Cannot read frame {frame_num}")
        return

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Save original
    cv2.imwrite(f"{out_dir}/00_original.jpg", frame)
    print(f"Saved original frame {frame_num} ({w}x{h})")

    # --- Step 1: Gaussian blur ---
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cv2.imwrite(f"{out_dir}/01_blur.jpg", blur)

    # --- Step 2: Adaptive threshold (current settings) ---
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=8
    )
    cv2.imwrite(f"{out_dir}/02_thresh_adaptive.jpg", thresh)
    print("Saved adaptive threshold")

    # --- Step 2b: Try several blockSizes to find what works ---
    for bs in [15, 31, 51, 71]:
        for c in [4, 8, 12]:
            t = cv2.adaptiveThreshold(blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, bs, c)
            cv2.imwrite(f"{out_dir}/02_thresh_bs{bs}_c{c}.jpg", t)

    # --- Step 2c: Global Otsu threshold ---
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(f"{out_dir}/02_thresh_otsu.jpg", otsu)
    print("Saved Otsu threshold")

    # --- Step 3: Morphological close ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{out_dir}/03_morph_close.jpg", closed)

    # --- Step 4: Find and draw ALL contours, coloured by area ---
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_vis = frame.copy()

    ellipse_vis = frame.copy()
    print(f"\nFound {len(contours)} contours total")

    # Sort by area descending
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    # Draw top 20 contours + their ellipses
    for i, cnt in enumerate(contours_sorted[:20]):
        area = cv2.contourArea(cnt)
        # colour by rank
        col = (
            int(255 * (1 - i/20)),
            int(255 * i/20),
            128
        )
        cv2.drawContours(contour_vis, [cnt], -1, col, 2)

        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (ma, mi), ang = ellipse
            ratio = ma / (mi + 1e-5)
            cv2.ellipse(ellipse_vis, ellipse, col, 2)
            cv2.putText(ellipse_vis,
                        f"#{i} a={area:.0f} r={ratio:.1f}",
                        (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            if i < 20:
                print(f"  #{i:2d}  area={area:8.0f}  "
                      f"ellipse=({cx:.0f},{cy:.0f}) ma={ma:.0f} mi={mi:.0f} "
                      f"ratio={ratio:.2f}  pts={len(cnt)}")

    cv2.imwrite(f"{out_dir}/04_contours_top20.jpg", contour_vis)
    cv2.imwrite(f"{out_dir}/05_ellipses_top20.jpg", ellipse_vis)
    print(f"\nSaved contour and ellipse visualisations")

    # --- Step 5: Highlight what passes the current filter ---
    filtered_vis = frame.copy()
    passed = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > w*h*0.5:
            continue
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (ma, mi), ang = ellipse
        if mi < 1 or ma/mi > 6:
            continue
        if cx < w*0.1 or cx > w*0.9:
            continue
        if cy < h*0.1 or cy > h*0.9:
            continue
        cv2.ellipse(filtered_vis, ellipse, (0,255,0), 3)
        cv2.putText(filtered_vis, f"PASS a={area:.0f}",
                    (int(cx)-40, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        passed += 1

    cv2.imwrite(f"{out_dir}/06_passed_filter.jpg", filtered_vis)
    print(f"{passed} contours passed the current filter (green = candidate strip ellipse)")

    # --- Step 6: Show brightness profile around image centre ellipse guess ---
    # Sample brightness along a manually guessed ellipse at image centre
    # This shows if the strip pattern is readable at all
    cx_g, cy_g = w//2, h//2
    # Guess: strip is roughly in lower 2/3 of frame, medium sized
    for r in [80, 120, 160, 200]:
        profile_vis = frame.copy()
        vals = []
        pts = []
        for i in range(128):
            theta = 2*np.pi*i/128
            x = int(cx_g + r*np.cos(theta))
            y = int(cy_g + r*np.sin(theta))
            if 0<=x<w and 0<=y<h:
                v = int(gray[y,x])
                vals.append(v)
                pts.append((x,y))
                col = (0, int(255*(1-v/255)), int(255*v/255))
                cv2.circle(profile_vis, (x,y), 3, col, -1)
        cv2.imwrite(f"{out_dir}/07_brightness_r{r}.jpg", profile_vis)

    print(f"\nSaved brightness profile scans at r=80,120,160,200 from centre")
    print(f"\n--- All debug images saved to ./{out_dir}/ ---")
    print("Look at 05_ellipses_top20.jpg first — find the strip ellipse by eye,")
    print("then check 06_passed_filter.jpg to see if the filter catches it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out",   default="debug_frames")
    args = parser.parse_args()
    diagnose(args.video, args.frame, args.out)