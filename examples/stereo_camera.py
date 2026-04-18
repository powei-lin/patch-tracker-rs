"""
Stereo-camera patch tracker example using opencv.

Usage:
    python examples/stereo_camera.py --folder <path/to/euroc/sequence>

Expects EuRoC-style layout:
    <folder>/mav0/cam0/data/*.png
    <folder>/mav0/cam1/data/*.png
"""

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

import patch_tracker


def id_to_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color from track id."""
    rng = np.random.default_rng(track_id)
    r, g, b = rng.integers(50, 256, size=3)
    return int(b), int(g), int(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stereo-camera patch tracker example")
    parser.add_argument("--folder", required=True, help="EuRoC sequence root folder")
    parser.add_argument("--levels", type=int, default=4, help="Pyramid levels (default 4)")
    parser.add_argument("--grid-size", type=int, default=20, help="Grid cell size (default 20)")
    parser.add_argument("--no-display", action="store_true", help="Skip GUI window")
    args = parser.parse_args()

    folder = Path(args.folder)
    paths0 = sorted(glob.glob(str(folder / "mav0/cam0/data/*.png")))
    paths1 = sorted(glob.glob(str(folder / "mav0/cam1/data/*.png")))

    if not paths0 or not paths1:
        print(f"No images found under {folder}/mav0/cam{{0,1}}/data/", file=sys.stderr)
        sys.exit(1)

    if len(paths0) != len(paths1):
        n = min(len(paths0), len(paths1))
        print(f"Warning: cam0 has {len(paths0)} frames, cam1 has {len(paths1)}. Using {n}.")
        paths0, paths1 = paths0[:n], paths1[:n]

    tracker = patch_tracker.StereoPatchTracker(levels=args.levels, grid_size=args.grid_size)

    for p0, p1 in zip(paths0, paths1):
        frame0 = cv2.imread(p0)
        frame1 = cv2.imread(p1)
        if frame0 is None or frame1 is None:
            print(f"Could not read {p0} or {p1}", file=sys.stderr)
            continue

        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        tracker.process_frame(gray0, gray1)
        pts0, pts1 = tracker.get_track_points()

        vis0, vis1 = frame0.copy(), frame1.copy()
        for tid in pts0:
            color = id_to_color(tid)
            x0, y0 = pts0[tid]
            cv2.circle(vis0, (int(x0), int(y0)), 3, color, -1)
            if tid in pts1:
                x1, y1 = pts1[tid]
                cv2.circle(vis1, (int(x1), int(y1)), 3, color, -1)

        print(f"{Path(p0).name}: {len(pts0)} stereo tracks")

        if not args.no_display:
            vis = np.concatenate([vis0, vis1], axis=1)
            cv2.imshow("stereo patch tracker", vis)
            key = cv2.waitKey(30)
            if key == ord("q"):
                break

    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
