"""
Single-camera patch tracker example using opencv.

Usage:
    python examples/single_camera.py --folder <path/to/image/folder>

The folder should contain *.png or *.jpg images (sorted by filename).
Tracking results are visualised with coloured dots per track id.
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
    parser = argparse.ArgumentParser(description="Single-camera patch tracker example")
    parser.add_argument("--folder", required=True, help="Folder containing PNG/JPG images")
    parser.add_argument("--levels", type=int, default=4, help="Pyramid levels (default 4)")
    parser.add_argument("--grid-size", type=int, default=20, help="Grid cell size (default 20)")
    parser.add_argument("--no-display", action="store_true", help="Skip GUI window")
    args = parser.parse_args()

    folder = Path(args.folder)
    paths = sorted(glob.glob(str(folder / "*.png"))) or sorted(
        glob.glob(str(folder / "*.jpg"))
    )
    if not paths:
        print(f"No PNG/JPG images found in {folder}", file=sys.stderr)
        sys.exit(1)

    tracker = patch_tracker.PatchTracker(levels=args.levels, grid_size=args.grid_size)

    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Could not read {path}", file=sys.stderr)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tracker.process_frame(gray)
        pts = tracker.get_track_points()

        vis = frame.copy()
        for tid, (x, y) in pts.items():
            color = id_to_color(tid)
            cv2.circle(vis, (int(x), int(y)), 3, color, -1)

        print(f"{Path(path).name}: {len(pts)} tracks")

        if not args.no_display:
            cv2.imshow("patch tracker", vis)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break

    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
