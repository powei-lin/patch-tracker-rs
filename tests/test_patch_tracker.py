"""
Tests for patch_tracker Python bindings.
Uses pytest and opencv-python-headless.
"""

import numpy as np
import pytest
import cv2
import patch_tracker

DATA_DIR = "tests/data"


def load_gray(filename: str) -> np.ndarray:
    img = cv2.imread(f"{DATA_DIR}/{filename}", cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Failed to load {filename}"
    return img


@pytest.fixture
def img0() -> np.ndarray:
    return load_gray("img0.png")


@pytest.fixture
def img1() -> np.ndarray:
    return load_gray("img1.png")


# ── PatchTracker ─────────────────────────────────────────────────────────────

class TestPatchTracker:
    def test_default_construction(self):
        t = patch_tracker.PatchTracker()
        assert isinstance(t, patch_tracker.PatchTracker)

    def test_custom_construction(self):
        t = patch_tracker.PatchTracker(levels=3, grid_size=32)
        assert isinstance(t, patch_tracker.PatchTracker)

    def test_process_first_frame_detects_points(self, img0):
        t = patch_tracker.PatchTracker()
        t.process_frame(img0)
        pts = t.get_track_points()
        assert len(pts) > 0, "Expected keypoints to be detected on first frame"

    def test_process_two_frames_tracks_points(self, img0, img1):
        t = patch_tracker.PatchTracker()
        t.process_frame(img0)
        t.process_frame(img1)
        pts = t.get_track_points()
        assert len(pts) > 0

    def test_get_track_points_returns_dict_of_float_tuples(self, img0):
        t = patch_tracker.PatchTracker()
        t.process_frame(img0)
        pts = t.get_track_points()
        assert isinstance(pts, dict)
        for k, v in pts.items():
            assert isinstance(k, int)
            assert isinstance(v, tuple) and len(v) == 2
            assert isinstance(v[0], float) and isinstance(v[1], float)

    def test_remove_id(self, img0):
        t = patch_tracker.PatchTracker()
        t.process_frame(img0)
        pts_before = t.get_track_points()
        ids = list(pts_before.keys())[:2]
        t.remove_id(ids)
        pts_after = t.get_track_points()
        for i in ids:
            assert i not in pts_after

    def test_add_points(self, img0):
        t = patch_tracker.PatchTracker()
        t.process_frame(img0)
        count_before = len(t.get_track_points())
        t.add_points([(100.0, 200.0), (300.0, 400.0)])
        assert len(t.get_track_points()) == count_before + 2

    def test_process_frame_accepts_hwc1_array(self, img0):
        hwc = img0[:, :, np.newaxis]  # (H, W, 1)
        t = patch_tracker.PatchTracker()
        t.process_frame(hwc)
        assert len(t.get_track_points()) > 0

    def test_process_frame_rejects_wrong_dtype(self, img0):
        bad = img0.astype(np.float32)
        t = patch_tracker.PatchTracker()
        with pytest.raises(Exception):
            t.process_frame(bad)

    def test_process_frame_rejects_wrong_channels(self, img0):
        bgr = np.stack([img0, img0, img0], axis=-1)  # (H, W, 3)
        t = patch_tracker.PatchTracker()
        with pytest.raises(Exception):
            t.process_frame(bgr)

    def test_repr(self):
        t = patch_tracker.PatchTracker()
        assert "PatchTracker" in repr(t)

    def test_same_frame_twice_ids_are_monotonic(self, img0):
        t = patch_tracker.PatchTracker()
        t.process_frame(img0)
        t.process_frame(img0)
        pts = t.get_track_points()
        ids = sorted(pts.keys())
        # IDs are monotonically increasing (gaps are fine — some points are
        # dropped during tracking; new ones are assigned higher ids)
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids)), "IDs must be unique"


# ── StereoPatchTracker ────────────────────────────────────────────────────────

class TestStereoPatchTracker:
    def test_default_construction(self):
        st = patch_tracker.StereoPatchTracker()
        assert isinstance(st, patch_tracker.StereoPatchTracker)

    def test_process_first_frame_detects_points(self, img0, img1):
        st = patch_tracker.StereoPatchTracker()
        st.process_frame(img0, img1)
        pts0, pts1 = st.get_track_points()
        assert len(pts0) > 0
        assert len(pts1) > 0

    def test_stereo_ids_match(self, img0, img1):
        st = patch_tracker.StereoPatchTracker()
        st.process_frame(img0, img1)
        pts0, pts1 = st.get_track_points()
        assert set(pts0.keys()) == set(pts1.keys()), "Stereo track ids must match"

    def test_process_two_frames(self, img0, img1):
        st = patch_tracker.StereoPatchTracker()
        st.process_frame(img0, img1)
        st.process_frame(img0, img1)
        pts0, pts1 = st.get_track_points()
        assert len(pts0) > 0
        assert len(pts1) > 0

    def test_get_track_points_returns_two_dicts(self, img0, img1):
        st = patch_tracker.StereoPatchTracker()
        st.process_frame(img0, img1)
        result = st.get_track_points()
        assert isinstance(result, tuple) and len(result) == 2
        pts0, pts1 = result
        assert isinstance(pts0, dict)
        assert isinstance(pts1, dict)

    def test_remove_id(self, img0, img1):
        st = patch_tracker.StereoPatchTracker()
        st.process_frame(img0, img1)
        pts0, _ = st.get_track_points()
        ids = list(pts0.keys())[:2]
        st.remove_id(ids)
        pts0_after, pts1_after = st.get_track_points()
        for i in ids:
            assert i not in pts0_after
            assert i not in pts1_after

    def test_repr(self):
        st = patch_tracker.StereoPatchTracker()
        assert "StereoPatchTracker" in repr(st)


# ── ImagePyramid & free functions ─────────────────────────────────────────────

class TestImagePyramid:
    def test_build_image_pyramid_length(self, img0):
        pyr = patch_tracker.build_image_pyramid(img0, 4)
        assert len(pyr) == 4

    def test_build_image_pyramid_level0_same_size(self, img0):
        pyr = patch_tracker.build_image_pyramid(img0, 4)
        level0 = pyr[0]
        assert level0.shape == img0.shape

    def test_build_image_pyramid_each_level_half_size(self, img0):
        pyr = patch_tracker.build_image_pyramid(img0, 4)
        for i in range(1, 4):
            prev = pyr[i - 1]
            curr = pyr[i]
            assert curr.shape[0] == prev.shape[0] // 2
            assert curr.shape[1] == prev.shape[1] // 2

    def test_build_image_pyramid_dtype(self, img0):
        pyr = patch_tracker.build_image_pyramid(img0, 3)
        for i in range(3):
            assert pyr[i].dtype == np.uint8

    def test_negative_index(self, img0):
        pyr = patch_tracker.build_image_pyramid(img0, 4)
        assert pyr[-1].shape == pyr[3].shape

    def test_out_of_range_index_raises(self, img0):
        pyr = patch_tracker.build_image_pyramid(img0, 4)
        with pytest.raises(Exception):
            _ = pyr[10]

    def test_repr(self, img0):
        pyr = patch_tracker.build_image_pyramid(img0, 3)
        r = repr(pyr)
        assert "ImagePyramid" in r

    def test_single_level_pyramid(self, img0):
        pyr = patch_tracker.build_image_pyramid(img0, 1)
        assert len(pyr) == 1
        assert pyr[0].shape == img0.shape


class TestTrackPoints:
    def test_track_points_returns_dict(self, img0, img1):
        pyr0 = patch_tracker.build_image_pyramid(img0, 4)
        pyr1 = patch_tracker.build_image_pyramid(img1, 4)
        # seed with a few known good corners
        points = {i: (float(100 + i * 50), float(100 + i * 50)) for i in range(5)}
        result = patch_tracker.track_points(pyr0, pyr1, points)
        assert isinstance(result, dict)

    def test_track_points_ids_subset_of_input(self, img0, img1):
        pyr0 = patch_tracker.build_image_pyramid(img0, 4)
        pyr1 = patch_tracker.build_image_pyramid(img1, 4)
        t = patch_tracker.PatchTracker()
        t.process_frame(img0)
        input_pts = t.get_track_points()
        result = patch_tracker.track_points(pyr0, pyr1, input_pts)
        assert set(result.keys()).issubset(set(input_pts.keys()))

    def test_track_points_values_are_float_tuples(self, img0, img1):
        pyr0 = patch_tracker.build_image_pyramid(img0, 4)
        pyr1 = patch_tracker.build_image_pyramid(img1, 4)
        t = patch_tracker.PatchTracker()
        t.process_frame(img0)
        result = patch_tracker.track_points(pyr0, pyr1, t.get_track_points())
        for k, v in result.items():
            assert isinstance(k, int)
            assert isinstance(v, tuple) and len(v) == 2
            assert isinstance(v[0], float) and isinstance(v[1], float)

    def test_track_points_empty_input(self, img0, img1):
        pyr0 = patch_tracker.build_image_pyramid(img0, 4)
        pyr1 = patch_tracker.build_image_pyramid(img1, 4)
        result = patch_tracker.track_points(pyr0, pyr1, {})
        assert result == {}
