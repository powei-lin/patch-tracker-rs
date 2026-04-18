use image::{GrayImage, ImageBuffer};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::{PatchTracker, StereoPatchTracker, tracker};

fn array2_to_gray_image(arr: PyReadonlyArray2<u8>) -> PyResult<GrayImage> {
    let shape = arr.shape();
    let h = shape[0] as u32;
    let w = shape[1] as u32;
    let data = arr
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Array must be contiguous: {e}")))?
        .to_vec();
    ImageBuffer::from_raw(w, h, data)
        .ok_or_else(|| PyValueError::new_err("Failed to create image from array"))
}

fn array3_to_gray_image(arr: PyReadonlyArray3<u8>) -> PyResult<GrayImage> {
    let shape = arr.shape();
    if shape[2] != 1 {
        return Err(PyValueError::new_err(
            "3D array must have shape (H, W, 1) for grayscale",
        ));
    }
    let h = shape[0] as u32;
    let w = shape[1] as u32;
    let data = arr
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Array must be contiguous: {e}")))?
        .to_vec();
    ImageBuffer::from_raw(w, h, data)
        .ok_or_else(|| PyValueError::new_err("Failed to create image from array"))
}

fn to_gray_image(image: &Bound<'_, PyAny>) -> PyResult<GrayImage> {
    if let Ok(arr) = image.extract::<PyReadonlyArray2<u8>>() {
        return array2_to_gray_image(arr);
    }
    if let Ok(arr) = image.extract::<PyReadonlyArray3<u8>>() {
        return array3_to_gray_image(arr);
    }
    Err(PyValueError::new_err(
        "image must be a numpy array with shape (H, W) or (H, W, 1) and dtype uint8",
    ))
}

/// Single-camera patch tracker.
#[pyclass(name = "PatchTracker")]
pub struct PyPatchTracker {
    inner: PatchTracker,
}

#[pymethods]
impl PyPatchTracker {
    /// Create a new PatchTracker.
    ///
    /// Args:
    ///     levels: Number of image pyramid levels (default 4).
    ///     grid_size: Grid cell size in pixels for keypoint distribution (default 20).
    #[new]
    #[pyo3(signature = (levels=4, grid_size=20))]
    fn new(levels: u32, grid_size: u32) -> Self {
        Self {
            inner: PatchTracker::new(levels, grid_size),
        }
    }

    /// Process a new frame. Tracks existing points and detects new ones.
    ///
    /// Args:
    ///     image: Grayscale image as numpy array, shape (H, W) or (H, W, 1), dtype uint8.
    fn process_frame(&mut self, image: &Bound<'_, PyAny>) -> PyResult<()> {
        let img = to_gray_image(image)?;
        self.inner.process_frame(&img);
        Ok(())
    }

    /// Return currently tracked points.
    ///
    /// Returns:
    ///     dict mapping track id (int) to (x, y) pixel coordinates (float, float).
    fn get_track_points(&self) -> HashMap<usize, (f32, f32)> {
        self.inner.get_track_points()
    }

    /// Remove tracked points by id.
    ///
    /// Args:
    ///     ids: List of track ids to remove.
    fn remove_id(&mut self, ids: Vec<usize>) {
        self.inner.remove_id(&ids);
    }

    /// Manually add points to the tracker.
    ///
    /// Args:
    ///     points: List of (x, y) pixel coordinates.
    fn add_points(&mut self, points: Vec<(f32, f32)>) {
        self.inner.add_points(points);
    }

    fn __repr__(&self) -> String {
        format!(
            "PatchTracker(tracked_points={})",
            self.inner.get_track_points().len()
        )
    }
}

/// Stereo-camera patch tracker.
#[pyclass(name = "StereoPatchTracker")]
pub struct PyStereoPatchTracker {
    inner: StereoPatchTracker,
}

#[pymethods]
impl PyStereoPatchTracker {
    /// Create a new StereoPatchTracker.
    ///
    /// Args:
    ///     levels: Number of image pyramid levels (default 4).
    ///     grid_size: Grid cell size in pixels for keypoint distribution (default 20).
    #[new]
    #[pyo3(signature = (levels=4, grid_size=20))]
    fn new(levels: u32, grid_size: u32) -> Self {
        Self {
            inner: StereoPatchTracker::new(levels, grid_size),
        }
    }

    /// Process a new stereo frame pair.
    ///
    /// Args:
    ///     image0: Left grayscale image, shape (H, W) or (H, W, 1), dtype uint8.
    ///     image1: Right grayscale image, shape (H, W) or (H, W, 1), dtype uint8.
    fn process_frame(
        &mut self,
        image0: &Bound<'_, PyAny>,
        image1: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let img0 = to_gray_image(image0)?;
        let img1 = to_gray_image(image1)?;
        self.inner.process_frame(&img0, &img1);
        Ok(())
    }

    /// Return currently tracked points for both cameras.
    ///
    /// Returns:
    ///     Tuple of two dicts, each mapping track id (int) to (x, y) coordinates.
    ///     The same id in both dicts refers to the same stereo-matched feature.
    fn get_track_points(&self) -> (HashMap<usize, (f32, f32)>, HashMap<usize, (f32, f32)>) {
        let [pts0, pts1] = self.inner.get_track_points();
        (pts0, pts1)
    }

    /// Remove tracked points by id.
    ///
    /// Args:
    ///     ids: List of track ids to remove.
    fn remove_id(&mut self, ids: Vec<usize>) {
        self.inner.remove_id(&ids);
    }

    fn __repr__(&self) -> String {
        let [pts0, _] = self.inner.get_track_points();
        format!("StereoPatchTracker(tracked_points={})", pts0.len())
    }
}

#[pymodule]
pub fn patch_tracker(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPatchTracker>()?;
    m.add_class::<PyStereoPatchTracker>()?;
    m.add_class::<PyImagePyramid>()?;
    m.add_function(wrap_pyfunction!(py_build_image_pyramid, m)?)?;
    m.add_function(wrap_pyfunction!(py_track_points, m)?)?;
    Ok(())
}

fn gray_image_to_array2<'py>(py: Python<'py>, img: &GrayImage) -> Bound<'py, PyArray2<u8>> {
    let (w, h) = img.dimensions();
    let data = img.as_raw().clone();
    let arr = Array2::from_shape_vec((h as usize, w as usize), data).unwrap();
    arr.into_pyarray(py)
}

fn affine2_from_xy(x: f32, y: f32) -> nalgebra::Affine2<f32> {
    let mut v = nalgebra::Affine2::<f32>::identity();
    v.matrix_mut_unchecked().m13 = x;
    v.matrix_mut_unchecked().m23 = y;
    v
}

/// An image pyramid: a list of grayscale images at successive half-resolution levels.
#[pyclass(name = "ImagePyramid")]
pub struct PyImagePyramid {
    inner: Vec<GrayImage>,
}

#[pymethods]
impl PyImagePyramid {
    /// Number of pyramid levels.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Get the image at a given pyramid level as a numpy array (H, W) uint8.
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let len = self.inner.len() as isize;
        let i = if index < 0 { len + index } else { index };
        if i < 0 || i >= len {
            return Err(PyValueError::new_err(format!(
                "index {index} out of range for pyramid with {len} levels"
            )));
        }
        Ok(gray_image_to_array2(py, &self.inner[i as usize]))
    }

    fn __repr__(&self) -> String {
        let dims: Vec<String> = self
            .inner
            .iter()
            .map(|img| {
                let (w, h) = img.dimensions();
                format!("{w}x{h}")
            })
            .collect();
        format!("ImagePyramid([{}])", dims.join(", "))
    }
}

/// Build an image pyramid from a grayscale numpy array.
///
/// Args:
///     image: Grayscale image, shape (H, W) or (H, W, 1), dtype uint8.
///     levels: Number of pyramid levels.
///
/// Returns:
///     ImagePyramid with `levels` images, each half the resolution of the previous.
#[pyfunction]
#[pyo3(name = "build_image_pyramid")]
fn py_build_image_pyramid(image: &Bound<'_, PyAny>, levels: u32) -> PyResult<PyImagePyramid> {
    let img = to_gray_image(image)?;
    Ok(PyImagePyramid {
        inner: tracker::build_image_pyramid(&img, levels),
    })
}

/// Track points from one image pyramid to another.
///
/// This is a lower-level function. For typical use, prefer ``PatchTracker``.
///
/// Args:
///     pyramid0: Source image pyramid (previous frame).
///     pyramid1: Target image pyramid (current frame).
///     points: Dict mapping track id to ``(x, y)`` coordinates in pyramid0.
///
/// Returns:
///     Dict mapping surviving track ids to new ``(x, y)`` coordinates in pyramid1.
#[pyfunction]
#[pyo3(name = "track_points")]
fn py_track_points(
    pyramid0: &PyImagePyramid,
    pyramid1: &PyImagePyramid,
    points: HashMap<usize, (f32, f32)>,
) -> HashMap<usize, (f32, f32)> {
    let transforms: HashMap<usize, nalgebra::Affine2<f32>> = points
        .iter()
        .map(|(&id, &(x, y))| (id, affine2_from_xy(x, y)))
        .collect();
    tracker::track_points(&pyramid0.inner, &pyramid1.inner, &transforms)
        .into_iter()
        .map(|(id, t)| (id, (t.matrix().m13, t.matrix().m23)))
        .collect()
}
