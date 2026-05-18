use crate::corners_fast9::Corner;
use image::{GrayImage, imageops};
#[cfg(all(not(feature = "nalgebra033"), feature = "nalgebra034"))]
use nalgebra as na;
#[cfg(feature = "nalgebra033")]
use nalgebra_033 as na;

use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::AddAssign;

use crate::{
    image_utilities::{self, HalfSize},
    patch,
};

use log::info;

pub struct PatchTracker {
    last_keypoint_id: usize,
    tracked_points_map: HashMap<usize, na::Affine2<f32>>,
    previous_image_pyramid: Vec<GrayImage>,
    grid_size: u32,
    levels: u32,
    #[cfg(feature = "magic_point")]
    magic_point_detector: Option<crate::magic_point::MagicPointDetector>,
}
impl Default for PatchTracker {
    fn default() -> Self {
        Self::new(4, 20)
    }
}
impl PatchTracker {
    pub fn new(levels: u32, grid_size: u32) -> Self {
        Self {
            last_keypoint_id: 0,
            tracked_points_map: HashMap::new(),
            previous_image_pyramid: Vec::new(),
            grid_size,
            levels,
            #[cfg(feature = "magic_point")]
            magic_point_detector: None,
        }
    }

    /// Enable MagicPoint keypoint detection backed by an ONNX model loaded with
    /// ONNX Runtime (CoreML acceleration on Apple platforms).
    /// When active, the detection grid is fixed at [`crate::magic_point::CELL_SIZE`] (8).
    #[cfg(feature = "magic_point")]
    pub fn with_magic_point(mut self) -> Result<Self, ort::Error> {
        self.magic_point_detector = Some(crate::magic_point::MagicPointDetector::new()?);
        Ok(self)
    }

    /// Same as [`with_magic_point`] but also sets a minimum score threshold to
    /// filter low-confidence detections. `threshold` must be in `(0, 1]`.
    #[cfg(feature = "magic_point")]
    pub fn with_magic_point_threshold(
        mut self,
        threshold: f32,
    ) -> Result<Self, ort::Error> {
        self.magic_point_detector = Some(
            crate::magic_point::MagicPointDetector::new()?.with_threshold(threshold),
        );
        Ok(self)
    }

    fn run_detect(&mut self, image_pyramid: &[GrayImage]) -> Vec<Corner> {
        #[cfg(feature = "magic_point")]
        if let Some(ref mut detector) = self.magic_point_detector {
            let current_corners: Vec<Corner> = self
                .tracked_points_map
                .values()
                .map(|v| {
                    Corner::new(
                        v.matrix().m13.round() as u32,
                        v.matrix().m23.round() as u32,
                        0.0,
                    )
                })
                .collect();
            return detector.detect(&image_pyramid[0], &current_corners);
        }
        detect_keypoints(&self.tracked_points_map, image_pyramid, self.grid_size)
    }

    pub fn process_frame(&mut self, greyscale_image: &GrayImage) {
        // build current image pyramid
        let current_image_pyramid: Vec<GrayImage> =
            build_image_pyramid(greyscale_image, self.levels);

        if !self.previous_image_pyramid.is_empty() {
            info!("old points {}", self.tracked_points_map.len());
            // track prev points
            self.tracked_points_map = track_points(
                &self.previous_image_pyramid,
                &current_image_pyramid,
                &self.tracked_points_map,
            );
            info!("tracked old points {}", self.tracked_points_map.len());
        }
        // add new points
        let new_points = self.run_detect(&current_image_pyramid);
        for point in &new_points {
            let mut v = na::Affine2::<f32>::identity();

            v.matrix_mut_unchecked().m13 = point.x as f32;
            v.matrix_mut_unchecked().m23 = point.y as f32;
            self.tracked_points_map.insert(self.last_keypoint_id, v);
            self.last_keypoint_id += 1;
        }

        // update saved image pyramid
        self.previous_image_pyramid = current_image_pyramid;
    }
    pub fn get_track_points(&self) -> HashMap<usize, (f32, f32)> {
        self.tracked_points_map
            .iter()
            .map(|(k, v)| (*k, (v.matrix().m13, v.matrix().m23)))
            .collect()
    }
    pub fn remove_id(&mut self, ids: &[usize]) {
        for id in ids {
            self.tracked_points_map.remove(id);
        }
    }
    pub fn add_points(&mut self, points: Vec<(f32, f32)>) {
        for (x, y) in points {
            let mut v = na::Affine2::<f32>::identity();
            v.matrix_mut_unchecked().m13 = x;
            v.matrix_mut_unchecked().m23 = y;
            self.tracked_points_map.insert(self.last_keypoint_id, v);
            self.last_keypoint_id += 1;
        }
    }
}

pub struct StereoPatchTracker {
    last_keypoint_id: usize,
    tracked_points_map_cam0: HashMap<usize, na::Affine2<f32>>,
    previous_image_pyramid0: Vec<GrayImage>,
    tracked_points_map_cam1: HashMap<usize, na::Affine2<f32>>,
    previous_image_pyramid1: Vec<GrayImage>,
    grid_size: u32,
    levels: u32,
    #[cfg(feature = "magic_point")]
    magic_point_detector: Option<crate::magic_point::MagicPointDetector>,
}
impl Default for StereoPatchTracker {
    fn default() -> Self {
        Self::new(4, 20)
    }
}
impl StereoPatchTracker {
    pub fn new(levels: u32, grid_size: u32) -> Self {
        Self {
            last_keypoint_id: 0,
            tracked_points_map_cam0: HashMap::new(),
            previous_image_pyramid0: Vec::new(),
            tracked_points_map_cam1: HashMap::new(),
            previous_image_pyramid1: Vec::new(),
            grid_size,
            levels,
            #[cfg(feature = "magic_point")]
            magic_point_detector: None,
        }
    }

    /// Enable MagicPoint keypoint detection for both cameras.
    #[cfg(feature = "magic_point")]
    pub fn with_magic_point(mut self) -> Result<Self, ort::Error> {
        self.magic_point_detector = Some(crate::magic_point::MagicPointDetector::new()?);
        Ok(self)
    }

    /// Same as [`with_magic_point`] but also sets a minimum score threshold.
    #[cfg(feature = "magic_point")]
    pub fn with_magic_point_threshold(mut self, threshold: f32) -> Result<Self, ort::Error> {
        self.magic_point_detector =
            Some(crate::magic_point::MagicPointDetector::new()?.with_threshold(threshold));
        Ok(self)
    }

    fn run_detect_cam0(&mut self, image_pyramid: &[GrayImage]) -> Vec<Corner> {
        #[cfg(feature = "magic_point")]
        if let Some(ref mut detector) = self.magic_point_detector {
            let current_corners: Vec<Corner> = self
                .tracked_points_map_cam0
                .values()
                .map(|v| {
                    Corner::new(
                        v.matrix().m13.round() as u32,
                        v.matrix().m23.round() as u32,
                        0.0,
                    )
                })
                .collect();
            return detector.detect(&image_pyramid[0], &current_corners);
        }
        detect_keypoints(&self.tracked_points_map_cam0, image_pyramid, self.grid_size)
    }

    pub fn process_frame(&mut self, greyscale_image0: &GrayImage, greyscale_image1: &GrayImage) {
        // build current image pyramid
        let current_image_pyramid0: Vec<GrayImage> =
            build_image_pyramid(greyscale_image0, self.levels);
        let current_image_pyramid1: Vec<GrayImage> =
            build_image_pyramid(greyscale_image1, self.levels);

        // not initialized
        if !self.previous_image_pyramid0.is_empty() {
            info!("old points {}", self.tracked_points_map_cam0.len());
            // track prev points
            self.tracked_points_map_cam0 = track_points(
                &self.previous_image_pyramid0,
                &current_image_pyramid0,
                &self.tracked_points_map_cam0,
            );
            self.tracked_points_map_cam1 = track_points(
                &self.previous_image_pyramid1,
                &current_image_pyramid1,
                &self.tracked_points_map_cam1,
            );
            info!("tracked old points {}", self.tracked_points_map_cam0.len());
        }
        // add new points
        let new_points0 = self.run_detect_cam0(&current_image_pyramid0);
        let tmp_tracked_points0: HashMap<usize, _> = new_points0
            .iter()
            .enumerate()
            .map(|(i, point)| {
                let mut v = na::Affine2::<f32>::identity();
                v.matrix_mut_unchecked().m13 = point.x as f32;
                v.matrix_mut_unchecked().m23 = point.y as f32;
                (i, v)
            })
            .collect();

        let tmp_tracked_points1 = track_points(
            &current_image_pyramid0,
            &current_image_pyramid1,
            &tmp_tracked_points0,
        );

        for (key0, pt0) in tmp_tracked_points0 {
            if let Some(pt1) = tmp_tracked_points1.get(&key0) {
                self.tracked_points_map_cam0
                    .insert(self.last_keypoint_id, pt0);
                self.tracked_points_map_cam1
                    .insert(self.last_keypoint_id, *pt1);
                self.last_keypoint_id += 1;
            }
        }

        // update saved image pyramid
        self.previous_image_pyramid0 = current_image_pyramid0;
        self.previous_image_pyramid1 = current_image_pyramid1;
    }
    pub fn get_track_points(&self) -> [HashMap<usize, (f32, f32)>; 2] {
        let tracked_pts0 = self
            .tracked_points_map_cam0
            .iter()
            .map(|(k, v)| (*k, (v.matrix().m13, v.matrix().m23)))
            .collect();
        let tracked_pts1 = self
            .tracked_points_map_cam1
            .iter()
            .map(|(k, v)| (*k, (v.matrix().m13, v.matrix().m23)))
            .collect();
        [tracked_pts0, tracked_pts1]
    }
    pub fn remove_id(&mut self, ids: &[usize]) {
        for id in ids {
            self.tracked_points_map_cam0.remove(id);
            self.tracked_points_map_cam1.remove(id);
        }
    }
}

pub fn build_image_pyramid(greyscale_image: &GrayImage, levels: u32) -> Vec<GrayImage> {
    let mut out = Vec::with_capacity(levels as usize);
    const FILTER_TYPE: imageops::FilterType = imageops::FilterType::Triangle;
    out.push(greyscale_image.clone());
    (1..levels).for_each(|_| {
        let last_img = out.last().unwrap();
        let (w, h) = last_img.dimensions();
        if w % 2 == 0 && h % 2 == 0 {
            out.push(last_img.half_size());
        } else {
            let new_w = w / 2;
            let new_h = h / 2;
            out.push(imageops::resize(last_img, new_w, new_h, FILTER_TYPE))
        }
    });
    out
}

fn detect_keypoints(
    tracked_points_map: &HashMap<usize, na::Affine2<f32>>,
    image_pyramid: &[GrayImage],
    grid_size: u32,
) -> Vec<Corner> {
    let num_points_in_cell = 1;
    let current_corners: Vec<Corner> = tracked_points_map
        .values()
        .map(|v| {
            Corner::new(
                v.matrix().m13.round() as u32,
                v.matrix().m23.round() as u32,
                0.0,
            )
        })
        .collect();
    // let curr_img_luma8 = DynamicImage::ImageLuma16(grayscale_image.clone()).into_luma8();
    let detect_level = if image_pyramid.len() > 1 { 1 } else { 0 };
    let detect_image = &image_pyramid[detect_level];
    let detect_scale = 1 << detect_level;

    image_utilities::detect_key_points(
        &image_pyramid[0],
        detect_image,
        detect_scale,
        grid_size,
        &current_corners,
        num_points_in_cell,
    )
}
pub fn track_points(
    image_pyramid0: &[GrayImage],
    image_pyramid1: &[GrayImage],
    transform_maps0: &HashMap<usize, na::Affine2<f32>>,
) -> HashMap<usize, na::Affine2<f32>> {
    let transform_maps1: HashMap<usize, na::Affine2<f32>> = transform_maps0
        .par_iter()
        .filter_map(|(k, v)| {
            if let Some(new_v) = track_one_point(image_pyramid0, image_pyramid1, v) {
                // return Some((k.clone(), new_v));
                if let Some(old_v) = track_one_point(image_pyramid1, image_pyramid0, &new_v)
                    && (v.matrix() - old_v.matrix())
                        .fixed_view::<2, 1>(0, 2)
                        .norm_squared()
                        < 0.4
                {
                    return Some((*k, new_v));
                }
            }
            None
        })
        .collect();

    transform_maps1
}
pub fn track_one_point(
    image_pyramid0: &[GrayImage],
    image_pyramid1: &[GrayImage],
    transform0: &na::Affine2<f32>,
) -> Option<na::Affine2<f32>> {
    let levels = image_pyramid0.len() as u32;
    assert!(levels == image_pyramid1.len() as u32);
    let mut patch_valid = true;
    let mut transform1 = na::Affine2::<f32>::identity();
    transform1.matrix_mut_unchecked().m13 = transform0.matrix().m13;
    transform1.matrix_mut_unchecked().m23 = transform0.matrix().m23;

    for i in (0..levels).rev() {
        let scale_down = 1 << i;

        transform1.matrix_mut_unchecked().m13 /= scale_down as f32;
        transform1.matrix_mut_unchecked().m23 /= scale_down as f32;

        let pattern = patch::Pattern52::new(
            &image_pyramid0[i as usize],
            transform0.matrix().m13 / scale_down as f32,
            transform0.matrix().m23 / scale_down as f32,
        );
        patch_valid &= pattern.valid;
        if patch_valid {
            // Perform tracking on current level
            patch_valid &=
                track_point_at_level(&image_pyramid1[i as usize], &pattern, &mut transform1);
            if !patch_valid {
                return None;
            }
        } else {
            return None;
        }

        transform1.matrix_mut_unchecked().m13 *= scale_down as f32;
        transform1.matrix_mut_unchecked().m23 *= scale_down as f32;
        // transform1.matrix_mut_unchecked().m33 = 1.0;
    }
    let new_r_mat = transform0.matrix() * transform1.matrix();
    transform1.matrix_mut_unchecked().m11 = new_r_mat.m11;
    transform1.matrix_mut_unchecked().m12 = new_r_mat.m12;
    transform1.matrix_mut_unchecked().m21 = new_r_mat.m21;
    transform1.matrix_mut_unchecked().m22 = new_r_mat.m22;
    Some(transform1)
}

pub fn track_point_at_level(
    grayscale_image: &GrayImage,
    dp: &patch::Pattern52,
    transform: &mut na::Affine2<f32>,
) -> bool {
    // let mut patch_valid: bool = false;
    let optical_flow_max_iterations = 5;
    let patten = na::SMatrix::<f32, 52, 2>::from_fn(|i, j| {
        patch::Pattern52::PATTERN_RAW[i][j] / dp.pattern_scale_down
    })
    .transpose();
    // transform.
    // println!("before {}", transform.matrix());
    for _iteration in 0..optical_flow_max_iterations {
        let mut transformed_pat = transform.matrix().fixed_view::<2, 2>(0, 0) * patten;
        for i in 0..52 {
            transformed_pat
                .column_mut(i)
                .add_assign(transform.matrix().fixed_view::<2, 1>(0, 2));
        }
        // println!("{}", smatrix.transpose());
        // let mut res = na::SVector::<f32, PATTERN52_SIZE>::zeros();
        if let Some(res) = dp.residual(grayscale_image, &transformed_pat) {
            let inc = -dp.h_se2_inv_j_se2_t * res;

            // avoid NaN in increment (leads to SE2::exp crashing)
            if !inc.iter().all(|x| x.is_finite()) {
                return false;
            }
            if inc.norm() > 1e6 {
                return false;
            }
            let new_trans = transform.matrix() * image_utilities::se2_exp_matrix(&inc);
            *transform = na::Affine2::<f32>::from_matrix_unchecked(new_trans);
            let filter_margin = 2;
            if !image_utilities::inbound(
                grayscale_image,
                transform.matrix_mut_unchecked().m13,
                transform.matrix_mut_unchecked().m23,
                filter_margin,
            ) {
                return false;
            }
        }
    }

    true
}
