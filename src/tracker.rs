use image::{imageops, GrayImage};
use imageproc::corners::Corner;
use nalgebra as na;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::AddAssign;

use crate::{image_utilities, patch};

use log::info;

#[derive(Default)]
pub struct PatchTracker<const N: u32> {
    last_keypoint_id: usize,
    pub tracked_points_map: HashMap<usize, na::Affine2<f32>>,
    previous_image_pyramid: Vec<GrayImage>,
    initialized: bool,
}
impl<const LEVELS: u32> PatchTracker<LEVELS> {
    pub fn process_frame(&mut self, greyscale_image: &GrayImage) {
        const FILTER_TYPE: imageops::FilterType = imageops::FilterType::Nearest;
        // const FILTER_TYPE: imageops::FilterType = imageops::FilterType::Triangle;
        // build current image pyramid
        let (w, h) = greyscale_image.dimensions();

        let current_image_pyramid: Vec<GrayImage> = (0..LEVELS)
            .into_par_iter()
            .map(|i| {
                let scale_down: u32 = 1 << i;
                let (new_w, new_h) = (w / scale_down, h / scale_down);
                imageops::resize(greyscale_image, new_w, new_h, FILTER_TYPE)
            })
            .collect();

        if self.initialized {
            info!("old points {}", self.tracked_points_map.len());
            // track prev points
            self.tracked_points_map = track_points::<LEVELS>(
                &self.previous_image_pyramid,
                &current_image_pyramid,
                &self.tracked_points_map,
            );
            info!("tracked old points {}", self.tracked_points_map.len());

            // add new points
            let new_points = add_points(&self.tracked_points_map, greyscale_image);
            for point in &new_points {
                let mut v = na::Affine2::<f32>::identity();

                v.matrix_mut_unchecked().m13 = point.x as f32;
                v.matrix_mut_unchecked().m23 = point.y as f32;
                self.tracked_points_map.insert(self.last_keypoint_id, v);
                self.last_keypoint_id += 1;
            }
        } else {
            // add new points
            let new_points = add_points(&self.tracked_points_map, greyscale_image);
            for point in &new_points {
                let mut v = na::Affine2::<f32>::identity();

                v.matrix_mut_unchecked().m13 = point.x as f32;
                v.matrix_mut_unchecked().m23 = point.y as f32;
                self.tracked_points_map.insert(self.last_keypoint_id, v);
                self.last_keypoint_id += 1;
            }
            self.initialized = true;
        }

        // update saved image pyramid
        self.previous_image_pyramid = current_image_pyramid;
    }
}
fn add_points(
    tracked_points_map: &HashMap<usize, na::Affine2<f32>>,
    grayscale_image: &GrayImage,
) -> Vec<Corner> {
    const GRID_SIZE: u32 = 50;
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
    image_utilities::detect_key_points(
        grayscale_image,
        GRID_SIZE,
        &current_corners,
        num_points_in_cell,
    )
    // let mut prev_points =
    // Eigen::aligned_vector<Eigen::Vector2d> pts0;

    // for (const auto &kv : observations.at(0)) {
    //   pts0.emplace_back(kv.second.translation().template cast<double>());
    // }
}
fn track_points<const LEVELS: u32>(
    image_pyramid0: &[GrayImage],
    image_pyramid1: &[GrayImage],
    transform_maps0: &HashMap<usize, na::Affine2<f32>>,
) -> HashMap<usize, na::Affine2<f32>> {
    let transform_maps1: HashMap<usize, na::Affine2<f32>> = transform_maps0
        .par_iter()
        .filter_map(|(k, v)| {
            if let Some(new_v) = track_one_point::<LEVELS>(image_pyramid0, image_pyramid1, v) {
                // return Some((k.clone(), new_v));
                if let Some(old_v) =
                    track_one_point::<LEVELS>(image_pyramid1, image_pyramid0, &new_v)
                {
                    if (v.matrix() - old_v.matrix())
                        .fixed_view::<2, 1>(0, 2)
                        .norm_squared()
                        < 0.4
                    {
                        return Some((*k, new_v));
                    }
                }
            }
            None
        })
        .collect();

    transform_maps1
}
fn track_one_point<const LEVELS: u32>(
    image_pyramid0: &[GrayImage],
    image_pyramid1: &[GrayImage],
    transform0: &na::Affine2<f32>,
) -> Option<na::Affine2<f32>> {
    let mut patch_valid = true;
    let mut transform1 = na::Affine2::<f32>::identity();
    transform1.matrix_mut_unchecked().m13 = transform0.matrix().m13;
    transform1.matrix_mut_unchecked().m23 = transform0.matrix().m23;

    for i in (0..LEVELS).rev() {
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
