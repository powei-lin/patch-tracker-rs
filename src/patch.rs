use image::imageops;
use image::GrayImage;
use nalgebra as na;
use std::ops::AddAssign;

use crate::image_utilities;

pub const PATTERN52_SIZE: usize = 52;
pub struct Pattern52 {
    pub valid: bool,
    pub mean: f32,
    pub pos: na::SVector<f32, 2>,
    pub data: [f32; PATTERN52_SIZE], // negative if the point is not valid
    pub h_se2_inv_j_se2_t: na::SMatrix<f32, 3, PATTERN52_SIZE>,
    pub pattern_scale_down: f32,
}
impl Pattern52 {
    pub const PATTERN_RAW: [[f32; 2]; PATTERN52_SIZE] = [
        [-3.0, 7.0],
        [-1.0, 7.0],
        [1.0, 7.0],
        [3.0, 7.0],
        [-5.0, 5.0],
        [-3.0, 5.0],
        [-1.0, 5.0],
        [1.0, 5.0],
        [3.0, 5.0],
        [5.0, 5.0],
        [-7.0, 3.0],
        [-5.0, 3.0],
        [-3.0, 3.0],
        [-1.0, 3.0],
        [1.0, 3.0],
        [3.0, 3.0],
        [5.0, 3.0],
        [7.0, 3.0],
        [-7.0, 1.0],
        [-5.0, 1.0],
        [-3.0, 1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [3.0, 1.0],
        [5.0, 1.0],
        [7.0, 1.0],
        [-7.0, -1.0],
        [-5.0, -1.0],
        [-3.0, -1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
        [3.0, -1.0],
        [5.0, -1.0],
        [7.0, -1.0],
        [-7.0, -3.0],
        [-5.0, -3.0],
        [-3.0, -3.0],
        [-1.0, -3.0],
        [1.0, -3.0],
        [3.0, -3.0],
        [5.0, -3.0],
        [7.0, -3.0],
        [-5.0, -5.0],
        [-3.0, -5.0],
        [-1.0, -5.0],
        [1.0, -5.0],
        [3.0, -5.0],
        [5.0, -5.0],
        [-3.0, -7.0],
        [-1.0, -7.0],
        [1.0, -7.0],
        [3.0, -7.0],
    ];

    // verified
    pub fn set_data_jac_se2(
        &mut self,
        greyscale_image: &GrayImage,
        j_se2: &mut na::SMatrix<f32, PATTERN52_SIZE, 3>,
    ) {
        let mut num_valid_points = 0;
        let mut sum: f32 = 0.0;
        let mut grad_sum_se2 = na::SVector::<f32, 3>::zeros();

        let mut jw_se2 = na::SMatrix::<f32, 2, 3>::identity();
        let mut i = 0;

        for pattern_pos in Self::PATTERN_RAW {
            let p = self.pos
                + na::SVector::<f32, 2>::new(
                    pattern_pos[0] / self.pattern_scale_down,
                    pattern_pos[1] / self.pattern_scale_down,
                );
            jw_se2[(0, 2)] = -pattern_pos[1] / self.pattern_scale_down;
            jw_se2[(1, 2)] = pattern_pos[0] / self.pattern_scale_down;

            if image_utilities::inbound(&greyscale_image, p.x, p.y, 2) {
                let val_grad = image_utilities::image_grad(&greyscale_image, p.x, p.y);

                self.data[i] = val_grad[0];
                sum += val_grad[0];
                let re = val_grad.fixed_rows::<2>(1).transpose() * jw_se2;
                j_se2.set_row(i, &re);
                grad_sum_se2.add_assign(j_se2.fixed_rows::<1>(i).transpose());
                num_valid_points += 1;
            } else {
                self.data[i] = -1.0;
            }
            i += 1;
        }

        self.mean = sum / num_valid_points as f32;

        let mean_inv = num_valid_points as f32 / sum;

        for i in 0..Self::PATTERN_RAW.len() {
            if self.data[i] >= 0.0 {
                let rhs = grad_sum_se2.transpose() * self.data[i] / sum;
                j_se2.fixed_rows_mut::<1>(i).add_assign(-rhs);
                self.data[i] *= mean_inv;
            } else {
                j_se2.set_row(i, &na::SMatrix::<f32, 1, 3>::zeros());
            }
        }
        *j_se2 *= mean_inv;
    }
    pub fn new(greyscale_image: &GrayImage, px: f32, py: f32) -> Pattern52 {
        let mut j_se2 = na::SMatrix::<f32, PATTERN52_SIZE, 3>::zeros();
        let mut p = Pattern52 {
            valid: false,
            mean: 1.0,
            pos: na::SVector::<f32, 2>::new(px, py),
            data: [0.0; PATTERN52_SIZE], // negative if the point is not valid
            h_se2_inv_j_se2_t: na::SMatrix::<f32, 3, 52>::zeros(),
            pattern_scale_down: 2.0,
        };
        p.set_data_jac_se2(greyscale_image, &mut j_se2);
        let h_se2 = j_se2.transpose() * j_se2;
        let mut h_se2_inv = na::SMatrix::<f32, 3, 3>::identity();

        match h_se2.cholesky() {
            Some(x) => {
                x.solve_mut(&mut h_se2_inv);
                p.h_se2_inv_j_se2_t = h_se2_inv * j_se2.transpose();

                // NOTE: while it's very unlikely we get a source patch with all black
                // pixels, since points are usually selected at corners, it doesn't cost
                // much to be safe here.

                // all-black patch cannot be normalized; will result in mean of "zero" and
                // H_se2_inv_J_se2_T will contain "NaN" and data will contain "inf"
                p.valid = p.mean > f32::EPSILON
                    && p.h_se2_inv_j_se2_t.iter().all(|x| x.is_finite())
                    && p.data.iter().all(|x| x.is_finite());
            }
            None => {}
        }

        p
    }
    pub fn residual(
        &self,
        greyscale_image: &GrayImage,
        transformed_pattern: &na::SMatrix<f32, 2, PATTERN52_SIZE>,
    ) -> Option<na::SVector<f32, PATTERN52_SIZE>> {
        let mut sum: f32 = 0.0;
        let mut num_valid_points = 0;
        let mut residual = na::SVector::<f32, PATTERN52_SIZE>::zeros();
        for i in 0..PATTERN52_SIZE {
            if image_utilities::inbound(
                greyscale_image,
                transformed_pattern[(0, i)],
                transformed_pattern[(1, i)],
                2,
            ) {
                let p = imageops::interpolate_bilinear(
                    greyscale_image,
                    transformed_pattern[(0, i)],
                    transformed_pattern[(1, i)],
                );
                residual[i] = p.unwrap().0[0] as f32;
                sum += residual[i];
                num_valid_points += 1;
            } else {
                residual[i] = -1.0;
            }
        }

        // all-black patch cannot be normalized
        if sum < f32::EPSILON {
            return None;
        }

        let mut num_residuals = 0;

        for i in 0..PATTERN52_SIZE {
            if residual[i] >= 0.0 && self.data[i] >= 0.0 {
                let val = residual[i];
                residual[i] = num_valid_points as f32 * val / sum - self.data[i];
                num_residuals += 1;
            } else {
                residual[i] = 0.0;
            }
        }
        if num_residuals > PATTERN52_SIZE / 2 {
            Some(residual)
        } else {
            None
        }
    }
}
