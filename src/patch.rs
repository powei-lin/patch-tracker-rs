use image::imageops;
use image::GrayImage;
#[cfg(all(not(feature = "nalgebra033"), feature = "nalgebra034"))]
use nalgebra as na;
#[cfg(feature = "nalgebra033")]
use nalgebra_033 as na;

use std::ops::AddAssign;
use wide::*;

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

        for (i, pattern_pos) in Self::PATTERN_RAW.into_iter().enumerate() {
            let p = self.pos
                + na::SVector::<f32, 2>::new(
                    pattern_pos[0] / self.pattern_scale_down,
                    pattern_pos[1] / self.pattern_scale_down,
                );
            jw_se2[(0, 2)] = -pattern_pos[1] / self.pattern_scale_down;
            jw_se2[(1, 2)] = pattern_pos[0] / self.pattern_scale_down;

            if image_utilities::inbound(greyscale_image, p.x, p.y, 2) {
                let val_grad = image_utilities::image_grad(greyscale_image, p.x, p.y);

                self.data[i] = val_grad[0];
                sum += val_grad[0];
                let d_i_d_se2 = val_grad.fixed_rows::<2>(1).transpose() * jw_se2;
                j_se2.set_row(i, &d_i_d_se2);
                grad_sum_se2.add_assign(j_se2.fixed_rows::<1>(i).transpose());
                num_valid_points += 1;
            } else {
                self.data[i] = -1.0;
            }
        }

        self.mean = sum / num_valid_points as f32;

        let mean_inv = num_valid_points as f32 / sum;

        let grad_sum_se2_div_sum = grad_sum_se2 / sum;
        let grad_sum_se2_div_sum_0 = f32x4::splat(grad_sum_se2_div_sum[0]);
        let grad_sum_se2_div_sum_1 = f32x4::splat(grad_sum_se2_div_sum[1]);
        let grad_sum_se2_div_sum_2 = f32x4::splat(grad_sum_se2_div_sum[2]);
        let mean_inv_v = f32x4::splat(mean_inv);
        let zero = f32x4::ZERO;

        for i in (0..Self::PATTERN_RAW.len()).step_by(4) {
            let mut data_v = f32x4::from(<[f32; 4]>::try_from(&self.data[i..i + 4]).unwrap());
            let mask = data_v.simd_ge(zero);

            // Update Jacobian columns
            // Column 0
            let mut col0 =
                f32x4::from(<[f32; 4]>::try_from(&j_se2.column(0).as_slice()[i..i + 4]).unwrap());
            col0 = mask.blend(col0 - grad_sum_se2_div_sum_0 * data_v, zero);
            j_se2.column_mut(0).as_mut_slice()[i..i + 4].copy_from_slice(&col0.to_array());

            // Column 1
            let mut col1 =
                f32x4::from(<[f32; 4]>::try_from(&j_se2.column(1).as_slice()[i..i + 4]).unwrap());
            col1 = mask.blend(col1 - grad_sum_se2_div_sum_1 * data_v, zero);
            j_se2.column_mut(1).as_mut_slice()[i..i + 4].copy_from_slice(&col1.to_array());

            // Column 2
            let mut col2 =
                f32x4::from(<[f32; 4]>::try_from(&j_se2.column(2).as_slice()[i..i + 4]).unwrap());
            col2 = mask.blend(col2 - grad_sum_se2_div_sum_2 * data_v, zero);
            j_se2.column_mut(2).as_mut_slice()[i..i + 4].copy_from_slice(&col2.to_array());

            // Update data
            data_v = mask.blend(data_v * mean_inv_v, data_v);
            self.data[i..i + 4].copy_from_slice(&data_v.to_array());
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

        if let Some(x) = h_se2.cholesky() {
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
        let weight = f32x4::splat(num_valid_points as f32 / sum);
        let zero = f32x4::ZERO;

        for i in (0..PATTERN52_SIZE).step_by(4) {
            let res_v = f32x4::from(<[f32; 4]>::try_from(&residual.as_slice()[i..i + 4]).unwrap());
            let data_v = f32x4::from(<[f32; 4]>::try_from(&self.data[i..i + 4]).unwrap());

            let mask = res_v.simd_ge(zero) & data_v.simd_ge(zero);
            num_residuals += mask.to_bitmask().count_ones();

            let final_res = mask.blend(weight * res_v - data_v, zero);
            residual.as_mut_slice()[i..i + 4].copy_from_slice(&final_res.to_array());
        }
        if num_residuals as usize > PATTERN52_SIZE / 2 {
            Some(residual)
        } else {
            None
        }
    }
}
