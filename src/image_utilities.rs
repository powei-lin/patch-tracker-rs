use image::{GrayImage, Luma};
use imageproc::corners::{corners_fast9, Corner};
#[cfg(feature = "nalgebra033")]
use nalgebra_033 as na;
#[cfg(all(not(feature = "nalgebra033"), feature = "nalgebra034"))]
use nalgebra as na;

use rayon::prelude::*;

pub type ImageLumaf32 = image::ImageBuffer<Luma<f32>, Vec<f32>>;
pub type Vec3f = na::SVector<f32, 3>;

pub fn image_gray_to_f32(img: &GrayImage) -> ImageLumaf32 {
    let buf = img.par_iter().map(|&p| p as f32).collect();
    ImageLumaf32::from_vec(img.width(), img.height(), buf).expect("failed to f32")
}

// Computes 3-channel gradient (dx, dy, dx*dx + dy*dy) and intensity map from a greyscale image.
pub fn compute_gradients(img: &GrayImage) -> (Vec<f32>, Vec<Vec3f>) {
    let w = img.width();
    let h = img.height();
    let size = (w * h) as usize;

    let mut intensity = vec![0.0f32; size];
    let mut gradients = vec![Vec3f::zeros(); size];

    // Copy intensity and convert to f32
    // Simple copy for now, optimization: combine with loops below if needed, but rayon helps.
    img.pixels().enumerate().for_each(|(i, p)| {
        // Note: GrayImage iterator is row-major
        intensity[i] = p[0] as f32;
    });

    // Compute gradients (central difference, handling borders)
    // Parallel iter over rows for efficiency
    // Safety: we are careful with indices.
    // Optimization: could be better with unsafe unchecked access or SIMD, keeping it safe for now.

    let w = w as usize;
    let h = h as usize;

    // We can't easily parallelize mutable writes to the same Vec without unsafe or splitting slices.
    // For simplicity in this `safe` version, we iterate normally or use chunks.
    // Let's use simple iter for now, optimize later.

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            let g_idx = y + x * h;

            // Central difference
            // dx = (I(x+1) - I(x-1)) * 0.5
            // dy = (I(y+1) - I(y-1)) * 0.5

            let val_right = intensity[idx + 1];
            let val_left = intensity[idx - 1];
            let val_down = intensity[idx + w];
            let val_up = intensity[idx - w];

            let dx = 0.5 * (val_right - val_left);
            let dy = 0.5 * (val_down - val_up);

            // Third channel is often used for weight or gradient magnitude squared in DSO,
            // but for now we put just 1.0 or placeholder.
            // In original DSO: vec3(dx, dy, sqrt(dx*dx + dy*dy) ?) -> actually correlation.
            // Let's store (dx, dy, 1.0) for now.
            gradients[g_idx] = Vec3f::new(dx, dy, dx * dx + dy * dy);
        }
    }

    (intensity, gradients)
}
pub fn compute_gradients_f32(img: &ImageLumaf32) -> Vec<Vec3f> {
    let w = img.width();
    let h = img.height();
    let size = (w * h) as usize;

    let mut intensity = vec![0.0f32; size];
    let mut gradients = vec![Vec3f::zeros(); size];

    // Copy intensity and convert to f32
    // Simple copy for now, optimization: combine with loops below if needed, but rayon helps.
    img.pixels().enumerate().for_each(|(i, p)| {
        // Note: GrayImage iterator is row-major
        intensity[i] = p[0];
    });

    // Compute gradients (central difference, handling borders)
    // Parallel iter over rows for efficiency
    // Safety: we are careful with indices.
    // Optimization: could be better with unsafe unchecked access or SIMD, keeping it safe for now.

    let w = w as usize;
    let h = h as usize;

    // We can't easily parallelize mutable writes to the same Vec without unsafe or splitting slices.
    // For simplicity in this `safe` version, we iterate normally or use chunks.
    // Let's use simple iter for now, optimize later.

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            let g_idx = y + x * h;

            // Central difference
            // dx = (I(x+1) - I(x-1)) * 0.5
            // dy = (I(y+1) - I(y-1)) * 0.5

            let val_right = intensity[idx + 1];
            let val_left = intensity[idx - 1];
            let val_down = intensity[idx + w];
            let val_up = intensity[idx - w];

            let dx = 0.5 * (val_right - val_left);
            let dy = 0.5 * (val_down - val_up);

            // Third channel is often used for weight or gradient magnitude squared in DSO,
            // but for now we put just 1.0 or placeholder.
            // In original DSO: vec3(dx, dy, sqrt(dx*dx + dy*dy) ?) -> actually correlation.
            // Let's store (dx, dy, 1.0) for now.
            gradients[g_idx] = Vec3f::new(dx, dy, dx * dx + dy * dy);
        }
    }

    gradients
}

pub trait HalfSize {
    fn half_size(&self) -> Self;
}

impl HalfSize for GrayImage {
    fn half_size(&self) -> Self {
        let (w, h) = (self.width() as usize, self.height() as usize);
        let new_w = w / 2;
        let new_h = h / 2;
        let src_ptr = self.as_raw();
        // Pre-allocate the buffer to avoid reallocs
        let mut dst = Vec::with_capacity(new_w * new_h);

        unsafe {
            let s_ptr = src_ptr.as_ptr();
            let d_ptr: *mut u8 = dst.as_mut_ptr();

            for y in 0..new_h {
                // Pre-calculate row starts to avoid multiplications in the inner loop
                let row0 = s_ptr.add((y * 2) * w);
                let row1 = s_ptr.add((y * 2 + 1) * w);
                let dst_row = d_ptr.add(y * new_w);

                for x in 0..new_w {
                    let x2 = x * 2;
                    // Read 4 pixels (2x2 block)
                    let p00 = *row0.add(x2) as u32;
                    let p01 = *row0.add(x2 + 1) as u32;
                    let p10 = *row1.add(x2) as u32;
                    let p11 = *row1.add(x2 + 1) as u32;

                    // Average: (sum + 2) / 4. Adding 2 handles rounding correctly.
                    let avg = (p00 + p01 + p10 + p11 + 2) >> 2;

                    *dst_row.add(x) = avg as u8;
                }
            }
            dst.set_len(new_w * new_h);
        }

        // Wrap back into GrayImage if needed, or just black_box the Vec
        GrayImage::from_raw(new_w as u32, new_h as u32, dst).unwrap()
    }
}

impl HalfSize for ImageLumaf32 {
    fn half_size(&self) -> Self {
        let (w, h) = (self.width() as usize, self.height() as usize);
        let new_w = w / 2;
        let new_h = h / 2;
        let src_ptr = self.as_raw();
        let mut dst = Vec::with_capacity(new_w * new_h);

        unsafe {
            let s_ptr = src_ptr.as_ptr();
            let d_ptr: *mut f32 = dst.as_mut_ptr();

            for y in 0..new_h {
                let row0 = s_ptr.add(y * 2 * w);
                let row1 = row0.add(w);
                let dst_row = d_ptr.add(y * new_w);

                for x in 0..new_w {
                    let x2 = x * 2;
                    let avg =
                        (*row0.add(x2) + *row0.add(x2 + 1) + *row1.add(x2) + *row1.add(x2 + 1))
                            * 0.25;

                    *dst_row.add(x) = avg;
                }
            }
            dst.set_len(new_w * new_h);
        }
        ImageLumaf32::from_raw(new_w as u32, new_h as u32, dst).unwrap()
    }
}

pub fn to_u8_image(image: &ImageLumaf32) -> GrayImage {
    let u8_data = image
        .clone()
        .into_vec()
        .iter()
        .map(|&x| {
            if x < 0.0 {
                0u8
            } else if x > 255.0 {
                255u8
            } else {
                x as u8
            }
        })
        .collect::<Vec<u8>>();
    GrayImage::from_raw(image.width(), image.height(), u8_data).unwrap()
}

pub fn image_grad(grayscale_image: &GrayImage, x: f32, y: f32) -> na::SVector<f32, 3> {
    // inbound
    let ix = x.floor() as u32;
    let iy = y.floor() as u32;

    let dx = x - ix as f32;
    let dy = y - iy as f32;

    let ddx = 1.0 - dx;
    let ddy = 1.0 - dy;

    let px0y0 = grayscale_image.get_pixel(ix, iy).0[0] as f32;
    let px1y0 = grayscale_image.get_pixel(ix + 1, iy).0[0] as f32;
    let px0y1 = grayscale_image.get_pixel(ix, iy + 1).0[0] as f32;
    let px1y1 = grayscale_image.get_pixel(ix + 1, iy + 1).0[0] as f32;

    let res0 = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 + dx * dy * px1y1;

    let pxm1y0 = grayscale_image.get_pixel(ix - 1, iy).0[0] as f32;
    let pxm1y1 = grayscale_image.get_pixel(ix - 1, iy + 1).0[0] as f32;

    let res_mx = ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0 + dx * dy * px0y1;

    let px2y0 = grayscale_image.get_pixel(ix + 2, iy).0[0] as f32;
    let px2y1 = grayscale_image.get_pixel(ix + 2, iy + 1).0[0] as f32;

    let res_px = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 + dx * dy * px2y1;

    let res1 = 0.5 * (res_px - res_mx);

    let px0ym1 = grayscale_image.get_pixel(ix, iy - 1).0[0] as f32;
    let px1ym1 = grayscale_image.get_pixel(ix + 1, iy - 1).0[0] as f32;

    let res_my = ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1 + dx * dy * px1y0;

    let px0y2 = grayscale_image.get_pixel(ix, iy + 2).0[0] as f32;
    let px1y2 = grayscale_image.get_pixel(ix + 1, iy + 2).0[0] as f32;

    let res_py = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 + dx * dy * px1y2;

    let res2 = 0.5 * (res_py - res_my);

    na::SVector::<f32, 3>::new(res0, res1, res2)
}

pub fn point_in_bound(keypoint: &Corner, height: u32, width: u32, radius: u32) -> bool {
    keypoint.x >= radius
        && keypoint.x <= width - radius
        && keypoint.y >= radius
        && keypoint.y <= height - radius
}

pub fn inbound(image: &GrayImage, x: f32, y: f32, radius: u32) -> bool {
    let x = x.round() as u32;
    let y = y.round() as u32;

    x >= radius && y >= radius && x < image.width() - radius && y < image.height() - radius
}

pub fn se2_exp_matrix(a: &na::SVector<f32, 3>) -> na::SMatrix<f32, 3, 3> {
    let theta = a[2];
    let mut so2 = na::Rotation2::new(theta);
    let sin_theta_by_theta;
    let one_minus_cos_theta_by_theta;

    if theta.abs() < f32::EPSILON {
        let theta_sq = theta * theta;
        sin_theta_by_theta = 1.0f32 - 1.0 / 6.0 * theta_sq;
        one_minus_cos_theta_by_theta = 0.5f32 * theta - 1. / 24. * theta * theta_sq;
    } else {
        let cos = so2.matrix_mut_unchecked().m22;
        let sin = so2.matrix_mut_unchecked().m21;
        sin_theta_by_theta = sin / theta;
        one_minus_cos_theta_by_theta = (1. - cos) / theta;
    }
    let mut se2_mat = na::SMatrix::<f32, 3, 3>::identity();
    se2_mat.m11 = so2.matrix_mut_unchecked().m11;
    se2_mat.m12 = so2.matrix_mut_unchecked().m12;
    se2_mat.m21 = so2.matrix_mut_unchecked().m21;
    se2_mat.m22 = so2.matrix_mut_unchecked().m22;
    se2_mat.m13 = sin_theta_by_theta * a[0] - one_minus_cos_theta_by_theta * a[1];
    se2_mat.m23 = one_minus_cos_theta_by_theta * a[0] + sin_theta_by_theta * a[1];
    se2_mat
}

pub fn detect_key_points(
    image: &GrayImage,
    detect_image: &GrayImage,
    detect_scale: u32,
    grid_size: u32,
    current_corners: &Vec<Corner>,
    num_points_in_cell: u32,
) -> Vec<Corner> {
    const EDGE_THRESHOLD: u32 = 19;
    const MIN_THRESHOLD: u8 = 10;
    let h = image.height();
    let w = image.width();

    let x_start = (w % grid_size) / 2;
    let y_start = (h % grid_size) / 2;

    let grid_cols = (w / grid_size) as usize;
    let grid_rows = (h / grid_size) as usize;

    // Track how many points each cell already has
    let mut grid_count = vec![0u32; grid_rows * grid_cols];

    for corner in current_corners {
        if corner.x >= x_start && corner.y >= y_start {
            let gx = ((corner.x - x_start) / grid_size) as usize;
            let gy = ((corner.y - y_start) / grid_size) as usize;
            if gx < grid_cols && gy < grid_rows {
                grid_count[gy * grid_cols + gx] += 1;
            }
        }
    }

    // Run FAST9 on the detection image (possibly lower resolution) once
    let mut all_fast_corners = corners_fast9(detect_image, MIN_THRESHOLD);

    // Sort descending by score to prioritize the strongest corners
    all_fast_corners.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Assign best corners to empty grid cells
    let mut result = Vec::new();

    for mut corner in all_fast_corners {
        // Scale coordinates back to full resolution
        corner.x *= detect_scale;
        corner.y *= detect_scale;

        if !point_in_bound(&corner, h, w, EDGE_THRESHOLD) {
            continue;
        }
        if corner.x < x_start || corner.y < y_start {
            continue;
        }

        let gx = ((corner.x - x_start) / grid_size) as usize;
        let gy = ((corner.y - y_start) / grid_size) as usize;

        if gx >= grid_cols || gy >= grid_rows {
            continue;
        }

        let cell_idx = gy * grid_cols + gx;
        if grid_count[cell_idx] >= num_points_in_cell {
            continue;
        }

        grid_count[cell_idx] += 1;
        result.push(corner);
    }

    result
}
