use image::GrayImage;
use wide::{i16x8, CmpGt, CmpLt};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Corner {
    pub x: u32,
    pub y: u32,
    pub score: f32,
}

impl Corner {
    pub fn new(x: u32, y: u32, score: f32) -> Corner {
        Corner { x, y, score }
    }
}

pub fn fast_corner_score(image: &GrayImage, threshold: u8, x: u32, y: u32) -> u8 {
    let mut max = 255u8;
    let mut min = threshold;

    loop {
        if max == min {
            return max;
        }

        let mean = ((max as u16 + min as u16) / 2u16) as u8;
        let probe = if max == min + 1 { max } else { mean };

        if is_corner_fast9_scalar(image, probe, x, y) {
            min = probe;
        } else {
            max = probe - 1;
        }
    }
}

#[inline(always)]
fn load_8u8_to_i16x8(ptr: *const u8) -> i16x8 {
    let bytes = unsafe { std::ptr::read_unaligned(ptr as *const [u8; 8]) };
    i16x8::new([
        bytes[0] as i16,
        bytes[1] as i16,
        bytes[2] as i16,
        bytes[3] as i16,
        bytes[4] as i16,
        bytes[5] as i16,
        bytes[6] as i16,
        bytes[7] as i16,
    ])
}

fn search_span<F>(circle: &[i16; 16], length: u8, f: F) -> bool
where
    F: Fn(&i16) -> bool,
{
    let mut nb_ok = 0u8;
    let mut nb_ok_start = None;
    for c in circle.iter() {
        if f(c) {
            nb_ok += 1;
            if nb_ok == length {
                return true;
            }
        } else {
            if nb_ok_start.is_none() {
                nb_ok_start = Some(nb_ok);
            }
            nb_ok = 0;
        }
    }
    nb_ok + nb_ok_start.unwrap_or(0) >= length
}

fn is_corner_fast9_scalar(image: &GrayImage, threshold: u8, x: u32, y: u32) -> bool {
    let c = image.get_pixel(x, y)[0] as i16;
    let low_thresh = c - threshold as i16;
    let high_thresh = c + threshold as i16;

    let p0 = image.get_pixel(x, y - 3)[0] as i16;
    let p8 = image.get_pixel(x, y + 3)[0] as i16;
    let p4 = image.get_pixel(x + 3, y)[0] as i16;
    let p12 = image.get_pixel(x - 3, y)[0] as i16;

    let above = (p12 > high_thresh || p4 > high_thresh) && (p8 > high_thresh || p0 > high_thresh);
    let below = (p12 < low_thresh || p4 < low_thresh) && (p8 < low_thresh || p0 < low_thresh);

    if !above && !below {
        return false;
    }

    let pixels = [
        p0,
        image.get_pixel(x + 1, y - 3)[0] as i16,
        image.get_pixel(x + 2, y - 2)[0] as i16,
        image.get_pixel(x + 3, y - 1)[0] as i16,
        p4,
        image.get_pixel(x + 3, y + 1)[0] as i16,
        image.get_pixel(x + 2, y + 2)[0] as i16,
        image.get_pixel(x + 1, y + 3)[0] as i16,
        p8,
        image.get_pixel(x - 1, y + 3)[0] as i16,
        image.get_pixel(x - 2, y + 2)[0] as i16,
        image.get_pixel(x - 3, y + 1)[0] as i16,
        p12,
        image.get_pixel(x - 3, y - 1)[0] as i16,
        image.get_pixel(x - 2, y - 2)[0] as i16,
        image.get_pixel(x - 1, y - 3)[0] as i16,
    ];

    if above && search_span(&pixels, 9, |&p| p > high_thresh) {
        return true;
    }
    if below && search_span(&pixels, 9, |&p| p < low_thresh) {
        return true;
    }
    false
}

pub fn simd_corners_fast9(image: &GrayImage, threshold: u8) -> Vec<Corner> {
    let width = image.width() as usize;
    let width_isize = width as isize;
    let height = image.height() as usize;
    let mut corners = Vec::new();

    if width < 7 || height < 7 {
        return corners;
    }

    let img_ptr = image.as_raw().as_ptr();

    let ring_offsets: [isize; 16] = [
        -3 * (width_isize),
        -3 * (width_isize) + 1,
        -2 * (width_isize) + 2,
        -(width_isize) + 3,
        3,
        (width_isize) + 3,
        2 * (width_isize) + 2,
        3 * (width_isize) + 1,
        3 * (width_isize),
        3 * (width_isize) - 1,
        2 * (width_isize) - 2,
        (width_isize) - 3,
        -3,
        -(width_isize) - 3,
        -2 * (width_isize) - 2,
        -3 * (width_isize) - 1,
    ];

    let t = i16x8::splat(threshold as i16);

    for y in 3..height - 3 {
        let mut x = 3;

        while x + 7 < width - 3 {
            let center_ptr = unsafe { img_ptr.add(y * width + x) };
            let center = load_8u8_to_i16x8(center_ptr);
            let high_thresh = center + t;
            let low_thresh = center - t;

            let p0 = load_8u8_to_i16x8(unsafe { center_ptr.offset(ring_offsets[0]) });
            let p8 = load_8u8_to_i16x8(unsafe { center_ptr.offset(ring_offsets[8]) });

            let above_0 = p0.simd_gt(high_thresh);
            let below_0 = p0.simd_lt(low_thresh);
            let above_8 = p8.simd_gt(high_thresh);
            let below_8 = p8.simd_lt(low_thresh);

            let above_08 = above_0 | above_8;
            let below_08 = below_0 | below_8;

            if (above_08 | below_08).to_bitmask() == 0 {
                x += 8;
                continue;
            }

            let p4 = load_8u8_to_i16x8(unsafe { center_ptr.offset(ring_offsets[4]) });
            let p12 = load_8u8_to_i16x8(unsafe { center_ptr.offset(ring_offsets[12]) });

            let above_4 = p4.simd_gt(high_thresh);
            let below_4 = p4.simd_lt(low_thresh);
            let above_12 = p12.simd_gt(high_thresh);
            let below_12 = p12.simd_lt(low_thresh);

            let count_above = ((above_0 | above_8) & (above_4 | above_12))
                | (above_0 & above_8)
                | (above_4 & above_12);
            let count_below = ((below_0 | below_8) & (below_4 | below_12))
                | (below_0 & below_8)
                | (below_4 & below_12);

            let pass_quick = count_above | count_below;

            if pass_quick.to_bitmask() == 0 {
                x += 8;
                continue;
            }

            let mut ring = [i16x8::splat(0); 16];
            ring[0] = p0;
            ring[4] = p4;
            ring[8] = p8;
            ring[12] = p12;
            for i in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15] {
                ring[i] = load_8u8_to_i16x8(unsafe { center_ptr.offset(ring_offsets[i]) });
            }

            let mut above = [i16x8::splat(0); 16];
            let mut below = [i16x8::splat(0); 16];
            for i in 0..16 {
                above[i] = ring[i].simd_gt(high_thresh);
                below[i] = ring[i].simd_lt(low_thresh);
            }

            let check_9 = |arr: &[i16x8; 16]| -> u32 {
                let mut a2 = [i16x8::splat(0); 16];
                for i in 0..16 {
                    a2[i] = arr[i] & arr[(i + 1) % 16];
                }

                let mut a4 = [i16x8::splat(0); 16];
                for i in 0..16 {
                    a4[i] = a2[i] & a2[(i + 2) % 16];
                }

                let mut a8 = [i16x8::splat(0); 16];
                for i in 0..16 {
                    a8[i] = a4[i] & a4[(i + 4) % 16];
                }

                let mut a9 = [i16x8::splat(0); 16];
                for i in 0..16 {
                    a9[i] = a8[i] & arr[(i + 8) % 16];
                }

                let mut final_a = a9[0];
                for item in a9.iter().skip(1) {
                    final_a |= item;
                }

                final_a.to_bitmask()
            };

            let mask_above = check_9(&above) & pass_quick.to_bitmask();
            let mask_below = check_9(&below) & pass_quick.to_bitmask();
            let final_mask = mask_above | mask_below;

            if final_mask != 0 {
                for i in 0..8 {
                    if (final_mask & (1 << i)) != 0 {
                        let cx = (x + i) as u32;
                        let cy = y as u32;
                        let score = fast_corner_score(image, threshold, cx, cy);
                        corners.push(Corner::new(cx, cy, score as f32));
                    }
                }
            }

            x += 8;
        }

        for cx in x..width - 3 {
            if is_corner_fast9_scalar(image, threshold, cx as u32, y as u32) {
                let score = fast_corner_score(image, threshold, cx as u32, y as u32);
                corners.push(Corner::new(cx as u32, y as u32, score as f32));
            }
        }
    }

    corners
}
