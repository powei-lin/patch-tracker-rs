use diol::prelude::*;
use image::{GrayImage, ImageReader};
use patch_tracker::{
    build_image_pyramid, na, track_one_point, track_points, PatchTracker, Pattern52,
};
use std::collections::HashMap;

fn read_test_image(path: &str) -> GrayImage {
    ImageReader::open(path)
        .unwrap()
        .decode()
        .unwrap()
        .to_luma8()
}

fn load_test_images() -> (GrayImage, GrayImage) {
    (
        read_test_image("tests/data/img0.png"),
        read_test_image("tests/data/img1.png"),
    )
}

fn bench_process_frame(bencher: Bencher, _: ()) {
    let (img0, img1) = load_test_images();
    bencher.bench(|| {
        let mut tracker = PatchTracker::<4>::default();
        tracker.process_frame(&img0);
        tracker.process_frame(&img1);
    });
}

fn bench_track_points(bencher: Bencher, num_points: usize) {
    let (img0, img1) = load_test_images();
    let pyramid0 = build_image_pyramid(&img0, 4);
    let pyramid1 = build_image_pyramid(&img1, 4);

    // Detect points from first frame and take num_points
    let mut tracker = PatchTracker::<4>::default();
    tracker.process_frame(&img0);
    let all_points = tracker.get_track_points();

    let transform_maps: HashMap<usize, na::Affine2<f32>> = all_points
        .iter()
        .take(num_points)
        .map(|(&id, &(x, y))| {
            let mut v = na::Affine2::<f32>::identity();
            v.matrix_mut_unchecked().m13 = x;
            v.matrix_mut_unchecked().m23 = y;
            (id, v)
        })
        .collect();

    bencher.bench(|| {
        track_points::<4>(&pyramid0, &pyramid1, &transform_maps);
    });
}

fn bench_track_one_point(bencher: Bencher, level: u32) {
    let (img0, img1) = load_test_images();
    let pyramid0 = build_image_pyramid(&img0, level);
    let pyramid1 = build_image_pyramid(&img1, level);
    let (w, h) = img0.dimensions();

    let mut transform0 = na::Affine2::<f32>::identity();
    transform0.matrix_mut_unchecked().m13 = w as f32 / 2.0;
    transform0.matrix_mut_unchecked().m23 = h as f32 / 2.0;

    match level {
        1 => {
            bencher.bench(|| {
                track_one_point::<1>(&pyramid0, &pyramid1, &transform0);
            });
        }
        2 => {
            bencher.bench(|| {
                track_one_point::<2>(&pyramid0, &pyramid1, &transform0);
            });
        }
        3 => {
            bencher.bench(|| {
                track_one_point::<3>(&pyramid0, &pyramid1, &transform0);
            });
        }
        4 => {
            bencher.bench(|| {
                track_one_point::<4>(&pyramid0, &pyramid1, &transform0);
            });
        }
        5 => {
            bencher.bench(|| {
                track_one_point::<5>(&pyramid0, &pyramid1, &transform0);
            });
        }
        _ => panic!("unsupported level {level}"),
    }
}

fn bench_pattern52_new(bencher: Bencher, _: ()) {
    let (img0, _) = load_test_images();
    let (w, h) = img0.dimensions();
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;

    bencher.bench(|| {
        Pattern52::new(&img0, cx, cy);
    });
}

fn main() -> eyre::Result<()> {
    let bench = Bench::from_args()?;
    bench.register("process_frame", bench_process_frame, [()]);
    bench.register("track_points", bench_track_points, [10, 50, 100, 200]);
    bench.register("track_one_point", bench_track_one_point, [1, 2, 3, 4, 5]);
    bench.register("pattern52_new", bench_pattern52_new, [()]);
    bench.run()?;
    Ok(())
}
