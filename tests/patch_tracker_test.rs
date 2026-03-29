use image::ImageReader;
use patch_tracker::{build_image_pyramid, track_one_point, track_points, PatchTracker, Pattern52};
use std::collections::HashMap;

fn load_test_images() -> (image::GrayImage, image::GrayImage) {
    let img0 = ImageReader::open("tests/data/img0.png")
        .expect("Failed to open img0.png")
        .decode()
        .expect("Failed to decode img0.png")
        .to_luma8();
    let img1 = ImageReader::open("tests/data/img1.png")
        .expect("Failed to open img1.png")
        .decode()
        .expect("Failed to decode img1.png")
        .to_luma8();
    (img0, img1)
}

#[test]
fn test_build_image_pyramid() {
    let (img0, _) = load_test_images();
    let (w, h) = img0.dimensions();
    let levels = 4;
    let pyramid = build_image_pyramid(&img0, levels);

    assert_eq!(pyramid.len(), levels as usize);
    assert_eq!(pyramid[0].dimensions(), (w, h));

    // Each level should be roughly half the previous
    for i in 1..levels as usize {
        let (pw, ph) = pyramid[i - 1].dimensions();
        let (cw, ch) = pyramid[i].dimensions();
        assert_eq!(cw, pw / 2);
        assert_eq!(ch, ph / 2);
    }
}

#[test]
fn test_pattern52_creation() {
    let (img0, _) = load_test_images();
    let (w, h) = img0.dimensions();

    // Create a pattern at the center of the image (guaranteed inbound)
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let pattern = Pattern52::new(&img0, cx, cy);

    assert!(pattern.valid, "Pattern at center of image should be valid");
    assert!(pattern.mean > 0.0);
}

#[test]
fn test_pattern52_edge_has_fewer_valid_points() {
    let (img0, _) = load_test_images();
    let (w, h) = img0.dimensions();

    // Pattern at center should have all 52 valid data points
    let center_pattern = Pattern52::new(&img0, w as f32 / 2.0, h as f32 / 2.0);
    assert!(center_pattern.valid);
    let center_valid_count = center_pattern.data.iter().filter(|&&v| v >= 0.0).count();
    assert_eq!(
        center_valid_count, 52,
        "Center pattern should have all 52 valid points"
    );

    // Pattern near the edge should have fewer valid data points
    // With pattern_scale_down=2.0, the max offset is 7/2=3.5 pixels
    // At position (2, 2), some pattern points will produce negative coordinates
    let edge_pattern = Pattern52::new(&img0, 2.0, 2.0);
    let edge_valid_count = edge_pattern.data.iter().filter(|&&v| v >= 0.0).count();
    assert!(
        edge_valid_count < center_valid_count,
        "Edge pattern should have fewer valid points than center pattern. \
         Edge: {edge_valid_count}, Center: {center_valid_count}"
    );
}

#[test]
fn test_track_one_point_identity() {
    let (img0, _) = load_test_images();
    let pyramid0 = build_image_pyramid(&img0, 4);

    // Tracking from image to itself should return approximately the same position
    let (w, h) = img0.dimensions();
    let mut transform0 = nalgebra::Affine2::<f32>::identity();
    transform0.matrix_mut_unchecked().m13 = w as f32 / 2.0;
    transform0.matrix_mut_unchecked().m23 = h as f32 / 2.0;

    let result = track_one_point::<4>(&pyramid0, &pyramid0, &transform0);
    assert!(
        result.is_some(),
        "Tracking a point on the same image should succeed"
    );

    let tracked = result.unwrap();
    let dx = (tracked.matrix().m13 - transform0.matrix().m13).abs();
    let dy = (tracked.matrix().m23 - transform0.matrix().m23).abs();

    assert!(
        dx < 1.0 && dy < 1.0,
        "Self-tracking should converge near the original position, got dx={dx}, dy={dy}"
    );
}

#[test]
fn test_track_points_between_frames() {
    let (img0, img1) = load_test_images();
    let pyramid0 = build_image_pyramid(&img0, 4);
    let pyramid1 = build_image_pyramid(&img1, 4);

    let (w, h) = img0.dimensions();

    // Create several points at various locations in the image
    let test_positions = vec![
        (w as f32 * 0.3, h as f32 * 0.3),
        (w as f32 * 0.5, h as f32 * 0.5),
        (w as f32 * 0.7, h as f32 * 0.3),
        (w as f32 * 0.3, h as f32 * 0.7),
        (w as f32 * 0.7, h as f32 * 0.7),
    ];

    let mut transform_maps: HashMap<usize, nalgebra::Affine2<f32>> = HashMap::new();
    for (i, (px, py)) in test_positions.iter().enumerate() {
        let mut v = nalgebra::Affine2::<f32>::identity();
        v.matrix_mut_unchecked().m13 = *px;
        v.matrix_mut_unchecked().m23 = *py;
        transform_maps.insert(i, v);
    }

    let tracked = track_points::<4>(&pyramid0, &pyramid1, &transform_maps);

    // Between two sequential frames, at least some points should be tracked successfully
    assert!(
        !tracked.is_empty(),
        "At least some points should be tracked between consecutive frames"
    );

    // Tracked points should have reasonable positions (within image bounds)
    for (_id, transform) in &tracked {
        let x = transform.matrix().m13;
        let y = transform.matrix().m23;
        assert!(
            x >= 0.0 && x < w as f32 && y >= 0.0 && y < h as f32,
            "Tracked point should be within image bounds, got ({x}, {y})"
        );
    }
}

#[test]
fn test_patch_tracker_process_first_frame() {
    let (img0, _) = load_test_images();
    let mut tracker = PatchTracker::<4>::default();

    tracker.process_frame(&img0);

    let points = tracker.get_track_points();
    assert!(
        !points.is_empty(),
        "After processing the first frame, points should be detected"
    );

    // All points should be within image bounds
    let (w, h) = img0.dimensions();
    for (_id, (x, y)) in &points {
        assert!(
            *x >= 0.0 && *x < w as f32 && *y >= 0.0 && *y < h as f32,
            "Detected point should be within image bounds, got ({x}, {y})"
        );
    }
}

#[test]
fn test_patch_tracker_process_two_frames() {
    let (img0, img1) = load_test_images();
    let mut tracker = PatchTracker::<4>::default();

    tracker.process_frame(&img0);
    let points_after_first = tracker.get_track_points();
    let num_first = points_after_first.len();

    tracker.process_frame(&img1);
    let points_after_second = tracker.get_track_points();
    let num_second = points_after_second.len();

    // After the second frame, we should still have tracked points
    assert!(
        num_second > 0,
        "After processing two frames, tracked points should remain"
    );

    // Some of the original points from the first frame should still exist
    let common_ids: Vec<_> = points_after_first
        .keys()
        .filter(|k| points_after_second.contains_key(k))
        .collect();

    assert!(
        !common_ids.is_empty(),
        "Some points from the first frame should be tracked into the second frame. \
         First frame had {num_first} points, second frame has {num_second} points."
    );

    // Tracked points should have moved (images are different) but not by a huge amount
    for id in &common_ids {
        let (x0, y0) = points_after_first[id];
        let (x1, y1) = points_after_second[id];
        let dist = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
        assert!(
            dist < 200.0,
            "Tracked point {id} moved too far: ({x0},{y0}) -> ({x1},{y1}), dist={dist}"
        );
    }
}

#[test]
fn test_patch_tracker_remove_id() {
    let (img0, _) = load_test_images();
    let mut tracker = PatchTracker::<4>::default();

    tracker.process_frame(&img0);
    let points = tracker.get_track_points();
    assert!(!points.is_empty());

    // Remove some IDs
    let ids_to_remove: Vec<usize> = points.keys().take(3).copied().collect();
    tracker.remove_id(&ids_to_remove);

    let points_after_remove = tracker.get_track_points();
    for id in &ids_to_remove {
        assert!(
            !points_after_remove.contains_key(id),
            "Point with id {id} should have been removed"
        );
    }
    assert_eq!(
        points_after_remove.len(),
        points.len() - ids_to_remove.len(),
        "Number of points should decrease by the number of removed IDs"
    );
}

#[test]
fn test_pattern52_residual_self() {
    let (img0, _) = load_test_images();
    let (w, h) = img0.dimensions();

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let pattern = Pattern52::new(&img0, cx, cy);
    assert!(pattern.valid);

    // Compute residual against itself at the same location -> residual should be near zero
    let patten_coords = nalgebra::SMatrix::<f32, 52, 2>::from_fn(|i, j| {
        Pattern52::PATTERN_RAW[i][j] / pattern.pattern_scale_down
    })
    .transpose();

    let transform = nalgebra::Affine2::<f32>::identity();
    let mut transformed_pat = transform.matrix().fixed_view::<2, 2>(0, 0) * patten_coords;
    for i in 0..52 {
        use std::ops::AddAssign;
        transformed_pat
            .column_mut(i)
            .add_assign(nalgebra::SVector::<f32, 2>::new(cx, cy));
    }

    let residual = pattern.residual(&img0, &transformed_pat);
    assert!(
        residual.is_some(),
        "Residual against itself should be valid"
    );

    let res = residual.unwrap();
    let max_residual = res.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_residual < 0.01,
        "Self-residual should be near zero, got max={max_residual}"
    );
}
