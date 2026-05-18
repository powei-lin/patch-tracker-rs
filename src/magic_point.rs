use crate::corners_fast9::Corner;
use image::GrayImage;
use ort::{ep::CoreML, session::Session, value::Tensor};

/// Fixed cell/grid size used by the MagicPoint (SuperPoint) architecture.
pub const CELL_SIZE: u32 = 8;
static MODEL_BYTES: &[u8] = include_bytes!("data/magic_point.onnx");

pub struct MagicPointDetector {
    session: Session,
    /// Minimum score threshold for keeping a detected keypoint.
    /// Scores are the per-cell softmax probabilities produced by the model.
    /// Range is (0, 1]; default is `0.0` (keep all keypoints).
    pub threshold: f32,
}

impl MagicPointDetector {
    pub fn new() -> Result<Self, ort::Error> {
        let session = Session::builder()?
            .with_execution_providers([CoreML::default().build()])?
            .commit_from_memory(MODEL_BYTES)?;
        // .commit_from_file(model_path)?;
        Ok(Self {
            session,
            threshold: 0.0,
        })
    }

    /// Set the minimum score threshold for keeping detected keypoints.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Run MagicPoint detection on `image`.
    ///
    /// The ONNX model (exported via `DetectWrapper`) already applies softmax,
    /// argmax-per-cell, dustbin filtering, and score thresholding internally.
    /// Its output is **`"keypoints"` (N, 2) float32** with each row = `[row, col]`.
    ///
    /// This function converts those coordinates to `Corner`s and applies
    /// a grid-based NMS to avoid placing new detections on cells that are
    /// already covered by `current_corners`.
    pub fn detect(&mut self, image: &GrayImage, current_corners: &[Corner]) -> Vec<Corner> {
        let h = image.height() as usize;
        let w = image.width() as usize;

        // Normalize to [0, 1] float32, shape (1, 1, H, W)
        let input_data: Vec<f32> = image.iter().map(|&p| p as f32 / 255.0).collect();
        let shape = vec![1i64, 1, h as i64, w as i64];

        let tensor = match Tensor::from_array((shape, input_data)) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("MagicPoint: failed to create input tensor: {e}");
                return vec![];
            }
        };

        let outputs = match self.session.run(ort::inputs![tensor]) {
            Ok(o) => o,
            Err(e) => {
                log::warn!("MagicPoint: inference failed: {e}");
                return vec![];
            }
        };

        // Output 0 "keypoints": (N, 2) float32, each row = [row, col]
        let (out_shape, kp_data) = match outputs[0].try_extract_tensor::<f32>() {
            Ok(t) => t,
            Err(e) => {
                log::warn!("MagicPoint: failed to extract keypoints tensor: {e}");
                return vec![];
            }
        };

        if out_shape.len() != 2 || out_shape[1] != 2 {
            log::warn!(
                "MagicPoint: unexpected keypoints shape {:?}, expected (N, 2)",
                out_shape
            );
            return vec![];
        }

        let n = out_shape[0] as usize;

        // Output 1 "scores": (N,) float32
        let scores: Vec<f32> = match outputs[1].try_extract_tensor::<f32>() {
            Ok((_, score_data)) => score_data.to_vec(),
            Err(e) => {
                log::warn!("MagicPoint: failed to extract scores tensor: {e}");
                vec![1.0f32; n]
            }
        };

        // Convert (row, col) → Corner { x = col, y = row }, filter by threshold
        let threshold = self.threshold;
        let detected: Vec<Corner> = (0..n)
            .filter(|&i| scores[i] >= threshold)
            .map(|i| {
                let row = kp_data[i * 2] as u32;
                let col = kp_data[i * 2 + 1] as u32;
                Corner::new(col, row, scores[i])
            })
            .collect();

        apply_grid_nms(detected, image.width(), image.height(), current_corners)
    }
}

// ---------------------------------------------------------------------------
// Grid-based NMS: keep at most one detection per CELL_SIZE×CELL_SIZE cell,
// skipping cells already occupied by currently-tracked corners.
// ---------------------------------------------------------------------------

fn apply_grid_nms(
    detected: Vec<Corner>,
    w: u32,
    h: u32,
    current_corners: &[Corner],
) -> Vec<Corner> {
    let grid_size = CELL_SIZE;
    let x_start = (w % grid_size) / 2;
    let y_start = (h % grid_size) / 2;
    let grid_cols = (w / grid_size) as usize;
    let grid_rows = (h / grid_size) as usize;

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

    let mut result = Vec::new();
    for corner in detected {
        if corner.x < x_start || corner.y < y_start {
            continue;
        }
        let gx = ((corner.x - x_start) / grid_size) as usize;
        let gy = ((corner.y - y_start) / grid_size) as usize;
        if gx >= grid_cols || gy >= grid_rows {
            continue;
        }
        let cell_idx = gy * grid_cols + gx;
        if grid_count[cell_idx] >= 1 {
            continue;
        }
        grid_count[cell_idx] += 1;
        result.push(corner);
    }
    result
}
