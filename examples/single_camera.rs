use glob::glob;
use image::ImageReader;
use patch_tracker::PatchTracker;

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    folder: String,
}

fn id_to_color(id: u64) -> [u8; 3] {
    const M: u32 = 2u32.pow(24);
    fastrand::seed(id);
    let color_num = fastrand::u32(0..M);
    [
        ((color_num >> 16) % 256) as u8,
        ((color_num >> 8) % 256) as u8,
        (color_num % 256) as u8,
    ]
}

fn main() {
    let args = Args::parse();

    env_logger::init();

    let path = args.folder;
    let mut path_list: Vec<PathBuf> = glob(format!("{}/*.png", path).as_str())
        .expect("Failed to read glob pattern")
        .filter_map(Result::ok)
        .collect();
    if path_list.is_empty() {
        println!("there's no png in this folder.");
        path_list = glob(format!("{}/*.jpg", path).as_str())
            .expect("Failed to read glob pattern")
            .filter_map(Result::ok)
            .collect();
        if path_list.is_empty() {
            println!("there's no jpg in this folder.");
            return;
        }
    }
    // let mut point_tracker = PatchTracker::new(4, 32);
    let mut point_tracker = PatchTracker::new(4, 20)
        .with_magic_point_threshold(0.5)
        .unwrap();

    let mut prev_points: std::collections::HashMap<usize, (f32, f32)> =
        std::collections::HashMap::new();

    const FPS: u32 = 10;
    let start_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let delta_time = 1.0 / FPS as f64;
    let rec = rerun::RecordingStreamBuilder::new("single camera")
        .spawn_opts(&rerun::SpawnOptions {
            port: 9875,
            ..Default::default()
        })
        .unwrap();

    for (i, path) in path_list.iter().enumerate() {
        let curr_img = ImageReader::open(path).unwrap().decode().unwrap();
        let curr_img_luma8 = curr_img.to_luma8();

        point_tracker.process_frame(&curr_img_luma8);

        rec.set_timestamp_secs_since_epoch("stable_time", start_time + delta_time * i as f64);
        rec.log("image", &rerun::EncodedImage::from_file(path).unwrap())
            .unwrap();

        let curr_points = point_tracker.get_track_points();

        let (colors, points): (Vec<_>, Vec<(f32, f32)>) = curr_points
            .iter()
            .map(|(&id, &(x, y))| {
                let color = id_to_color(id as u64);
                (color, (x + 0.5, y + 0.5))
            })
            .unzip();
        rec.log(
            "image/points",
            &rerun::Points2D::new(points).with_colors(colors),
        )
        .unwrap();

        let mut line_strips = Vec::new();
        let mut line_colors = Vec::new();

        for (&id, &(curr_x, curr_y)) in &curr_points {
            if let Some(&(prev_x, prev_y)) = prev_points.get(&id) {
                line_strips.push([(prev_x + 0.5, prev_y + 0.5), (curr_x + 0.5, curr_y + 0.5)]);
                line_colors.push(id_to_color(id as u64));
            }
        }

        if !line_strips.is_empty() {
            rec.log(
                "image/lines",
                &rerun::LineStrips2D::new(line_strips).with_colors(line_colors),
            )
            .unwrap();
        }

        prev_points = curr_points;
    }
}
