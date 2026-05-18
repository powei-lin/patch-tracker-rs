use glob::glob;
use image::ImageReader;
use patch_tracker::StereoPatchTracker;

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
    let path_list0: Vec<PathBuf> = glob(format!("{}/mav0/cam0/data/*.png", path).as_str())
        .expect("Failed to read glob pattern")
        .filter_map(Result::ok)
        .collect();
    if path_list0.is_empty() {
        println!("there's no png in this folder.");
        return;
    }
    let path_list1: Vec<PathBuf> = glob(format!("{}/mav0/cam1/data/*.png", path).as_str())
        .expect("Failed to read glob pattern")
        .filter_map(Result::ok)
        .collect();
    if path_list1.is_empty() {
        println!("there's no png in this folder.");
        return;
    }
    // let mut point_tracker = StereoPatchTracker::new(5, 16);
    let mut point_tracker = StereoPatchTracker::new(5, 8)
        .with_magic_point_threshold(0.5)
        .unwrap();

    let mut prev_points0: std::collections::HashMap<usize, (f32, f32)> =
        std::collections::HashMap::new();
    let mut prev_points1: std::collections::HashMap<usize, (f32, f32)> =
        std::collections::HashMap::new();

    const FPS: u32 = 5;
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

    for (i, (path0, path1)) in path_list0.iter().zip(path_list1.iter()).enumerate() {
        if i >= path_list0.len() {
            break;
        }
        let curr_img0 = ImageReader::open(path0).unwrap().decode().unwrap();
        let curr_img1 = ImageReader::open(path1).unwrap().decode().unwrap();
        let curr_img0_luma8 = curr_img0.to_luma8();
        let curr_img1_luma8 = curr_img1.to_luma8();

        point_tracker.process_frame(&curr_img0_luma8, &curr_img1_luma8);

        rec.set_timestamp_secs_since_epoch("stable_time", start_time + delta_time * i as f64);
        rec.log("image0", &rerun::EncodedImage::from_file(path0).unwrap())
            .unwrap();
        rec.log("image1", &rerun::EncodedImage::from_file(path1).unwrap())
            .unwrap();

        let [curr_points0, curr_points1] = point_tracker.get_track_points();

        // Cam0 points and lines
        let (colors0, points0): (Vec<_>, Vec<(f32, f32)>) = curr_points0
            .iter()
            .map(|(&id, &(x, y))| {
                let color = id_to_color(id as u64);
                (color, (x + 0.5, y + 0.5))
            })
            .unzip();
        rec.log(
            "image0/points",
            &rerun::Points2D::new(points0).with_colors(colors0),
        )
        .unwrap();

        let mut line_strips0 = Vec::new();
        let mut line_colors0 = Vec::new();
        for (&id, &(curr_x, curr_y)) in &curr_points0 {
            if let Some(&(prev_x, prev_y)) = prev_points0.get(&id) {
                line_strips0.push([(prev_x + 0.5, prev_y + 0.5), (curr_x + 0.5, curr_y + 0.5)]);
                line_colors0.push(id_to_color(id as u64));
            }
        }
        if !line_strips0.is_empty() {
            rec.log(
                "image0/lines",
                &rerun::LineStrips2D::new(line_strips0).with_colors(line_colors0),
            )
            .unwrap();
        }

        // Cam1 points and lines
        let (colors1, points1): (Vec<_>, Vec<(f32, f32)>) = curr_points1
            .iter()
            .map(|(&id, &(x, y))| {
                let color = id_to_color(id as u64);
                (color, (x + 0.5, y + 0.5))
            })
            .unzip();
        rec.log(
            "image1/points",
            &rerun::Points2D::new(points1).with_colors(colors1),
        )
        .unwrap();

        let mut line_strips1 = Vec::new();
        let mut line_colors1 = Vec::new();
        for (&id, &(curr_x, curr_y)) in &curr_points1 {
            if let Some(&(prev_x, prev_y)) = prev_points1.get(&id) {
                line_strips1.push([(prev_x + 0.5, prev_y + 0.5), (curr_x + 0.5, curr_y + 0.5)]);
                line_colors1.push(id_to_color(id as u64));
            }
        }
        if !line_strips1.is_empty() {
            rec.log(
                "image1/lines",
                &rerun::LineStrips2D::new(line_strips1).with_colors(line_colors1),
            )
            .unwrap();
        }

        prev_points0 = curr_points0;
        prev_points1 = curr_points1;
    }
}
