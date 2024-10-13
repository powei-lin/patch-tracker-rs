use ab_glyph::{FontRef, PxScale};
use glob::glob;
use image::ImageReader;
use patch_tracker::StereoPatchTracker;

use image::Rgb;
use imageproc::drawing::{draw_cross_mut, draw_text_mut};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use show_image::{create_window, event};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    folder: String,
}

fn id_to_color(id: u64) -> [u8; 3] {
    let mut rng = ChaCha8Rng::seed_from_u64(id);
    let color_num = rng.gen_range(0..2u32.pow(24));
    [
        ((color_num >> 16) % 256) as u8,
        ((color_num >> 8) % 256) as u8,
        (color_num % 256) as u8,
    ]
}

#[show_image::main]
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
    let mut point_tracker = StereoPatchTracker::<4>::default();

    const FPS: u32 = 5;
    let window = create_window("image", Default::default()).unwrap();
    let font = FontRef::try_from_slice(include_bytes!("DejaVuSans.ttf")).unwrap();

    for (i, event) in window.event_channel().unwrap().into_iter().enumerate() {
        let start = Instant::now();
        if i >= path_list0.len() {
            break;
        }
        let curr_img0 = ImageReader::open(&path_list0[i]).unwrap().decode().unwrap();
        let curr_img1 = ImageReader::open(&path_list1[i]).unwrap().decode().unwrap();
        let curr_img0_luma8 = curr_img0.to_luma8();
        let curr_img1_luma8 = curr_img1.to_luma8();

        point_tracker.process_frame(&curr_img0_luma8, &curr_img1_luma8);

        // drawing
        let mut curr_img_rgb0 = curr_img0.to_rgb8();
        let mut curr_img_rgb1 = curr_img1.to_rgb8();

        let height = 10.0;
        let scale = PxScale {
            x: height,
            y: height,
        };

        let [tracked_pts0, tracked_pts1] = point_tracker.get_track_points();
        for (id, (x, y)) in tracked_pts0 {
            let color = Rgb(id_to_color(id as u64));
            draw_cross_mut(&mut curr_img_rgb0, color, x as i32, y as i32);
            let text = format!("{}", id);
            draw_text_mut(
                &mut curr_img_rgb0,
                color,
                x as i32,
                y as i32,
                scale,
                &font,
                &text,
            );
        }
        for (id, (x, y)) in tracked_pts1 {
            let color = Rgb(id_to_color(id as u64));
            draw_cross_mut(&mut curr_img_rgb1, color, x as i32, y as i32);
            let text = format!("{}", id);
            draw_text_mut(
                &mut curr_img_rgb1,
                color,
                x as i32,
                y as i32,
                scale,
                &font,
                &text,
            );
        }
        // let output_name = format!("output/{:05}.png", i);
        // let _ = curr_img_rgb.save(output_name);
        // combine images
        let width = curr_img_rgb0.width();
        let height = curr_img_rgb0.height() * 2;
        let mut c0 = curr_img_rgb0.to_vec();
        let mut c1 = curr_img_rgb1.to_vec();
        c0.append(&mut c1);
        let cur = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(width, height, c0).unwrap();
        window.set_image("image-001", cur).unwrap();
        if let event::WindowEvent::KeyboardInput(event) = event {
            println!("{:#?}", event);
            if event.input.key_code == Some(event::VirtualKeyCode::Escape)
                && event.input.state.is_pressed()
            {
                break;
            }
        }
        let duration = start.elapsed();

        let one_frame = Duration::new(0, 1_000_000_000u32 / FPS);
        if let Some(rest) = one_frame.checked_sub(duration) {
            ::std::thread::sleep(rest);
        }
    }
}
