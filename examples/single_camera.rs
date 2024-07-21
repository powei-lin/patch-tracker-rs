use ab_glyph::{FontRef, PxScale};
use glob::glob;
use image::io::Reader as ImageReader;
use patch_tracker::PatchTracker;

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

#[show_image::main]
fn main() {
    let args = Args::parse();

    let path = args.folder;
    let path_list: Vec<PathBuf> = glob(format!("{}/*.png", path).as_str())
        .expect("Failed to read glob pattern")
        .filter_map(Result::ok)
        .collect();
    if path_list.len() == 0 {
        println!("there's no png in this folder.");
        return;
    }
    let mut point_tracker = PatchTracker::<4>::default();

    const FPS: u32 = 30;
    let window = create_window("image", Default::default()).unwrap();
    let mut i = 0;

    for event in window.event_channel().unwrap() {
        let start = Instant::now();
        let curr_img = ImageReader::open(&path_list[i]).unwrap().decode().unwrap();
        let curr_img_luma8 = curr_img.to_luma8();

        point_tracker.process_frame(&curr_img_luma8);

        // drawing
        let mut curr_img_rgb = curr_img.to_rgb8();
        let font = FontRef::try_from_slice(include_bytes!("DejaVuSans.ttf")).unwrap();

        let height = 10.0;
        let scale = PxScale {
            x: height,
            y: height,
        };

        for (id, corner) in &point_tracker.tracked_points_map {
            let x = corner.matrix().m13;
            let y = corner.matrix().m23;
            let mut rng = ChaCha8Rng::seed_from_u64(*id as u64);
            let color_num = rng.gen_range(0..2u32.pow(24));
            let color = Rgb([
                ((color_num >> 16) % 256) as u8,
                ((color_num >> 8) % 256) as u8,
                (color_num % 256) as u8,
            ]);
            draw_cross_mut(&mut curr_img_rgb, color, x as i32, y as i32);
            let text = format!("{}", id);
            draw_text_mut(
                &mut curr_img_rgb,
                color,
                x as i32,
                y as i32,
                scale,
                &font,
                &text,
            );
        }
        let output_name = format!("output/{:05}.png", i);
        let _ = curr_img_rgb.save(output_name);

        window.set_image("image-001", curr_img_rgb).unwrap();
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
        i += 1;
    }
}
