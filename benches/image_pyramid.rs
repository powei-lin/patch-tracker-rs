use diol::prelude::*;
use image::{GrayImage, ImageReader};
use patch_tracker::build_image_pyramid;

fn read_test_image(path: &str) -> GrayImage {
    ImageReader::open(path)
        .unwrap()
        .decode()
        .unwrap()
        .to_luma8()
}

fn bench_build_image_pyramid(bencher: Bencher, levels: u32) {
    let img = read_test_image("tests/data/img0.png");
    bencher.bench(|| {
        build_image_pyramid(&img, levels);
    });
}

fn main() -> eyre::Result<()> {
    let bench = Bench::from_args()?;
    bench.register(
        "build_image_pyramid",
        bench_build_image_pyramid,
        [1, 2, 3, 4, 5],
    );
    bench.run()?;
    Ok(())
}
