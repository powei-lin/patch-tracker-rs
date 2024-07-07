# patch-tracker-rs

```rust
use patch_tracker::PatchTracker;

let mut point_tracker = PatchTracker::<4>::default();
point_tracker.process_frame(&img_luma8);
```

# Example
* [EuRoC dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
<img src="docs/euroc.avif" width="600" alt="Slow down for show case.">

* [TUM Visual-Inertial Dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)
<img src="docs/tum_vi.avif" width="600" alt="Slow down for show case.">

* [The UZH FPV Dataset](https://fpv.ifi.uzh.ch/datasets)
<img src="docs/uzh.avif" width="600" alt="Slow down for show case.">
