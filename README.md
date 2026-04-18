# patch-tracker-rs
[![crate](https://img.shields.io/crates/v/patch-tracker.svg)](https://crates.io/crates/patch-tracker)
[![PyPI - Version](https://img.shields.io/pypi/v/patch-tracker.svg)](https://pypi.org/project/patch-tracker)

```rust
use patch_tracker::PatchTracker;

let mut point_tracker = PatchTracker::default();
point_tracker.process_frame(&img_luma8);
```

```python
from patch_tracker import PatchTracker
tracker = PatchTracker()
tracker.process_frame(gray)
pts = tracker.get_track_points()
```

# Example
* [EuRoC dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
<img src="docs/euroc.avif" width="600" alt="Slow down for show case.">

* [TUM Visual-Inertial Dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)
<img src="docs/tum_vi.avif" width="600" alt="Slow down for show case.">

* [The UZH FPV Dataset](https://fpv.ifi.uzh.ch/datasets)
<img src="docs/uzh.avif" width="600" alt="Slow down for show case.">
