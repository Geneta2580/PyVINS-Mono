# Local model checkpoints

Runtime inference reads model weights only from this directory:

- `dino_salad.ckpt`
  - <https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt>
  - SHA-256: `6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada`
- `dinov2_vitb14_pretrain.pth`
  - <https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth>
  - SHA-256: `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`

The weight files are intentionally ignored by Git. Model code remains pinned
separately; SALAD is a Git submodule and DINOv2 is selected through
`loop_detector_dinov2_path`.

New inference models should follow the same convention: keep model code in a
repository/submodule, store weights in this directory, and expose the weight
path as a configuration entry. Runtime code must not implicitly download a
checkpoint.
