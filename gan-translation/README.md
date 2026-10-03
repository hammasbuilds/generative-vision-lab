# Image-to-image translation

Three architectures across five datasets.

*A demonstration / portfolio project.*

| model | task | result |
|---|---|---|
| [pix2pix-facades](pix2pix-facades/) | pix2pix - semantic label map to building photograph | 12.697 |
| [pix2pix-maps](pix2pix-maps/) | pix2pix - map tile to aerial photograph | 14.934 |
| [cut-h2z](cut-h2z/) | CUT - horse to zebra (contrastive, unpaired) | see folder |
| [cyclegan-a2o](cyclegan-a2o/) | CycleGAN - apple to orange (unpaired) | 0.0958 |
| [cyclegan-s2w](cyclegan-s2w/) | CycleGAN - summer to winter (unpaired) | 0.1049 |
| [pix2pix-facades-l1](pix2pix-facades-l1/) | pix2pix - facades, L1 only (no adversarial term) | 13.613 |

## Notes

- `cut-h2z` and `cyclegan-h2z` train on the same horse2zebra dataset with matched 100-epoch schedules.
