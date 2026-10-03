# pix2pix - semantic label map to building photograph

*A demonstration / portfolio project.*

| metric | value |
|---|---|
| val PSNR | 12.697 |

## Notes

- 200 epochs on 400 training pairs - the published schedule of 100 constant plus 100 linear decay.
- Samples show the semantic label map, the generated photograph and the ground truth at epochs 5 through 200.
- pix2pix-facades-l1 is the matched control: identical architecture, data, batch, rate and schedule, with the adversarial term switched off. It scores 13.613 PSNR / 0.4369 SSIM against this run's 12.697 / 0.3928.
