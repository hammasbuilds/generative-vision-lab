# pix2pix - facades, L1 only (no adversarial term)

*A demonstration / portfolio project.*

| metric | value |
|---|---|
| val PSNR | 13.613 |
| val SSIM | 0.4369 |
| val PSNR, L1 + adversarial arm | 12.697 |
| val SSIM, L1 + adversarial arm | 0.3928 |

## Notes

- Same architecture, same data and the same 200-epoch budget as pix2pix-facades, with the adversarial term switched off. The budget is held identical on purpose so the only difference between the two runs is the loss.
- Measured on the same 60-image validation split: the L1-only arm scores higher on BOTH reference metrics, +0.916 dB PSNR and +0.044 SSIM. The common framing - that L1 wins PSNR while the adversarial arm wins on perceptual quality - is only half supported here, because SSIM favours L1 too.
- So the metrics do not settle which output looks better; that is what the sample panels are for. Neither PSNR nor SSIM is a perceptual score, and no perceptual metric was run.
