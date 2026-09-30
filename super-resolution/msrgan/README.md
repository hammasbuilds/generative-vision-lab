# msrgan - x6 super resolution (L1 + adversarial)

*A demonstration / portfolio project.*

| metric | value |
|---|---|
| val PSNR | 27.574 |

## Notes

- Trained with L1 + adversarial.
- The x4 published checkpoint was loaded into an x6 architecture: RRDBNet upsamples by interpolate-then-conv so every weight transfers, while MSRResNet uses PixelShuffle and its second upsampling conv had to widen from 256 to 576 channels and be reinitialised. The transfer count is printed at load time rather than assumed.
- PART OF THIS RUN TRAINED AT THE 1e-5 FLOOR. A learning-rate override rebuilt a decaying schedule on top of the constant one, so the rate fell to the floor while the log still reported it held. The number is real but understated, and these four are not a clean architecture comparison.
