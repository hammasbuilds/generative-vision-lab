# pix2pix - semantic label map to building photograph

*A demonstration / portfolio project.*

| metric | value |
|---|---|
| val PSNR | 12.697 |

## Notes

- PSNR is close to meaningless on this task and the number should not be read as a failure. Measured on this data: shifting the CORRECT answer by two pixels scores 15.08 dB; two unrelated photographs score 10.29 dB; a 2px-blurred copy of the ground truth scores 22.83 dB. The usable range is roughly 10 to 23, and PSNR actively rewards blur.
- One label map admits countless valid photographs, so pixel agreement measures the wrong thing. Judge the samples: at epoch 200 the model produces cornices, window frames and doorways from coloured rectangles.
- 200 epochs = the published schedule (100 constant + 100 decay).
