# esrgan - x6 super resolution (L1 + adversarial)

*A demonstration / portfolio project.*

| metric | value |
|---|---|
| val PSNR | - |

## Notes

- Trained with L1 + adversarial.
- Fine-tuned from the published x4 checkpoint into an x6 architecture, with the upsampling stage rebuilt for a factor of 6.
- 20,000 steps on 720 training frames, evaluated on held-out val and test splits.
