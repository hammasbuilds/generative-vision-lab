# Super resolution

Reconstruct a 6x larger image from a downsampled input.

*A demonstration / portfolio project.*

| model | task | result |
|---|---|---|
| [msrgan](msrgan/) | msrgan - x6 super resolution (L1 + adversarial) | 27.574 |
| [realesrnet](realesrnet/) | realesrnet - x6 super resolution (L1 only) | 27.432 |
| [realesrgan](realesrgan/) | realesrgan - x6 super resolution (L1 + adversarial) | 27.352 |
| [esrgan](esrgan/) | esrgan - x6 super resolution (L1 + adversarial) | - |

## Notes

- `cut-h2z` and `cyclegan-h2z` train on the same horse2zebra dataset with matched 100-epoch schedules.
