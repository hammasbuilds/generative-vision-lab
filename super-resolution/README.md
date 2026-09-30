# Super resolution

Reconstruct a 6x larger image from a downsampled input.

*A demonstration / portfolio project.*

| model | task | result |
|---|---|---|
| [msrgan](msrgan/) | msrgan - x6 super resolution (L1 + adversarial) | 27.574 |
| [realesrnet](realesrnet/) | realesrnet - x6 super resolution (L1 only) | 27.432 |

## Notes

- `cut-h2z` and `cyclegan-h2z` share the horse2zebra dataset and a matched 100-epoch schedule, so the two unpaired methods can be compared directly. The other models vary the task rather than the method and are not a ranking.
