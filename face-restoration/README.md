# Face restoration

Restore degraded face crops. Degradation is synthesised at load time.

*A demonstration / portfolio project.*

| | |
|---|---|
| model | GFPGANv1, generator only |
| data | FFHQ, 1,000 faces at 512x512, split 800/150/50 |

## Result

| split | PSNR (dB) |
|---|---|
| train | 30.085 |
| val | 29.977 |
| test | 30.020 |

For comparison:

| | |
|---|---|
| before fine-tuning | 27.130 |

## Notes

- Generator fine-tuned with L1 on synthetically degraded FFHQ crops, 15,000 steps at batch 2.
- Sample images show the degraded input, the pretrained output and the fine-tuned output against ground truth.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
