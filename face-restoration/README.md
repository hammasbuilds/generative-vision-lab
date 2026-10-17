# Face restoration

Restore degraded face crops. The degradation is applied to clean FFHQ faces at load time.

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

## Notes

- Generator fine-tuned with L1 on FFHQ crops degraded at load time, 12,500 steps at batch 2 (`results.json`).
- Sample images show the degraded input, an earlier checkpoint of this run and the fine-tuned output against ground truth. The run resumed (its logged history starts at step 11,000) and the training script writes its `before` images when a run resumes, so they are not the pretrained weights.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
