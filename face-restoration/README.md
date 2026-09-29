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

- UNDERSTATED, and worth saying so. The run reached 29.98 dB by step 11,000 and then gained nothing for its final 4,000 steps because its learning rate had collapsed to 1e-7 - a scheduling bug, not convergence. The number is real but the model was not finished learning when the budget ran out.
- No discriminator is published for GFPGANv1, so this trains the generator alone with L1. It is not the paper's adversarial recipe and is not presented as one.
- PSNR is a weak proxy for face restoration - it rewards smoothness, which is the opposite of the goal. Judge the sample images.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
