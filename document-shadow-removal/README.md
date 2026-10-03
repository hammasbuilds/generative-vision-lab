# Document shadow removal

Remove shadows cast across a photographed document page.

*A demonstration / portfolio project.*

| | |
|---|---|
| model | DocShadow, fine-tuned from the published SD7K weights |
| data | ISTD, shadow / shadow-free page pairs |

## Result

| split | PSNR (dB) |
|---|---|
| train | 30.667 |
| val | 30.492 |
| test | 30.595 |

For comparison:

| | |
|---|---|
| published SD7K weights, no fine-tuning | 27.148 |

## Notes

- Fine-tuned for 56 epochs on 800 training pairs; the best val score came at epoch 50 and five further epochs did not beat it, so the run was stopped there.
- Train 30.67 / val 30.49 / test 30.60 - a tight spread across all three splits.
- Samples show the shadowed input, the published weights' output, this run's output and the ground truth.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
