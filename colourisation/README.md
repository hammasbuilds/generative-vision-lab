# Colourisation

Predict colour (ab) from luminance (L) in Lab space.

*A demonstration / portfolio project.*

| | |
|---|---|
| model | ColourNet, 4.38 M parameters, trained from scratch |
| data | WIDER FACE, 2,850 crowded real-world scenes at 512x512, split 2280/428/142 |

## Result

| split | PSNR (dB) |
|---|---|
| train | 22.539 |
| val | 22.267 |
| test | 21.911 |

## Notes

- Stopped at epoch 100 of 120. Training loss was still falling (0.0836 -> 0.0825) while validation drifted DOWN (22.241 -> 22.220): overfitting, so more epochs would have made it worse. The best checkpoint is kept.
- Scene variety matters more than resolution for a colour prior, which is why WIDER FACE was used rather than the higher-resolution DIV2K also available here. 315 near-grey and 60 low-colour-spread images were dropped when the set was built.
- Learning rate was measured, not assumed: over 10 epochs each, 1e-5 gained +0.0169 with a 0.013 dB spread while 1e-4 gained +0.0036 with a 0.187 dB spread and ended lower. 1e-5 was used.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
