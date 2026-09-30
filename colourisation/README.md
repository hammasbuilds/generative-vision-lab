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

- Trained from scratch for 100 epochs on 2,280 images, predicting the ab channels from L in Lab space.
- The dataset was filtered for colour content: near-grey and low-saturation images were dropped when it was built.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
