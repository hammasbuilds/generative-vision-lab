# Satellite cloud removal

Remove haze and cloud from aerial imagery.

*A demonstration / portfolio project.*

| | |
|---|---|
| model | SpA-GAN generator (0.22 M parameters), fine-tuned |
| data | RICE, 2,000 aligned cloudy/clear pairs at 256x256, split 1600/300/100 |

## Result

| split | PSNR (dB) |
|---|---|
| train | 24.126 |
| val | 23.967 |
| test | 23.696 |

For comparison:

| | |
|---|---|
| do nothing (cloudy vs clear) | 11.90 |
| shipped RICE1 weights, zero-shot | 11.29 |
| shipped RICE2 weights, zero-shot | 12.46 |

## Notes

- The shipped RICE1 weights score BELOW doing nothing on this data, and RICE2 only 0.56 dB above it - RICE is satellite cirrus while this set is low-altitude aerial haze. This is a cross-domain fine-tune, not a continuation of a matched pretrained model, and the 60 epochs are doing nearly all of the work.
- Train 24.13 / val 23.97 / test 23.70 is a tight spread, so the gain is generalisation rather than memorisation.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
