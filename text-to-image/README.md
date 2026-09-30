# Text to image

Generate an image from a text prompt.

*A demonstration / portfolio project.*

| | |
|---|---|
| model | tiny-SD (segmind/tiny-sd), UNet fine-tuned; VAE and text encoder frozen |
| data | Pokemon captions, 833 caption/image pairs at 512x512 |

## Result

| split | validation loss (lower is better) |
|---|---|
| train | 0.0574 |
| val | 0.0548 |
| best val | 0.0513 |

## Notes

- 12,600 steps at batch 4, lr 1e-5, on 833 pairs. The run was stopped when two successive checks gained less than 0.2% - the remaining 2,400 steps of the 15,000 budget were not spent.
- The best validation loss, 0.0513, was recorded at step 100; the loss then moved between 0.052 and 0.061 for the rest of the run. Diffusion training loss tracks sample quality loosely, so the samples are the thing to look at: `samples/` holds the base model and the fine-tuned model on the same four prompts.
- No held-out test split - this set is small enough that train and validation are the only two splits used.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
