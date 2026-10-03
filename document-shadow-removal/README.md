# Document shadow removal

Remove shadows cast across a photographed document page.

*A demonstration / portfolio project.*

| | |
|---|---|
| model | DocShadow, fine-tuned from the published SD7K weights |
| data | ISTD, shadow / shadow-free page pairs |

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
