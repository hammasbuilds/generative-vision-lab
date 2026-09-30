# generative-vision-lab

Image-to-image restoration and translation, plus generation from text.

*A demonstration / portfolio project.*

| project | what | test result |
|---|---|---|
| [cloud-removal](cloud-removal/) | Satellite cloud removal | 23.696 |
| [colourisation](colourisation/) | Colourisation | 21.911 |
| [face-restoration](face-restoration/) | Face restoration | 30.020 |
| [super-resolution](super-resolution/) | 4 models | see folder |
| [text-to-image](text-to-image/) | Text to image | 0.0513 |
| [gan-translation](gan-translation/) | 3 models | see folder |

## Method

Every model is trained on a fixed 80/15/5 split and reported on train, validation and test.

Trained weights are not included: checkpoints run 109-294 MB against GitHub's 100 MB limit. Each project carries its full metric history and the code to reproduce the run.
