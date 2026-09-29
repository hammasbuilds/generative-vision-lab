# Image-to-image translation

Three architectures across five datasets.

*A demonstration / portfolio project.*

| model | task | result |
|---|---|---|
| [pix2pix-facades](pix2pix-facades/) | pix2pix - semantic label map to building photograph | 12.697 |
| [pix2pix-maps](pix2pix-maps/) | pix2pix - map tile to aerial photograph | 14.934 |
| [cut-h2z](cut-h2z/) | CUT - horse to zebra (contrastive, unpaired) | see folder |

## Notes

- `cut-h2z` and `cyclegan-h2z` share the horse2zebra dataset and a matched 100-epoch schedule, so the two unpaired methods can be compared directly. The other models vary the task rather than the method and are not a ranking.
