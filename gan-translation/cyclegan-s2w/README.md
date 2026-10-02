# CycleGAN - summer to winter (unpaired)

*A demonstration / portfolio project.*

| metric | value |
|---|---|
| cycle-consistency L1 | 0.1049 |

## Notes

- 30 epochs on the summer2winter_yosemite set, not the 100 the other CycleGAN runs used. The run was stopped early: the GPU dropped to half its clock (1035 of 2100 MHz, 46 of 110 W) and an epoch went from 7 to 29 minutes, which put the remaining 70 epochs at about 34 hours for a cycle-consistency gain of 0.009 per 10 epochs.
- Samples show the same inputs translated at epochs 5 through 30, so the trend over training is visible even though the budget is shorter than apple2orange's.
