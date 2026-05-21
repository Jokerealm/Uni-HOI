# Single-Stream DiT 35K Training Result

Run: `outputs/unimodel_wai_real_smoke`

Configuration summary:

- Training data: `sample_data/WAI_prepared/sequences`
- Evaluation data: `sample_data/BEHAVE_heldout_prepared/sequences`
- Input mode: single image
- Coordinate mode: relative
- Max steps: 35000
- Global batch: 4
- Effective epochs: about 97.83
- Learning rate: `1e-4`, constant after 200 warmup steps
- Periodic evaluation: every 100 steps with `periodic_test_max_batches=1`
- Final evaluation: full BEHAVE heldout set with `test_max_batches=0`

## Step Selection

The periodic step curve is based on one heldout batch per checkpoint, so it is useful for trend selection rather than as the final benchmark number.

Using the balanced Chamfer score `CD-mean = (CD-h + CD-o + CD-c) / 3`, the best checkpoint is around **3000 steps**:

| Step | Loss | CD-h | CD-o | CD-c | CD-mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3000 | 0.469080 | 0.999356 | 1.023499 | 0.788353 | 0.937069 |

The single lowest contact Chamfer (`CD-c`) occurs at 200 steps, but this is an early one-batch spike and is less stable as the model-selection point:

| Step | Loss | CD-h | CD-o | CD-c | CD-mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.596124 | 0.983709 | 1.222353 | 0.740833 | 0.982298 |

## Overfitting Point

The run starts to overfit after roughly **3000-4000 steps**. The supervised training loss continues to decrease, reaching its lowest periodic value at 13300 steps, but heldout Chamfer gets worse after the early best window.

Reference points from the periodic curve:

| Step | Loss | CD-h | CD-o | CD-c | CD-mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3000 | 0.469080 | 0.999356 | 1.023499 | 0.788353 | 0.937069 |
| 4000 | 0.418481 | 1.023831 | 1.512329 | 0.833797 | 1.123319 |
| 13300 | 0.375784 | 1.028335 | 1.527626 | 0.854247 | 1.136736 |
| 35000 | 0.409374 | 1.013187 | 1.483922 | 0.918158 | 1.138422 |

## Final Full Heldout Evaluation

The final checkpoint at 35000 steps was also evaluated on the full BEHAVE heldout set:

| Step | Loss | CD-h | CD-o | CD-c |
| ---: | ---: | ---: | ---: | ---: |
| 35000 | 0.468048 | 0.877061 | 0.891454 | 0.670546 |

This full-set result is the current reported evaluation result for the completed 35000-step run. For checkpoint selection, the periodic trend still suggests using the 3000-step checkpoint as the best balanced checkpoint.
