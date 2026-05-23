# Model Test Results

This table tracks the best test result for each model modification.

Metric note: `CD-mean = (CD-h + CD-o + CD-c) / 3`. Lower is better for Chamfer metrics.

| Date | Model / Change | Run | Eval Split | Eval Type | Best Step | Selection Metric | Loss | CD-h | CD-o | CD-c | CD-mean | Notes |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-05-21 | Single-stream DiT baseline | `outputs/unimodel_wai_real_smoke` | BEHAVE heldout | Periodic, 1 heldout batch/checkpoint | 3000 | Lowest balanced Chamfer `CD-mean` | 0.469080 | 0.999356 | 1.023499 | 0.788353 | 0.937069 | 35k-step run; periodic curve suggests overfitting after about 3000-4000 steps. Final full heldout eval at step 35000: loss 0.468048, CD-h 0.877061, CD-o 0.891454, CD-c 0.670546. |
| 2026-05-22 | Dual-stream DiT + dense contact | `outputs/HOI_contact` | BEHAVE heldout | Periodic, 1 heldout batch/checkpoint | 400 | Lowest balanced Chamfer `CD-mean` | 0.536751 | 0.968247 | 0.883364 | 0.758166 | 0.869926 | 35k-step run with `lambda_dense_contact=0.01`; periodic best improves CD-mean by 0.067143 (-7.17%) vs single-stream best, mainly from lower object Chamfer. Final full heldout eval at step 35000: loss 0.466278, CD-h 0.821474, CD-o 0.878777, CD-c 0.655671, CD-mean 0.785307 (-3.41% vs single-stream final full CD-mean 0.813020). Periodic final checkpoint CD-mean is 1.204004, so checkpoint selection still favors the early window. |

## Single vs Dual Stream

Lower is better for all Chamfer metrics. On the same BEHAVE heldout protocol, the dual-stream run is better than the single-stream baseline under both comparison views:

| Comparison | Single-stream CD-mean | Dual-stream CD-mean | Delta | Relative |
| --- | ---: | ---: | ---: | ---: |
| Periodic best checkpoint | 0.937069 | 0.869926 | -0.067143 | -7.17% |
| Final full heldout at 35000 | 0.813020 | 0.785307 | -0.027713 | -3.41% |
