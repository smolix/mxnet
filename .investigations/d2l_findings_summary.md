# d2l-neu mxnet-errors.md triage summary (2026-05-18)

Reviewed 16 reported failures against the 2026-05-18 wheel/build.

## Wheel-side fixes

| Cat | What | Status | Fix |
|---|---|---|---|
| 2 | DeadKernel: transformer, self-attention-positional-encoding, bert | ✅ Already fixed | commit `c57970216` (libdl.so glibc 2.34+ fallback). All 3 notebooks now pass. |
| 1 | "could not execute a primitive" in 10 notebooks (recsys, recurrent, minibatch-sgd) | 🔧 Patch applied, rebuild in flight | `.investigations/d2l_cat1_atfork.patch`: clear oneDNN primitive cache + disable DNNL in DataLoader worker children via `pthread_atfork`. |

## Book-side fixes (no wheel change needed)

| Cat | What | Real root cause |
|---|---|---|
| 3 | `sentiment-analysis-rnn` "OOM" after 677s | `init.Xavier()` fails on flat 1D `_i2h_weight_initializer` of `rnn.LSTM`. Switch to `init.Normal(0.01)` or use selective init. Actual GPU footprint stable at 756 MiB. |
| 4 | `ssd.ipynb` IndexError 1860 vs size-1 | d2l `assign_anchor_to_bbox` divides by `num_gt_boxes=1` improperly. nonzero/argmax semantics in MXNet 2.0 differ from 1.x; book code needs update. |
| 5 | `neumf` >60min timeout | Book's `evaluate_ranking()` builds 943 `gluon.data.DataLoader` per eval × every-epoch. Fix: `eval_step=5` or single-DataLoader rewrite. Training itself fits in 5.4 min. |

## Repro scripts
- `.investigations/d2l_cat1_repro.py` — primitive failure (real wheel bug)
- `.investigations/d2l_cat2_repro.py` — DeadKernel regression guard (now passing)
- `.investigations/d2l_cat3_oom.py` — confirms no OOM
- `.investigations/d2l_cat5_hang.py` — confirms training is fast; eval is slow
