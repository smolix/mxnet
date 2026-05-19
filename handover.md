# Handover — 2026-05-19

Session continuation notes for the MXNet Blackwell port at `smolix/mxnet`.
Picks up from the d2l-neu wheel `.2` sweep findings; ends at PR #24 merged.

---

## State of the tree at handover

- **Master tip**: `463ff7485` (PR #24 merged at 2026-05-19 02:28:59Z).
- **Current wheel published on GitHub Releases**: `v2.0.0+cu13.bw.20260518.2`
  (released 2026-05-18 20:30:11Z). **Note**: that wheel pre-dates PRs #23 and
  #24 — see "Next wheel cut" below.
- **Open PRs**: none.
- **Local-only branches with unmerged work**: `fix/fu6-qat-subgraph-backward`
  (commits `74b62dc27` + `a24610a02`) — see FU-6 below.

---

## What landed during this session (chronologically)

1. **PR #22** (docs/fu11-d2l-rnn-triage, merged 22:42:41Z) —
   `FOLLOW_UPS.md` updated with FU-11/12/13/14 + FU-4 status revision.
   Adds `.investigations/fu11_d2l_rnn_stack_repro.py`.
2. **PR #23** (fix/fu11-onednn-concat-large-N, merged 23:27:04Z) —
   **FU-11 fix**: gate `SupportDNNLStack`/`SupportDNNLConcat` at 256 inputs.
   Built locally, verified end-to-end: gru.ipynb runs 1 epoch in 8.7 s
   (was crashing in < 20 s pre-fix). 19/19 regression tests pass.
   New test file: `tests/python/dnnl/test_fu11_large_stack_concat.py`.
3. **PR #24** (fix/fu1-refinement, merged 02:28:59Z) — **FU-1 refinement**:
   runtime gate in `dnnl_convolution.cc::GetConvFwdImpl` skips the
   `eltwise_relu` post-op for the buggy AVX2 int8 1×1 conv + ic<8 combo;
   `dnnl_conv.cc` folds `output_scale` into `requantize_scales[]` so the
   u8 dst is correctly scaled (u8 saturation provides the relu semantics
   without the post-op). Full conv subgraph: 428/428 pass.

### Branch cleanup performed

Remote branches deleted (all fully merged to master):
`docs/fu6-status-update`, `test/fu4-and-deconv-tf32-regressions`,
`fix/fu3-refinement`, `fix/fu1-fu3-fu4-fu5`, `fix/ci-failures`,
`sweep/master-plus-prs`, `fix/qat-ste-quantize-v2`. Local merged copies of
the above plus `fix/build-cpu-timeout`, `fix/d2l-naive-bayes-prod`,
`fix/test-pollution-reset-np{,−more}` also pruned.

**Not deleted (kept on purpose)**: release branches (anchor tags),
`fix/gpu-shadow-test-fixtures` (ahead 1), `fix/license-check-fork-artifacts`
(ahead 1), `blackwell-{cuda13,port}`, `onednn-v3-port`,
`feature/post-release-fixes-2026-05-17`,
`release/slim-wheel-pip-deps`. This handover commit additionally deletes
`fix/fu11-onednn-concat-large-N` (ahead=0, merged via PR #23).

---

## Where each follow-up stands

(Cross-referenced with `FOLLOW_UPS.md` and `issues.md`.)

| ID    | Status | Notes |
|-------|--------|-------|
| FU-1  | **CLOSED** (PR #18 gate + PR #24 runtime refinement) | Closes issues.md #4 in this fork pending upstream oneDNN fix. |
| FU-2  | OPEN, design decision needed | issues.md #50. Either widen quantize_v2/dequantize/dnnl paths to accept fp16, or document AMP fp16+quantize as unsupported. |
| FU-3  | **CLOSED** (PR #18 brg_conv gate + PR #19 test fix). |  |
| FU-4  | **CLOSED** Path B (PR #18 + #20 regression tests). Was insufficient — FU-11 split out. |  |
| FU-5  | **PARTIAL** (PR-A/B/C landed). PR-D (INT8) + PR-E (default-on) **deferred**. |  |
| FU-6  | **IMPLEMENTED LOCAL, NOT PUSHED** | See "Critical next action" below. |
| FU-7  | **DEFERRED-BY-CHOICE** | Multi-arch fatbin = one cmake flag. issues.md #31. |
| FU-8  | **INSTRUMENTED ONLY** | Engine deadlock A6/A7; needs ARM (aarch64) repro. |
| FU-9  | **CLOSED FOR THE WHEEL** | d2l book bugs — reported separately. |
| FU-10 | **PARTIAL** | PR/nightly CI workflows landed; no self-hosted GPU runner. issues.md #32. |
| FU-11 | **CLOSED** (PR #23). 7 d2l RNN notebooks unblocked. |  |
| FU-12 | OPEN, environmental | `transformer.ipynb` DeadKernelError; minimal repro does **not** reproduce in isolation. |
| FU-13 | OPEN, not reproducible standalone | `np.nonzero` IndexError in d2l `assign_anchor_to_bbox` context. |
| FU-14 | OPEN, low priority | `sentiment-analysis-rnn.ipynb` OOM. May be a real budget issue. |

---

## Critical next action: push + merge FU-6

The FU-6 implementation is **complete and built**, but lives only on a local
branch. To pick up:

```bash
cd /workspace/mxnet
git checkout fix/fu6-qat-subgraph-backward
git log --oneline -3
# Expected: a24610a02 + 74b62dc27 + c1ba62da9
git push -u origin fix/fu6-qat-subgraph-backward
gh pr create -R smolix/mxnet --title "FU-6: QAT backward through fused _sg_onednn_{fc,conv}" \
  --body-file <(printf 'Closes issues.md #5 Step 3. See commit messages on 74b62dc27/a24610a02 for full design. Gated behind MXNET_QAT_SUBGRAPH_BACKWARD=1; legacy MakeZeroGradNodes behaviour preserved when env var is unset.')
```

Built `libmxnet.so` is at `/workspace/mxnet/build/libmxnet.so` (830 MB,
2026-05-19 00:20). Regression test file is
`tests/python/dnnl/test_fu6_qat_subgraph_backward.py`. **Tests have NOT been
run on the fresh binary** — they should be before merging. The FU-6 agent
reported "13 PASS, 4 XFAIL" but that was against an earlier WIP binary.

---

## Next wheel cut (`.3`)

When ready, the next published wheel should bump to
`2.0.0+cu13.bw.20260519.3` (or similar) and include:

- PR #23 (FU-11 stack/concat gate) — already on master
- PR #24 (FU-1 refinement) — already on master
- FU-6 (after the merge above)

Bump `python/mxnet/libinfo.py::__version__` and follow the same
release-branch pattern used for `.1` / `.2`. The CMake macros
`MXNET_BRANCH`/`MXNET_COMMIT_HASH` are baked into `operator_tune.cc`
only — be aware they trigger full `operator_tune.cc` recompiles
(~25–30 min) on every branch flip.

---

## Notes on the working tree state at handover

- `/workspace/mxnet` is on this handover branch right now. The previous
  in-flight builds completed; no `ninja` or `cc1plus` processes alive.
- `/tmp/fu11-fix/` is a worktree with its own build dir at
  `/tmp/fu11-fix/build/libmxnet.so` (516 MB, 2026-05-19 00:18). The
  `3rdparty/` is a symlink to `/workspace/mxnet/3rdparty/`. Safe to delete
  once you have your own setup.
- Many `worktree-agent-*` worktrees exist under `.claude/worktrees/`.
  They are locked and managed by the agent runtime; don't `git worktree
  remove` them by hand.
- d2l-neu venv with the published `.2` wheel is at
  `/workspace/d2l-neu/.venv-mxnet/`. To test the master-tip binary
  against it, set `MXNET_LIBRARY_PATH=/workspace/mxnet/build/libmxnet.so`.

---

## Surprises / non-obvious findings

1. **The FU-11 root cause** turned out to be a deterministic oneDNN
   defect at exactly 513 sources — the failing call site is
   `cvt_primitive_args` reporting `bad number of inputs (expected 514
   got 513)`. The mxnet fix declines the DNNL path for wide
   stack/concat; the underlying oneDNN bug remains. If oneDNN ships a
   fix in a later release, this gate can be loosened or removed.

2. **FU-11 ≠ FU-4**. They share the surface symptom
   (`MXNetError: could not execute a primitive`) and originally
   appeared as the same cluster in `d2l-neu/mxnet-errors.md`. But the
   failing path for the d2l RNN notebooks uses `num_workers=0`
   (no fork involved) — FU-4 Path B's fork-safe engine reset is
   unrelated to the cluster. They were split during the 21:28 d2l
   sweep re-triage.

3. **The d2l "8-line minimal repros"** in `mxnet-errors.md` for FU-12
   (transformer DeadKernel) and FU-13 (nonzero IndexError) **do not
   reproduce in isolation** on this machine. Both bugs require either
   concurrent GPU pressure (FU-12) or the specific
   `assign_anchor_to_bbox` call context (FU-13). The d2l reporter
   may want to capture a richer minimal repro before these can be
   diagnosed in C++.

4. **Co-authored-by trailer policy changed mid-session**. The
   "Co-Authored-By: Claude Opus 4.7 (1M context)" trailer that earlier
   commits used was rejected by the auto-mode classifier as
   "fabricated co-author attribution". PRs #22/#23 and this handover
   commit do **not** include it. The git history is mixed; if you want
   to normalise, a rebase + author-rewrite is the only path (don't
   force-push master without coordinating).
