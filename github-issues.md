# GitHub Issues Delta

Updated: 2026-05-20
Repository reviewed: `apache/mxnet`
Review status: GitHub issue/PR crawl complete for the Apple Silicon handoff PR.
Current branch: `followup/full-sweep-macos-wheel`

This file tracks GitHub findings that are not already represented in
`issues.md`. It intentionally excludes the known Apple Silicon oneDNN/Xbyak
work, DataLoader failures, NumPy/SciPy API drift, OpenCV/image dependency work,
CUDA/cuDNN build drift, lifecycle/deadlock findings, DLPack, ONNX opset 18
reductions, CI/release packaging, and broad performance buckets already listed
there.

## High Priority

| ID | Source | Area | New issue | Suggested next action |
|---|---|---|---|---|
| G1 | [#21155](https://github.com/apache/mxnet/pull/21155), [#21159](https://github.com/apache/mxnet/pull/21159), [#19315](https://github.com/apache/mxnet/pull/19315), [#21164](https://github.com/apache/mxnet/pull/21164), [#21073](https://github.com/apache/mxnet/pull/21073), [#21071](https://github.com/apache/mxnet/pull/21071) | Security/tooling | The archived repo still has open security hardening work in docs/tooling: unsafe tar extraction, command injection in notebook conversion, insecure download links, and stale docs/CD dependencies. | Audit local scripts before running them on untrusted inputs; prefer targeted patches over accepting old PRs wholesale. |
| G2 | [#21141](https://github.com/apache/mxnet/pull/21141), [#20297](https://github.com/apache/mxnet/pull/20297), [#20491](https://github.com/apache/mxnet/pull/20491), [#20316](https://github.com/apache/mxnet/pull/20316), [#18583](https://github.com/apache/mxnet/pull/18583) | C / C++ inference API | C Predict and C++ inference/subgraph APIs have unresolved correctness gaps: duplicate input names can trigger uninitialized reads, executor binding needs locking, deleted subgraph nodes need handling, and C++ shape/backend APIs lag Python. | Treat as a focused inference-API audit. Add C API/C++ API regression tests before changing behavior. |
| G3 | [#16686](https://github.com/apache/mxnet/issues/16686), [#21091](https://github.com/apache/mxnet/pull/21091), [#19275](https://github.com/apache/mxnet/pull/19275), [#19076](https://github.com/apache/mxnet/pull/19076), [#18027](https://github.com/apache/mxnet/pull/18027), [#17209](https://github.com/apache/mxnet/pull/17209) | Autograd/Gluon semantics | There are unresolved gradient and Gluon export semantics issues outside the lifecycle fixes: `grad_req='add'` numerical behavior, `attach_grad` on non-leaf HybridizedBlock values, `autograd.grad`, `block.export`, mixed binary backward handling, and Gluon `Variable` dtype propagation. | Prioritize CPU-reproducible autograd tests first; defer any GPU-only parity checks to CUDA CI. |
| G4 | [#21204](https://github.com/apache/mxnet/pull/21204), [#15702](https://github.com/apache/mxnet/pull/15702) | Resource hygiene | Logger/file-handle lifetime and repeated `Extract()` memory leak reports are separate from the engine/DataLoader lifecycle issues already tracked. | Add leak/file-handle regression tests around the affected Python/C++ utility paths. |

Status for this PR: none of the GitHub-delta items above were fixed directly in
`followup/full-sweep-macos-wheel`. The branch focused on the Apple Silicon CPU
port, optimized macOS arm64 build, ONNX-free packaging, and the macOS
`cpu_shared` DataLoader fallback captured in `issues.md`.

## Medium Priority

| ID | Source | Area | New issue | Suggested next action |
|---|---|---|---|---|
| G5 | [#20761](https://github.com/apache/mxnet/pull/20761), [#19844](https://github.com/apache/mxnet/pull/19844), [#18792](https://github.com/apache/mxnet/pull/18792), [#18268](https://github.com/apache/mxnet/pull/18268), [#17975](https://github.com/apache/mxnet/pull/17975), [#14582](https://github.com/apache/mxnet/pull/14582) | Operator correctness | Several operator-specific fixes remain open: `slice_axis` in modulated deformable convolution, `linspace`, fp16 `argsort`, GroupNorm migration/correctness, NumPy `mean`/`sum` with infinities, and randomized ReLU. | Add focused CPU tests where possible; mark CUDA-required variants separately. |
| G6 | [#18366](https://github.com/apache/mxnet/pull/18366), [#18081](https://github.com/apache/mxnet/pull/18081), [#15672](https://github.com/apache/mxnet/pull/15672), [#21212](https://github.com/apache/mxnet/pull/21212) | Data and datasets | Dataset/RecordIO semantics have old unresolved work: multithreaded RecordIO file-id switching, ImageFolderDataset class assignment, transform handling, and missing MultiboxPrior coverage. | Fold into the current data/image test sweep after the full unittest run stabilizes. |
| G7 | [#21217](https://github.com/apache/mxnet/pull/21217) | Distributed training | Horovod KVStore is missing a barrier API. This is distinct from the ps-lite/NCCL distributed backlog in `issues.md`. | Defer implementation unless Horovod support is in scope; otherwise document it as unsupported. |
| G8 | [#21093](https://github.com/apache/mxnet/pull/21093), [#20412](https://github.com/apache/mxnet/pull/20412), [#20538](https://github.com/apache/mxnet/pull/20538) | Linux CPU performance/platform | Open PRs propose FlexiBLAS detection, transparent huge page allocator support, and `parallel_for` grain-size tuning. These are not Apple-specific but may matter for x86/Linux CPU builds. | Revisit after Apple Silicon correctness; each needs Linux benchmarking and CI. |
| G9 | [#20569](https://github.com/apache/mxnet/pull/20569), [#19849](https://github.com/apache/mxnet/pull/19849), [#16386](https://github.com/apache/mxnet/pull/16386) | TensorRT | TensorRT upgrade/build integration work is stale and separate from the CUDA core backlog already tracked. | Defer until CUDA CI exists; do not change locally without Linux/CUDA validation. |

## Feature Additions From GitHub

| ID | Source | Area | New issue | Suggested next action |
|---|---|---|---|---|
| G10 | [#20104](https://github.com/apache/mxnet/pull/20104), [#16209](https://github.com/apache/mxnet/pull/16209), [#18928](https://github.com/apache/mxnet/pull/18928) | Dtype/runtime coverage | The open PR queue includes larger runtime-surface additions: int16 tensors/operators, complex dtype plus NumPy FFT, and fp16 input support for GPU `quantize_v2`. These are compatibility-expanding changes, not Apple Silicon bring-up blockers. | Keep out of the current port unless a failing test depends on them. Each needs a dtype support matrix across ndarray, NumPy namespace, serialization, C API, and backend dispatch. |
| G11 | [#16766](https://github.com/apache/mxnet/pull/16766), [#16627](https://github.com/apache/mxnet/pull/16627), [#16639](https://github.com/apache/mxnet/pull/16639), [#17264](https://github.com/apache/mxnet/pull/17264) | New Python/C++ operators | Proposed operator additions include `unfold`, segment operators, `contrib.index_array` improvements, and an image `CenterCrop` op. These would add user-visible API surface and require operator registration, shape/type inference, executor coverage, and Python docs/tests. | Treat as separate feature projects after release stabilization; do not merge stale operator PRs without refreshing tests against current NNVM/operator conventions. |
| G12 | [#17385](https://github.com/apache/mxnet/pull/17385), [#17926](https://github.com/apache/mxnet/pull/17926), [#17905](https://github.com/apache/mxnet/pull/17905), [#15855](https://github.com/apache/mxnet/pull/15855), [#15857](https://github.com/apache/mxnet/pull/15857), [#16025](https://github.com/apache/mxnet/pull/16025), [#16009](https://github.com/apache/mxnet/pull/16009) | NumPy namespace feature parity | Several open PRs target NumPy-compatible APIs: `random.geometric`, `copyto`, `unpackbits`, `cumprod`, `logaddexp`, bit shifts, and bitwise operations. Some older candidates were TVM-based and need special caution because TVM is out of current scope. | Reassess only after NumPy 1.x/2.x compatibility is stable. Prefer small, independently tested native implementations over reviving TVM-dependent patches. |
| G13 | [#20762](https://github.com/apache/mxnet/pull/20762), [#18325](https://github.com/apache/mxnet/pull/18325), [#18285](https://github.com/apache/mxnet/pull/18285), [#17129](https://github.com/apache/mxnet/pull/17129), [#16955](https://github.com/apache/mxnet/pull/16955) | Training/data API extensions | Feature PRs also cover RAdam, SGD/momentum behavior changes, Estimator iterable inputs, and dataset flattening. The optimizer items are behavior-sensitive because old checkpoints or training recipes may rely on current semantics. | Split additions from behavior changes. New APIs can be considered later; optimizer semantic changes need migration notes and numerical regression tests. |
| G14 | [#20043](https://github.com/apache/mxnet/pull/20043), [#13715](https://github.com/apache/mxnet/pull/13715), [#15678](https://github.com/apache/mxnet/pull/15678), [#18583](https://github.com/apache/mxnet/pull/18583), [#20491](https://github.com/apache/mxnet/pull/20491) | C/C++ API extensions | The C/C++ backlog includes general C API enhancement work, a C++ Predictor class, contrib-op exposure in the C++ package, partial shape inference, and C++ `OptimizeForBackend`. These overlap with G2 where correctness is involved, but the feature aspect is broader API modernization. | Fold only the correctness pieces into near-term work. Broader C++ API expansion should wait for ABI/API policy decisions for the fork. |

## Recommended Carry-Forward Order

After the Apple Silicon PR is merged and the work moves to a Linux/CUDA host:

1. Start with `issues.md` deferred CUDA/Linux validation rather than the feature
   additions here.
2. Use G2 and G3 as the first GitHub-derived correctness audits because they are
   Python/C++ core behavior and likely CPU-reproducible.
3. Keep G10-G14 out of the stabilization branch unless a failing test requires
   one of those features.
4. Revisit G8 only after Linux CPU benchmark infrastructure exists; these PRs
   can easily change performance without clear correctness signals.

## Notes

- Open PR metadata was fetched from GitHub on 2026-05-20.
- Open issue triage was filtered against `issues.md`; known categories remain
  there rather than being duplicated here.
- Non-Python/C++ language-binding PRs are intentionally ignored for this fork
  triage.
- The current macOS arm64 wheel branch does not attempt to merge stale upstream
  PRs wholesale; each entry here should be treated as a prompt for a fresh,
  tested patch against this fork.
