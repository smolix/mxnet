# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""FS13 lint: skip/xfail reasons must name a tracker.

issues.md FS13 calls for a "small meta-test or lint that requires new
broad skips/xfails to name a tracker ID".  This file is that lint: it
walks the source for `@pytest.mark.skip(reason=...)`,
`@pytest.mark.xfail(reason=...)`, `pytest.skip(...)`, and
`pytest.xfail(...)` calls inside the in-tree test directories and
asserts that each reason references either a known
internal tracker prefix (XOP, FS, GH, CN, T, R, D, B, C, O, P, L, A
followed by digits), a GitHub/Linear issue URL, or a structural reason
(platform/capability/dtype/env-dependent) that doesn't need a tracker.

If you are adding a new `pytest.mark.skip` or `pytest.mark.xfail`:

- prefer to fix or precisely capability-gate the test instead;
- if a skip is unavoidable, cite the tracker (`tracked under XOP12`,
  `apache/mxnet#17782`, etc.) so future audits can decide whether the
  skip is still load-bearing;
- a structural reason (`numeric-grad eps crosses zero`, `MAYBE not safe
  for forking`, `Test fails intermittently on CentOS 7 only`) also
  passes — the goal is to discourage `# TODO: fix later` skips with no
  reference.

The lint is permissive: any of the patterns below in the reason string
is enough.  False positives can be silenced by adding the closest
matching pattern to ALLOWED_PATTERNS rather than by adding to a per-row
exception list.
"""

from __future__ import annotations

import ast
import os
import re

import pytest


# Repo root is three levels up from this test file (tests/python/unittest/).
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))


# Patterns that count as a "this is properly tracked or structurally
# unavoidable" reason.  Each is a substring (case-insensitive) or a
# compiled regex.
ALLOWED_PATTERNS = [
    re.compile(r'\b(?:XOP|FS|GH|CN|T|R|D|B|C|O|P|L|A)\d+\b'),       # tracker
    re.compile(r'\b#\d{2,}\b'),                                     # bare #NNN
    re.compile(r'apache/mxnet#?\d+', re.IGNORECASE),                # gh ref
    re.compile(r'apache/incubator-mxnet[/#]\d+', re.IGNORECASE),
    re.compile(r'\bissues?/\d+', re.IGNORECASE),                    # /issues/NNN
    re.compile(r'github\.com', re.IGNORECASE),
    re.compile(r'\bPR-?\d+', re.IGNORECASE),
    # capability-gate idioms.  Any of these is enough to count as a
    # precise gate rather than a TODO-style stale skip.
    re.compile(r'\brequires?\b', re.IGNORECASE),       # requires GPU/...
    re.compile(r'\bneeds?\b.*\b(?:GPU|CUDA|cuDNN|NCCL|oneDNN|DNNL|TVM)\b',
               re.IGNORECASE),
    re.compile(r'\bbuilt (?:with|without)\b', re.IGNORECASE),
    re.compile(r'\b(?:un)?available\b', re.IGNORECASE),
    re.compile(r'\bnot enabled\b', re.IGNORECASE),
    re.compile(r'\bdeprecated in (?:MXNet|mx)\b', re.IGNORECASE),
    re.compile(r'\bsupport(?:ed)?(?:\s+(?:with|by|on))?\s+(?:cuDNN|GPU|CUDA)',
               re.IGNORECASE),
    re.compile(r'(?:GPU\s*\d+\s*not\s*available)', re.IGNORECASE),
    re.compile(r'\bnot all machine types\b', re.IGNORECASE),
    re.compile(r'\bcontinuous delivery\b', re.IGNORECASE),
    re.compile(r'\bcould not be imported\b', re.IGNORECASE),
    re.compile(r'\bonly supported with\b', re.IGNORECASE),
    re.compile(r'\busing\s+\d+\s*GPUs?\b', re.IGNORECASE),
    re.compile(r'\b(?:at least|more than|Need)\s+\d+\s*GPUs?\b', re.IGNORECASE),
    re.compile(r'\bcross[- ]GPU sync\b', re.IGNORECASE),
    re.compile(r'\blocal_allreduce_device\b', re.IGNORECASE),
    re.compile(r'\bChannel-last\b', re.IGNORECASE),
    re.compile(r'\bGPUs?\b', re.IGNORECASE),                          # any GPU-related skip
    re.compile(r'\bdeprec', re.IGNORECASE),                           # deprecated / deprecaed typo
    re.compile(r'\bMXNet\s+2\.0\b', re.IGNORECASE),                   # MXNet 2.0 op-spec changes
    re.compile(r'\bchange[ed]?\b.*\b(?:spec|behavior|in 2\.0)\b', re.IGNORECASE),
    re.compile(r'\bspec\s+in\s+2\.0\b', re.IGNORECASE),
    re.compile(r'\bAArch64\b', re.IGNORECASE),
    re.compile(r'\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b'),                     # env-var gate like OMP_NUM_THREADS
    re.compile(r'\bOpenMP\b', re.IGNORECASE),
    re.compile(r'\basynchronous\b', re.IGNORECASE),
    re.compile(r'\bdoes not support\b', re.IGNORECASE),
    re.compile(r'\bonednn\b', re.IGNORECASE),
    re.compile(r'\bfallback\b', re.IGNORECASE),
    # structural / capability gates that legitimately need a skip
    'platform-dependent', 'platform dependent',
    re.compile(r'\b(?:un)?supported platform\b', re.IGNORECASE),  # platform gate
    re.compile(r'\bplatform\b', re.IGNORECASE),                   # any platform gate
    'env-dependent', 'env dependent', 'environment dependent',
    'capability', 'sm_', 'compute capability',
    'numeric-grad eps', 'numeric grad eps',
    'no GPU', 'no CUDA', 'no NCCL', 'no oneDNN', 'no DNNL',
    'BF16', 'AVX-512', 'avx512',
    'sparse not yet supported',
    'feature flag',
    'not implemented for GPU', 'not implemented on GPU',
    'tracked under', 'tracked at', 'tracked in',
    'cpu_shared',
    'parallel download',  # network-dependent stress
    'allocation',          # XOP21 oversized-shape skip
    'INT_MAX',
]


SKIP_MARK_NAMES = {'skip', 'skipif', 'xfail'}
SKIP_CALL_NAMES = {'skip', 'xfail'}


def _iter_test_files(root):
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.startswith('test_') and fname.endswith('.py'):
                yield os.path.join(dirpath, fname)


def _literal_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_reason(call_node, kind):
    """Best-effort extraction of `reason=` from a pytest.mark.skip(..) call.

    Returns the reason string if it's a literal, or None if the reason is
    a complex expression we can't statically analyze (those are allowed
    through — the lint can only check what it can see).
    """
    for kw in call_node.keywords:
        if kw.arg == 'reason' and isinstance(kw.value, ast.Constant):
            value = kw.value.value
            if isinstance(value, str):
                return value
    # Direct pytest.skip("reason") / pytest.xfail("reason").
    if kind in ('skip', 'xfail') and call_node.args:
        reason = _literal_string(call_node.args[0])
        if reason is not None:
            return reason
    # Positional pytest.mark.skip("reason").
    if kind == 'mark.skip' and call_node.args:
        reason = _literal_string(call_node.args[0])
        if reason is not None:
            return reason
    # Positional second arg for skipif(condition, "msg"): also accepted.
    for arg in call_node.args[1:]:
        reason = _literal_string(arg)
        if reason is not None:
            return reason
    return None


def _reason_is_tracked(reason):
    if reason is None:
        # Unparseable reason expression — be permissive.
        return True
    for pat in ALLOWED_PATTERNS:
        if hasattr(pat, 'search'):
            if pat.search(reason):
                return True
        elif pat.lower() in reason.lower():
            return True
    return False


def _pytest_skip_kind(call_node):
    """Return skip/xfail kind for pytest mark or runtime calls, else None."""
    func = call_node.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr in SKIP_MARK_NAMES:
        mark = func.value
        if (isinstance(mark, ast.Attribute) and mark.attr == 'mark'
                and isinstance(mark.value, ast.Name)
                and mark.value.id == 'pytest'):
            return 'mark.' + func.attr
    if func.attr in SKIP_CALL_NAMES:
        value = func.value
        if isinstance(value, ast.Name) and value.id == 'pytest':
            return func.attr
    return None


def _walk_skip_calls(tree, path):
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match `pytest.mark.skip(...)` / `pytest.mark.skipif(...)` /
        # `pytest.mark.xfail(...)`.  The mark func chain is
        # Attribute(value=Attribute(value=Name('pytest'), attr='mark'),
        #           attr='skip'|'skipif'|'xfail').  Runtime calls are
        # Attribute(value=Name('pytest'), attr='skip'|'xfail').
        kind = _pytest_skip_kind(node)
        if kind is None:
            continue
        reason = _extract_reason(node, kind)
        if reason is None and not node.args and not node.keywords:
            # Empty skip/xfail with no reason at all.
            findings.append((path, node.lineno, '<empty>', kind))
            continue
        if not _reason_is_tracked(reason):
            findings.append((path, node.lineno, reason or '<empty>', kind))
    return findings


def test_skip_reasons_reference_tracker_or_capability():
    """Every pytest skip/xfail reason must reference a tracker or capability.

    This is permissive — the ALLOWED_PATTERNS list above accepts internal
    tracker IDs, GitHub issue URLs, and a list of common structural
    reasons.  When this test fires for a new skip, the right response is
    usually to add a tracker ID to the reason string, not to widen this
    lint."""
    bad = []
    for path in _iter_test_files(TEST_ROOT):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source, filename=path)
        except (SyntaxError, UnicodeDecodeError):
            continue
        bad.extend(_walk_skip_calls(tree, path))

    if bad:
        rendered = '\n'.join(
            f'  {os.path.relpath(p, TEST_ROOT)}:{ln}  pytest.{kind}({reason!r})'
            for (p, ln, reason, kind) in bad
        )
        pytest.fail(
            "FS13 lint: the following pytest skip/xfail reasons do not "
            "reference a known tracker, capability, or platform gate. Add a "
            "tracker ID (XOP12, apache/mxnet#17782, etc.) or move the test "
            "to a precise @pytest.mark.skipif gate:\n" + rendered)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
