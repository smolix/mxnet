#!/usr/bin/env python3
#
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

"""Shared download verification helpers for repo-local dependency builders."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def _require_checksum() -> bool:
    """True when an un-pinned download must be treated as a hard error.

    Set MXNET_DEPS_REQUIRE_CHECKSUM=1 in release/CD so a missing pin fails the
    build instead of silently fetching an unverified archive.
    """
    return os.environ.get("MXNET_DEPS_REQUIRE_CHECKSUM", "").strip().lower() in (
        "1", "true", "yes", "on")


def _manifest_path() -> Path:
    return Path(__file__).with_name("download_checksums.json")


def _load_manifest() -> dict[str, str]:
    path = _manifest_path()
    data = json.loads(path.read_text())
    entries = data.get("archives", [])
    checksums: dict[str, str] = {}
    for entry in entries:
        url = entry["url"]
        checksum = entry["sha256"].lower()
        if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
            raise SystemExit(f"Invalid SHA256 checksum for {url} in {path}")
        if url in checksums:
            raise SystemExit(f"Duplicate download checksum entry for {url} in {path}")
        checksums[url] = checksum
    return checksums


def expected_sha256_for_url(url: str) -> str | None:
    return _load_manifest().get(url)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as in_file:
        for chunk in iter(lambda: in_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_sha256: str, source: str) -> None:
    expected = expected_sha256.lower()
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(
            f"SHA256 mismatch for {source}: expected {expected}, got {actual}. "
            f"Remove {path} and retry if the cached archive is stale."
        )
    print(f"Verified SHA256 for {path}", flush=True)


def verify_archive_if_pinned(path: Path, url: str) -> str | None:
    expected_sha256 = expected_sha256_for_url(url)
    if expected_sha256 is None:
        msg = (f"no pinned SHA256 checksum for {url}; archive cannot be "
               f"checksum-verified. Add it to {_manifest_path()}.")
        if _require_checksum():
            raise SystemExit(
                f"Error: {msg} (MXNET_DEPS_REQUIRE_CHECKSUM is set)")
        print(f"Warning: {msg}", file=sys.stderr, flush=True)
        return None
    verify_sha256(path, expected_sha256, url)
    return expected_sha256
