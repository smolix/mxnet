#!/usr/bin/env python3

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

"""Read-only release-staging provenance checks for MXNet wheels."""

import argparse
import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version


FEATURES = ("USE_CUDA", "USE_OPENCV")
TRUE_VALUES = {"1", "ON", "TRUE", "YES"}
FALSE_VALUES = {"0", "OFF", "FALSE", "NO"}
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ProvenanceError(RuntimeError):
    pass


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _git_output(repo_root, args):
    result = subprocess.run(
        ["git", "-C", str(repo_root)] + args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return result.stdout.strip()


def _version(value):
    try:
        return str(Version(value))
    except InvalidVersion as err:
        raise ProvenanceError("invalid PEP 440 version {!r}".format(value)) from err


def _bool_value(raw):
    value = raw.strip().upper()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    return None


def read_git_state(repo_root):
    tracked = _git_output(
        repo_root, ["status", "--porcelain", "--untracked-files=no"]
    ).splitlines()
    full = _git_output(
        repo_root, ["status", "--porcelain", "--untracked-files=normal"]
    ).splitlines()
    commit = _git_output(repo_root, ["rev-parse", "HEAD"])
    untracked = [line for line in full if line.startswith("?? ")]
    return {
        "commit": commit,
        "short_commit": commit[:12],
        "dirty": bool(tracked),
        "tracked_change_count": len(tracked),
        "tracked_changes": tracked,
        "untracked_count": len(untracked),
        "untracked_entries": untracked,
    }


def read_package_version(repo_root, package_version=None):
    env_version = os.environ.get("MXNET_PACKAGE_VERSION", "").strip()
    if package_version:
        raw_version, source = package_version.strip(), "argument"
    elif env_version:
        raw_version, source = env_version, "MXNET_PACKAGE_VERSION"
    else:
        libinfo = repo_root / "python" / "mxnet" / "libinfo.py"
        raw_version = runpy.run_path(str(libinfo))["__version__"]
        source = str(libinfo.relative_to(repo_root))
    return {"version": _version(raw_version), "source": source}


def read_cmake_features(repo_root, cmake_cache=None):
    cache = cmake_cache or repo_root / "build" / "CMakeCache.txt"
    report = {
        "cache_path": str(cache),
        "cache_found": cache.exists(),
    }
    for name in FEATURES:
        report[name] = {"enabled": None, "raw": None}
    if not cache.exists():
        return report

    for line in cache.read_text().splitlines():
        if line.startswith("//") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        key = key.split(":", 1)[0]
        if key in FEATURES:
            raw = raw.strip()
            report[key] = {"enabled": _bool_value(raw), "raw": raw}
    return report


def read_wheels(wheel_paths, package_version, package_name):
    expected_name = canonicalize_name(package_name)
    rows = []
    for wheel_path in wheel_paths:
        if not wheel_path.name.endswith(".whl"):
            raise ProvenanceError("{} is not a wheel".format(wheel_path))
        name, version, build, tags = parse_wheel_filename(wheel_path.name)
        rows.append({
            "path": str(wheel_path),
            "filename": wheel_path.name,
            "distribution": str(name),
            "version": str(version),
            "build_tag": build,
            "tags": sorted(str(tag) for tag in tags),
            "exists": wheel_path.exists(),
            "distribution_matches_package": canonicalize_name(str(name)) == expected_name,
            "version_matches_package": _version(str(version)) == package_version,
        })
    return rows


def collect_provenance(
    repo_root=None,
    wheel_paths=None,
    package_version=None,
    cmake_cache=None,
    package_name="mxnet",
):
    root = repo_root or _repo_root()
    package = read_package_version(root, package_version)
    return {
        "repo_root": str(root),
        "expected_package_name": package_name,
        "git": read_git_state(root),
        "package": package,
        "features": read_cmake_features(root, cmake_cache),
        "wheels": read_wheels(wheel_paths or [], package["version"], package_name),
    }


def _expected_flag(value):
    if value is None:
        return None
    return value == "on"


def validate_provenance(
    report,
    allow_dirty=False,
    strict_untracked=False,
    allow_missing_cache=False,
    expect_commit=None,
    expect_cuda=None,
    expect_opencv=None,
):
    errors = []
    git = report["git"]
    if not COMMIT_RE.match(git["commit"]):
        errors.append("git commit is not a 40-character hexadecimal SHA")
    if expect_commit and not git["commit"].startswith(expect_commit.lower()):
        errors.append(
            "git commit {} does not match expected {}".format(
                git["short_commit"], expect_commit
            )
        )
    if git["dirty"] and not allow_dirty:
        errors.append(
            "tracked working tree is dirty ({} changed path{})".format(
                git["tracked_change_count"],
                "" if git["tracked_change_count"] == 1 else "s",
            )
        )
    if strict_untracked and git["untracked_count"]:
        errors.append("working tree has {} untracked path{}".format(
            git["untracked_count"], "" if git["untracked_count"] == 1 else "s"))

    features = report["features"]
    if not features["cache_found"] and not allow_missing_cache:
        errors.append("CMake cache not found at {}".format(features["cache_path"]))
    for name in FEATURES:
        if features[name]["enabled"] is None and not allow_missing_cache:
            errors.append("{} was not found as a boolean CMake feature".format(name))

    for name, expected in {
        "USE_CUDA": _expected_flag(expect_cuda),
        "USE_OPENCV": _expected_flag(expect_opencv),
    }.items():
        if expected is not None and features[name]["enabled"] is not expected:
            errors.append("{} expected {}, found {}".format(
                name, "ON" if expected else "OFF", features[name]["raw"] or "unknown"))

    for wheel in report["wheels"]:
        if not wheel["exists"]:
            errors.append("wheel file does not exist: {}".format(wheel["path"]))
        if not wheel["distribution_matches_package"]:
            errors.append("{} distribution {} does not match expected package {}".format(
                wheel["filename"], wheel["distribution"], report["expected_package_name"]))
        if not wheel["version_matches_package"]:
            errors.append("{} version {} does not match package version {}".format(
                wheel["filename"], wheel["version"], report["package"]["version"]))
    return errors


def _flag_text(value):
    return "ON" if value is True else "OFF" if value is False else "unknown"


def format_text_report(report, errors):
    git, package, features = report["git"], report["package"], report["features"]
    lines = [
        "Release provenance:",
        "  commit: {}".format(git["commit"]),
        "  tracked dirty: {} ({} changed path{})".format(
            "yes" if git["dirty"] else "no",
            git["tracked_change_count"],
            "" if git["tracked_change_count"] == 1 else "s",
        ),
        "  untracked paths: {}".format(git["untracked_count"]),
        "  package version: {} ({})".format(package["version"], package["source"]),
        "  CMake cache: {}{}".format(
            features["cache_path"], "" if features["cache_found"] else " (missing)"),
    ]
    for name in FEATURES:
        raw = "" if features[name]["raw"] is None else " [{}]".format(features[name]["raw"])
        lines.append("  {}: {}{}".format(name, _flag_text(features[name]["enabled"]), raw))

    lines.append("  wheels:" if report["wheels"] else "  wheels: none")
    for wheel in report["wheels"]:
        lines.append(
            "    {}: distribution={} version={} version_match={} exists={}".format(
                wheel["path"], wheel["distribution"], wheel["version"],
                "yes" if wheel["version_matches_package"] else "no",
                "yes" if wheel["exists"] else "no"))

    lines.append("Validation: {}".format("failed" if errors else "ok"))
    lines.extend("  - {}".format(error) for error in errors)
    return "\n".join(lines)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="*", type=Path)
    parser.add_argument("--wheel", action="append", type=Path, default=[])
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument("--cmake-cache", type=Path)
    parser.add_argument("--package-version")
    parser.add_argument("--package-name", default="mxnet")
    parser.add_argument("--expect-commit")
    parser.add_argument("--expect-cuda", choices=("on", "off"))
    parser.add_argument("--expect-opencv", choices=("on", "off"))
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--strict-untracked", action="store_true")
    parser.add_argument("--allow-missing-cache", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    try:
        report = collect_provenance(
            repo_root=args.repo_root,
            wheel_paths=list(args.wheel) + list(args.wheels),
            package_version=args.package_version,
            cmake_cache=args.cmake_cache,
            package_name=args.package_name,
        )
        errors = validate_provenance(
            report,
            allow_dirty=args.allow_dirty,
            strict_untracked=args.strict_untracked,
            allow_missing_cache=args.allow_missing_cache,
            expect_commit=args.expect_commit,
            expect_cuda=args.expect_cuda,
            expect_opencv=args.expect_opencv,
        )
    except (OSError, ProvenanceError, subprocess.CalledProcessError) as err:
        print("release provenance check failed: {}".format(err), file=sys.stderr)
        return 2

    if args.as_json:
        report["validation_errors"] = errors
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_text_report(report, errors))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
