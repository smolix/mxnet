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
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version


FEATURES = ("USE_CUDA", "USE_CUDNN", "USE_NCCL", "USE_ONEDNN", "USE_OPENCV", "USE_OPENMP")
# Basenames of the OpenMP runtime libraries libmxnet may link: the LLVM/Intel
# runtime (libomp / libiomp5) and the GCC runtime (libgomp).  On macOS the
# runtime is vendored into the wheel (no system libomp); on Linux it is host-
# provided (part of the GCC runtime), exactly as libc/libstdc++ are.
OPENMP_RUNTIME_RE = re.compile(r"^lib(omp|iomp5|gomp)\b")
TRUE_VALUES = {"1", "ON", "TRUE", "YES"}
FALSE_VALUES = {"0", "OFF", "FALSE", "NO"}
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
BUILD_COMMIT_RE = re.compile(
    r"MXNET_COMMIT_HASH\s*=\s*(?:\\*[\"'])?([0-9a-f]{40}|Unavailable)"
)


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


def _build_dir_from_cache(repo_root, cmake_cache=None):
    cache = cmake_cache or repo_root / "build" / "CMakeCache.txt"
    if cache.name == "CMakeCache.txt":
        return cache.parent
    return repo_root / "build"


def _extract_build_commit_hashes(path):
    hashes = []
    try:
        with path.open("r", errors="ignore") as handle:
            for line in handle:
                hashes.extend(BUILD_COMMIT_RE.findall(line))
    except OSError:
        return []
    return sorted(set(hashes))


def read_build_metadata(repo_root, cmake_cache=None):
    build_dir = _build_dir_from_cache(repo_root, cmake_cache)
    report = {
        "build_dir": str(build_dir),
        "metadata_found": False,
        "commit_hashes": [],
        "sources": [],
    }
    commit_hashes = set()
    for path in (build_dir / "build.ninja", build_dir / "compile_commands.json"):
        source = {
            "path": str(path),
            "exists": path.exists(),
            "commit_hashes": [],
        }
        if path.exists():
            source["commit_hashes"] = _extract_build_commit_hashes(path)
            commit_hashes.update(source["commit_hashes"])
        report["sources"].append(source)

    report["commit_hashes"] = sorted(commit_hashes)
    report["metadata_found"] = bool(report["commit_hashes"])
    return report


def _readelf_dynamic_section(library_path):
    result = subprocess.run(
        ["readelf", "-d", str(library_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return result.stdout


def _parse_dynamic_section(dynamic_section):
    needed = []
    runpath = None
    for line in dynamic_section.splitlines():
        needed_match = re.search(r"Shared library: \[([^\]]+)\]", line)
        if needed_match:
            needed.append(needed_match.group(1))
            continue
        runpath_match = re.search(r"Library (?:rpath|runpath): \[([^\]]*)\]", line)
        if runpath_match:
            runpath = runpath_match.group(1)
    return {"needed": needed, "runpath": runpath}


def _otool(args):
    result = subprocess.run(
        ["otool"] + args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return result.stdout


def _macho_install_id(library_path):
    """Return the Mach-O install id (LC_ID_DYLIB), or '' if there is none."""
    lines = _otool(["-D", str(library_path)]).splitlines()
    # `otool -D` prints "<path>:" then the id on the next line.
    return lines[1].strip() if len(lines) > 1 else ""


def _macho_dependencies(library_path):
    """Return the Mach-O LC_LOAD_DYLIB paths, excluding the library's own id."""
    self_id = _macho_install_id(library_path)
    deps = []
    # `otool -L` prints "<path>:" then the id then one dependency per line as
    # "<path> (compatibility ...)".
    for line in _otool(["-L", str(library_path)]).splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        dep = line.split(" (")[0].strip()
        if not dep or dep == self_id:
            continue
        deps.append(dep)
    return deps


def _macho_rpaths(library_path):
    """Return the LC_RPATH paths of a Mach-O binary."""
    rpaths = []
    lines = _otool(["-l", str(library_path)]).splitlines()
    for index, line in enumerate(lines):
        if "cmd LC_RPATH" not in line:
            continue
        for follow in lines[index + 1:index + 4]:
            match = re.search(r"\bpath (.+?) \(offset", follow)
            if match:
                rpaths.append(match.group(1).strip())
                break
    return rpaths


def _classify_onnx_requires(metadata_text):
    """Classify how a wheel's METADATA depends on onnx.

    Returns (hard_required, extra_required):
      * hard_required  — a `Requires-Dist: onnx ...` line with NO environment
        marker, i.e. onnx is an unconditional install dependency (the CUDA wheel).
      * extra_required — an onnx requirement gated on `extra == "onnx"`, i.e. the
        optional `[onnx]` extra (present on every wheel).
    """
    hard = extra = False
    for line in metadata_text.splitlines():
        if not line.lower().startswith("requires-dist:"):
            continue
        spec = line.split(":", 1)[1].strip()
        try:
            requirement = Requirement(spec)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) != "onnx":
            continue
        if requirement.marker is not None and "extra" in str(requirement.marker):
            extra = True
        else:
            hard = True
    return hard, extra


def inspect_wheel_payload(wheel_path):
    payload = {
        "inspected": False,
        "error": None,
        "format": None,
        "has_libmxnet": False,
        "needed": [],
        "cudnn_needed": [],
        "nccl_needed": [],
        "opencv_needed": [],
        "opencv_bundled": [],
        "opencv_bundled_sonames": [],
        "openmp_needed": [],
        "openmp_bundled": [],
        "openmp_runtime_loader_relative": False,
        "onnx_pkg_files": [],
        "onnx_hard_require": False,
        "onnx_extra_require": False,
        "metadata_found": False,
        "runpath": None,
        "runpath_has_origin_lib": False,
        "runpath_has_nvidia_cudnn": False,
        "runpath_has_nvidia_nccl": False,
    }
    if not wheel_path.exists():
        return payload

    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            names = wheel.namelist()
            payload["opencv_bundled"] = sorted(
                name for name in names
                if name.startswith("mxnet/lib/libopencv_")
            )
            payload["opencv_bundled_sonames"] = sorted(
                {Path(name).name for name in payload["opencv_bundled"]}
            )
            # The OpenMP runtime is vendored into mxnet/lib/ on macOS (no system
            # libomp); on Linux it is host-provided and not bundled.
            payload["openmp_bundled"] = sorted(
                name for name in names
                if name.startswith("mxnet/lib/")
                and OPENMP_RUNTIME_RE.match(Path(name).name)
            )
            # ONNX integration (mxnet.onnx) is pure-Python and bundled in the
            # wheel (OI-27); the *dependency policy* (hard dep vs optional [onnx]
            # extra) is read from the wheel METADATA.
            payload["onnx_pkg_files"] = sorted(
                name for name in names if name.startswith("mxnet/onnx/")
            )
            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            if metadata_names:
                payload["metadata_found"] = True
                metadata_text = wheel.read(metadata_names[0]).decode(
                    "utf-8", "ignore")
                payload["onnx_hard_require"], payload["onnx_extra_require"] = (
                    _classify_onnx_requires(metadata_text))
            # Linux wheels ship an ELF libmxnet.so; macOS wheels ship a Mach-O
            # libmxnet.dylib.  Inspect whichever the wheel actually contains.
            if "mxnet/libmxnet.so" in names:
                libmxnet_name, payload["format"], local_name = (
                    "mxnet/libmxnet.so", "elf", "libmxnet.so")
            elif "mxnet/libmxnet.dylib" in names:
                libmxnet_name, payload["format"], local_name = (
                    "mxnet/libmxnet.dylib", "macho", "libmxnet.dylib")
            else:
                payload["error"] = (
                    "neither mxnet/libmxnet.so nor mxnet/libmxnet.dylib "
                    "found in wheel")
                return payload

            payload["has_libmxnet"] = True
            with tempfile.TemporaryDirectory(prefix="mxnet-wheel-provenance-") as tmpdir:
                libmxnet_path = Path(tmpdir) / local_name
                with wheel.open(libmxnet_name) as src, libmxnet_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

                if payload["format"] == "elf":
                    dynamic = _parse_dynamic_section(
                        _readelf_dynamic_section(libmxnet_path))
                    needed = dynamic["needed"]
                    payload["needed"] = needed
                    payload["opencv_needed"] = sorted(
                        s for s in needed if s.startswith("libopencv_"))
                    payload["openmp_needed"] = sorted(
                        s for s in needed if OPENMP_RUNTIME_RE.match(s))
                    payload["runpath"] = dynamic["runpath"]
                    entries = (
                        [] if dynamic["runpath"] is None
                        else dynamic["runpath"].split(":"))
                    payload["runpath_has_origin_lib"] = "$ORIGIN/lib" in entries
                    payload["runpath_has_nvidia_cudnn"] = (
                        "$ORIGIN/../nvidia/cudnn/lib" in entries)
                    payload["runpath_has_nvidia_nccl"] = (
                        "$ORIGIN/../nvidia/nccl/lib" in entries)
                else:
                    deps = _macho_dependencies(libmxnet_path)
                    rpaths = _macho_rpaths(libmxnet_path)
                    payload["needed"] = [os.path.basename(d) for d in deps]
                    opencv_refs = [
                        d for d in deps
                        if os.path.basename(d).startswith("libopencv_")]
                    payload["opencv_needed"] = sorted(
                        os.path.basename(d) for d in opencv_refs)
                    openmp_refs = [
                        d for d in deps
                        if OPENMP_RUNTIME_RE.match(os.path.basename(d))]
                    payload["openmp_needed"] = sorted(
                        os.path.basename(d) for d in openmp_refs)
                    # Self-containment: every OpenMP reference must be loader-
                    # relative (@loader_path/@rpath), i.e. the absolute build-host
                    # path was rewritten to resolve from inside the wheel.
                    payload["openmp_runtime_loader_relative"] = bool(openmp_refs) and all(
                        d.startswith("@loader_path") or d.startswith("@rpath")
                        for d in openmp_refs)
                    payload["runpath"] = ":".join(rpaths) if rpaths else None
                    # The Mach-O analog of "$ORIGIN/lib in RUNPATH": every OpenCV
                    # reference is loader-relative (@loader_path/@rpath), i.e. the
                    # absolute build-host path has been rewritten to resolve from
                    # inside the wheel.
                    payload["runpath_has_origin_lib"] = bool(opencv_refs) and all(
                        d.startswith("@loader_path") or d.startswith("@rpath")
                        for d in opencv_refs)

            payload["cudnn_needed"] = sorted(
                s for s in payload["needed"] if s.startswith("libcudnn"))
            payload["nccl_needed"] = sorted(
                s for s in payload["needed"] if s.startswith("libnccl"))
            payload["inspected"] = True
    except (OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as err:
        payload["error"] = str(err)
    return payload


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
            "payload": inspect_wheel_payload(wheel_path),
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
        "build": read_build_metadata(root, cmake_cache),
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
    expect_cudnn=None,
    expect_nccl=None,
    expect_onednn=None,
    expect_opencv=None,
    expect_onnx=None,
    expect_openmp=None,
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
        feature = features.get(name, {"enabled": None, "raw": None})
        if feature["enabled"] is None and not allow_missing_cache:
            errors.append("{} was not found as a boolean CMake feature".format(name))

    for name, expected in {
        "USE_CUDA": _expected_flag(expect_cuda),
        "USE_CUDNN": _expected_flag(expect_cudnn),
        "USE_NCCL": _expected_flag(expect_nccl),
        "USE_ONEDNN": _expected_flag(expect_onednn),
        "USE_OPENCV": _expected_flag(expect_opencv),
        "USE_OPENMP": _expected_flag(expect_openmp),
    }.items():
        feature = features.get(name, {"enabled": None, "raw": None})
        if expected is not None and feature["enabled"] is not expected:
            errors.append("{} expected {}, found {}".format(
                name, "ON" if expected else "OFF", feature["raw"] or "unknown"))

    build = report.get("build")
    if build and build["metadata_found"]:
        build_commits = build["commit_hashes"]
        if len(build_commits) != 1:
            errors.append(
                "CMake build metadata contains multiple MXNET_COMMIT_HASH values: "
                "{}".format(", ".join(build_commits))
            )
        elif build_commits[0] == "Unavailable":
            errors.append("CMake build metadata has MXNET_COMMIT_HASH=Unavailable")
        elif build_commits[0] != git["commit"]:
            errors.append(
                "CMake build metadata commit {} does not match git HEAD {}; "
                "rerun CMake configure and rebuild before packaging".format(
                    build_commits[0][:12], git["short_commit"]
                )
            )

    expected_cudnn = _expected_flag(expect_cudnn)
    expected_nccl = _expected_flag(expect_nccl)
    expected_opencv = _expected_flag(expect_opencv)
    expected_onnx = _expected_flag(expect_onnx)
    expected_openmp = _expected_flag(expect_openmp)
    for wheel in report["wheels"]:
        if not wheel["exists"]:
            errors.append("wheel file does not exist: {}".format(wheel["path"]))
        if not wheel["distribution_matches_package"]:
            errors.append("{} distribution {} does not match expected package {}".format(
                wheel["filename"], wheel["distribution"], report["expected_package_name"]))
        if not wheel["version_matches_package"]:
            errors.append("{} version {} does not match package version {}".format(
                wheel["filename"], wheel["version"], report["package"]["version"]))
        payload = wheel.get("payload")
        if payload is None:
            if expected_opencv is not None:
                errors.append("{} payload was not inspected".format(wheel["filename"]))
            continue
        if payload["error"] is not None:
            errors.append("{} payload inspection failed: {}".format(
                wheel["filename"], payload["error"]))
            continue
        if expected_cudnn is True:
            if not payload["has_libmxnet"]:
                errors.append("{} does not contain mxnet/libmxnet.so".format(
                    wheel["filename"]))
            if not payload["cudnn_needed"]:
                errors.append("{} libmxnet.so has no libcudnn NEEDED entries".format(
                    wheel["filename"]))
            if not payload["runpath_has_nvidia_cudnn"]:
                errors.append("{} libmxnet.so RUNPATH does not include "
                              "$ORIGIN/../nvidia/cudnn/lib".format(wheel["filename"]))
        elif expected_cudnn is False and payload["cudnn_needed"]:
            errors.append("{} libmxnet.so has cuDNN NEEDED entries despite "
                          "--expect-cudnn off: {}".format(
                              wheel["filename"], ", ".join(payload["cudnn_needed"])))

        if expected_nccl is True:
            if not payload["has_libmxnet"]:
                errors.append("{} does not contain mxnet/libmxnet.so".format(
                    wheel["filename"]))
            if not payload["nccl_needed"]:
                errors.append("{} libmxnet.so has no libnccl NEEDED entries".format(
                    wheel["filename"]))
            if not payload["runpath_has_nvidia_nccl"]:
                errors.append("{} libmxnet.so RUNPATH does not include "
                              "$ORIGIN/../nvidia/nccl/lib".format(wheel["filename"]))
        elif expected_nccl is False and payload["nccl_needed"]:
            errors.append("{} libmxnet.so has NCCL NEEDED entries despite "
                          "--expect-nccl off: {}".format(
                              wheel["filename"], ", ".join(payload["nccl_needed"])))

        if expected_opencv is True:
            if not payload["has_libmxnet"]:
                errors.append("{} does not contain mxnet/libmxnet.so".format(
                    wheel["filename"]))
            if not payload["opencv_needed"]:
                errors.append("{} libmxnet.so has no libopencv_* NEEDED entries".format(
                    wheel["filename"]))
            if not payload["opencv_bundled"]:
                errors.append("{} does not bundle mxnet/lib/libopencv_*".format(
                    wheel["filename"]))
            for soname in payload["opencv_needed"]:
                if soname not in payload["opencv_bundled_sonames"]:
                    errors.append("{} needs {} but does not bundle that SONAME".format(
                        wheel["filename"], soname))
            if not payload["runpath_has_origin_lib"]:
                errors.append("{} libmxnet.so RUNPATH does not include $ORIGIN/lib".format(
                    wheel["filename"]))
        elif expected_opencv is False and payload["opencv_needed"]:
            errors.append("{} libmxnet.so has OpenCV NEEDED entries despite "
                          "--expect-opencv off: {}".format(
                              wheel["filename"], ", ".join(payload["opencv_needed"])))

        if expected_onnx is not None and not payload["metadata_found"]:
            errors.append("{} has no .dist-info/METADATA to check the onnx "
                          "dependency policy".format(wheel["filename"]))
        if expected_onnx is True:
            if not payload["onnx_pkg_files"]:
                errors.append("{} does not ship the mxnet/onnx package".format(
                    wheel["filename"]))
            if not payload["onnx_hard_require"]:
                errors.append("{} METADATA does not declare onnx as a hard runtime "
                              "dependency (expected for the CUDA wheel)".format(
                                  wheel["filename"]))
        elif expected_onnx is False and payload["onnx_hard_require"]:
            errors.append("{} declares onnx as a hard dependency despite "
                          "--expect-onnx off (it should be the optional [onnx] "
                          "extra)".format(wheel["filename"]))

        # OpenMP runtime: the macOS wheel vendors libomp into mxnet/lib/ and
        # rewrites libmxnet's reference to a loader-relative path (no system
        # libomp to fall back on); on Linux the runtime (libgomp) is host-provided
        # and not bundled, so only the build flag (checked above) is asserted
        # there.  The self-containment payload check is therefore macOS-specific.
        if expected_openmp is True and payload["format"] == "macho":
            if not payload["openmp_needed"]:
                errors.append("{} libmxnet.dylib has no OpenMP runtime dependency "
                              "despite --expect-openmp on".format(wheel["filename"]))
            if not payload["openmp_bundled"]:
                errors.append("{} does not bundle the OpenMP runtime "
                              "(mxnet/lib/libomp*)".format(wheel["filename"]))
            if not payload["openmp_runtime_loader_relative"]:
                errors.append("{} libmxnet.dylib OpenMP reference is not "
                              "loader-relative (it would not resolve from inside "
                              "the wheel on a clean host)".format(wheel["filename"]))
        elif (expected_openmp is False and payload["format"] == "macho"
              and payload["openmp_bundled"]):
            errors.append("{} bundles an OpenMP runtime despite --expect-openmp "
                          "off".format(wheel["filename"]))
    return errors


def _flag_text(value):
    return "ON" if value is True else "OFF" if value is False else "unknown"


def format_text_report(report, errors):
    git, package, features = report["git"], report["package"], report["features"]
    build = report.get("build")
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
        feature = features.get(name, {"enabled": None, "raw": None})
        raw = "" if feature["raw"] is None else " [{}]".format(feature["raw"])
        lines.append("  {}: {}{}".format(name, _flag_text(feature["enabled"]), raw))

    if build:
        lines.append("  CMake build metadata: {}".format(
            ",".join(build["commit_hashes"]) if build["metadata_found"] else "not found"
        ))

    lines.append("  wheels:" if report["wheels"] else "  wheels: none")
    for wheel in report["wheels"]:
        lines.append(
            "    {}: distribution={} version={} version_match={} exists={}".format(
                wheel["path"], wheel["distribution"], wheel["version"],
                "yes" if wheel["version_matches_package"] else "no",
                "yes" if wheel["exists"] else "no"))
        payload = wheel.get("payload")
        if payload is not None:
            if payload["error"] is not None:
                lines.append("      payload: error={}".format(payload["error"]))
            else:
                lines.append(
                    "      payload: libmxnet={} cudnn_needed={} nccl_needed={} "
                    "opencv_needed={} opencv_bundled={} origin_lib_runpath={} "
                    "cudnn_runpath={} nccl_runpath={}".format(
                        "yes" if payload["has_libmxnet"] else "no",
                        ",".join(payload["cudnn_needed"]) or "none",
                        ",".join(payload["nccl_needed"]) or "none",
                        ",".join(payload["opencv_needed"]) or "none",
                        len(payload["opencv_bundled"]),
                        "yes" if payload["runpath_has_origin_lib"] else "no",
                        "yes" if payload["runpath_has_nvidia_cudnn"] else "no",
                        "yes" if payload["runpath_has_nvidia_nccl"] else "no"))
                lines.append(
                    "      onnx: pkg_files={} hard_require={} extra_require={}".format(
                        len(payload["onnx_pkg_files"]),
                        "yes" if payload["onnx_hard_require"] else "no",
                        "yes" if payload["onnx_extra_require"] else "no"))
                lines.append(
                    "      openmp: needed={} bundled={} loader_relative={}".format(
                        ",".join(payload["openmp_needed"]) or "none",
                        len(payload["openmp_bundled"]),
                        "yes" if payload["openmp_runtime_loader_relative"] else "no"))

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
    parser.add_argument("--expect-cudnn", choices=("on", "off"))
    parser.add_argument("--expect-nccl", choices=("on", "off"))
    parser.add_argument("--expect-onednn", choices=("on", "off"))
    parser.add_argument("--expect-opencv", choices=("on", "off"))
    parser.add_argument("--expect-onnx", choices=("on", "off"))
    parser.add_argument("--expect-openmp", choices=("on", "off"))
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
            expect_cudnn=args.expect_cudnn,
            expect_nccl=args.expect_nccl,
            expect_onednn=args.expect_onednn,
            expect_opencv=args.expect_opencv,
            expect_onnx=args.expect_onnx,
            expect_openmp=args.expect_openmp,
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
