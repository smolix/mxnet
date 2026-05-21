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

# pylint: disable=invalid-name, exec-used
"""Setup mxnet package."""
from __future__ import absolute_import
import os
import platform
import sys
from setuptools import find_packages # This must precede distutils

BASE_INSTALL_REQUIRES = [
    # NumPy 2.x compatibility is being restored incrementally; keep the
    # package metadata conservative until the full NumPy 2 test sweep passes.
    'numpy>=1.17,<2',
    'requests>=2.20.0,<3',
    'graphviz<0.9.0,>=0.8.1',
    'packaging>=20.0',
    'contextvars;python_version<"3.7"',
]

# CUDA / cuDNN / NCCL runtime libraries are NOT bundled in Linux CUDA wheels.
# libmxnet.so is patched with RUNPATH=$ORIGIN/lib:$ORIGIN/../nvidia/<pkg>/lib:/usr/local/cuda/lib64
# so the loader finds:
#   - cuDNN / NCCL via the pip-installed nvidia-*-cu13 wheels under
#     site-packages/nvidia/<pkg>/lib/  (PyTorch/JAX install layout)
#   - libcudart / libcublas / libcufft / libcusolver / libcurand /
#     libnvrtc out of the system CUDA 13 toolkit at /usr/local/cuda/
#
# As of 2026-05-17 NVIDIA has published only `nvidia-cudnn-cu13`
# (9.22) and `nvidia-nccl-cu13` (2.30) on PyPI for CUDA 13; the other
# nvidia-*-cu13 packages are placeholder stubs (0.0.1). When they
# ship real, append them here.
CUDA_RUNTIME_INSTALL_REQUIRES = [
    'nvidia-cudnn-cu13>=9.22,<10',
    'nvidia-nccl-cu13>=2.28,<3',
]

# Python image/RecordIO helpers import cv2 when OpenCV support is enabled.
# This is separate from native libopencv_*.so bundling policy.
OPENCV_PYTHON_INSTALL_REQUIRES = [
    'opencv-python>=4,<5',
]

# need to use distutils.core for correct placement of cython dll
kwargs = {}
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup, Distribution
    from setuptools.extension import Extension
    kwargs = {
        'install_requires': list(BASE_INSTALL_REQUIRES),
        'zip_safe': False,
    }

    # The wheel ships a native libmxnet shared library under mxnet/, so it must be
    # tagged as a binary, platform-specific distribution (Root-Is-Purelib=
    # false), not as a pure-python wheel.
    # Force that by declaring a distclass whose has_ext_modules() returns
    # True even when ext_modules is empty.
    class _BinaryDistribution(Distribution):
        def has_ext_modules(self):
            return True
    kwargs['distclass'] = _BinaryDistribution

with_cython = False
if '--with-cython' in sys.argv:
    with_cython = True
    sys.argv.remove('--with-cython')

# We can not import `mxnet.info.py` in setup.py directly since mxnet/__init__.py
# Will be invoked which introduces dependences
CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, 'mxnet/libinfo.py')
libinfo = {'__file__': libinfo_py}
exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)

LIB_PATH = libinfo['find_lib_path']()
__version__ = libinfo['__version__']

sys.path.insert(0, CURRENT_DIR)


def _env_flag(name):
    """Parse an optional boolean environment flag."""
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() not in ('0', 'false', 'off', 'no')


def _find_mxnet_packages():
    """Find Python packages, optionally omitting ONNX-only integration code."""
    excludes = []
    if _env_flag('MXNET_SETUP_EXCLUDE_ONNX'):
        excludes.extend([
            'mxnet.onnx',
            'mxnet.onnx.*',
            'mxnet.contrib.onnx',
            'mxnet.contrib.onnx.*',
        ])
    return find_packages(exclude=excludes)


def _package_data():
    """Return package data bundled into the mxnet package."""
    return {'mxnet': [
        'libmxnet.so',
        'libmxnet.dylib',
        'lib/*.so',
        'lib/*.so.*',
        'lib/*.dylib',
    ] + bundled_libs}


def _exclude_package_data():
    """Return package data exclusions for optional integrations."""
    if not _env_flag('MXNET_SETUP_EXCLUDE_ONNX'):
        return {}
    return {
        'mxnet': [
            'onnx/*',
            'onnx/**/*',
        ],
        'mxnet.contrib': [
            'onnx/*',
            'onnx/**/*',
        ],
    }


def _feature_enabled_from_cmake_cache(feature_name):
    """Read a USE_* feature flag from a nearby CMakeCache.txt."""
    cache_paths = []

    def add_cache_path(path):
        if path not in cache_paths:
            cache_paths.append(path)

    for lib_path in LIB_PATH:
        add_cache_path(os.path.join(os.path.dirname(lib_path), 'CMakeCache.txt'))
    add_cache_path(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'build', 'CMakeCache.txt')))

    prefix = '{}:'.format(feature_name)
    for cache_path in cache_paths:
        if not os.path.exists(cache_path):
            continue
        with open(cache_path) as cache_file:
            for line in cache_file:
                if line.startswith(prefix):
                    _, value = line.strip().split('=', 1)
                    return value.upper() in ('1', 'ON', 'TRUE', 'YES')
    return None


def _cuda_enabled_from_cmake_cache():
    """Read USE_CUDA from a nearby CMakeCache.txt when building from a tree."""
    return _feature_enabled_from_cmake_cache('USE_CUDA')


def _cuda_enabled_from_runtime():
    """Ask libmxnet for CUDA support when its dependencies are loadable."""
    try:
        from mxnet.runtime import Features
        return Features().is_enabled('CUDA')
    except Exception: # pylint: disable=broad-except
        return None


def _include_cuda_runtime_deps():
    """Return whether this wheel should depend on external NVIDIA CUDA libs."""
    override = _env_flag('MXNET_SETUP_ENABLE_CUDA_DEPS')
    if override is not None:
        return override
    if platform.system() != 'Linux':
        return False

    detected = _cuda_enabled_from_cmake_cache()
    if detected is not None:
        return detected
    detected = _cuda_enabled_from_runtime()
    if detected is not None:
        return detected

    # Keep the prior Linux packaging behavior if a CUDA build cannot be probed
    # because dependent shared libraries are unavailable during setup.
    return True


def _opencv_enabled_from_cmake_cache():
    """Read USE_OPENCV from a nearby CMakeCache.txt when building from a tree."""
    return _feature_enabled_from_cmake_cache('USE_OPENCV')


def _opencv_enabled_from_runtime():
    """Ask libmxnet for OpenCV support when its dependencies are loadable."""
    try:
        from mxnet.runtime import Features
        return Features().is_enabled('OPENCV')
    except Exception: # pylint: disable=broad-except
        return None


def _include_opencv_python_deps():
    """Return whether this wheel should depend on Python OpenCV."""
    override = _env_flag('MXNET_SETUP_ENABLE_OPENCV_DEPS')
    if override is not None:
        return override

    detected = _opencv_enabled_from_cmake_cache()
    if detected is not None:
        return detected
    detected = _opencv_enabled_from_runtime()
    if detected is not None:
        return detected

    # OpenCV is optional, so do not force cv2 into metadata when the build
    # feature cannot be probed.
    return False


if 'install_requires' in kwargs and _include_cuda_runtime_deps():
    kwargs['install_requires'].extend(CUDA_RUNTIME_INSTALL_REQUIRES)
if 'install_requires' in kwargs and _include_opencv_python_deps():
    kwargs['install_requires'].extend(OPENCV_PYTHON_INSTALL_REQUIRES)

# NVIDIA runtime libs are NOT bundled — see install_requires above.
# libmxnet.so's RUNPATH points at site-packages/nvidia/<pkg>/lib/ for them.
bundled_libs = []

# Try to generate auto-complete code (skipped when MXNET_SETUP_SKIP_AUTOCOMPLETE=1
# is set; useful when packaging in environments where loading libmxnet.so is slow
# or undesirable).
if not os.environ.get('MXNET_SETUP_SKIP_AUTOCOMPLETE'):
    try:
        from mxnet.base import _generate_op_module_signature
        from mxnet.ndarray.register import _generate_ndarray_function_code
        from mxnet.symbol.register import _generate_symbol_function_code
        _generate_op_module_signature('mxnet', 'symbol', _generate_symbol_function_code)
        _generate_op_module_signature('mxnet', 'ndarray', _generate_ndarray_function_code)
    except: # pylint: disable=bare-except
        pass

def config_cython():
    """Try to configure cython and return cython configuration"""
    if not with_cython:
        return []
    # pylint: disable=unreachable
    if os.name == 'nt':
        print("WARNING: Cython is not supported on Windows, will compile without cython module")
        return []

    try:
        from Cython.Build import cythonize
        subdir = "_cy3"
        ret = []
        path = "mxnet/cython"
        if os.name == 'nt':
            library_dirs = ['mxnet', '../build/Release', '../build']
            libraries = ['libmxnet']
        elif platform.system() == 'Darwin':
            library_dirs = [os.path.dirname(p) for p in LIB_PATH]
            libraries = ['mxnet']
            # Default paths to libmxnet.dylib relative to the generated Cython extension.
            # These precede DYLD_LIBRARY_PATH.
            extra_link_args = [
                "-Wl,-rpath,@loader_path/..",
                "-Wl,-rpath,@loader_path/../..",
                "-Wl,-rpath,@loader_path/../../../lib",
                "-Wl,-rpath,@loader_path/../../../build",
            ]
        else:
            library_dirs = [os.path.dirname(p) for p in LIB_PATH]
            libraries = ['mxnet']
            # Default paths to libmxnet.so relative to the shared library file generated by cython.
            # These precede LD_LIBRARY_PATH.
            extra_link_args = ["-Wl,-rpath,$ORIGIN/..:$ORIGIN/../../../lib:$ORIGIN/../../../build"]

        for fn in os.listdir(path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(Extension(
                f"mxnet.{subdir}.{fn[:-4]}",
                [f"mxnet/cython/{fn}"],
                include_dirs=["../include/", "../3rdparty/tvm/nnvm/include"],
                library_dirs=library_dirs,
                libraries=libraries,
                extra_link_args=extra_link_args,
                language="c++"))

        path = "mxnet/_ffi/_cython"
        for fn in os.listdir(path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(Extension(
                f"mxnet._ffi.{subdir}.{fn[:-4]}",
                [f"mxnet/_ffi/_cython/{fn}"],
                include_dirs=["../include/", "../3rdparty/tvm/nnvm/include"],
                library_dirs=library_dirs,
                libraries=libraries,
                extra_compile_args=["-std=c++17"],
                extra_link_args=extra_link_args,
                language="c++"))

        # If `force=True` is not used and you cythonize the modules for python2 and python3
        # successively, you need to delete `mxnet/cython/ndarray.cpp` after the first cythonize.
        return cythonize(ret, force=True)
    except ImportError:
        print("WARNING: Cython is not installed, will compile without cython module")
        return []


setup(name='mxnet',
      version=__version__,
      description=open(os.path.join(CURRENT_DIR, 'README.md')).read(),
      packages=_find_mxnet_packages(),
      # Bundle libmxnet.{so,dylib} plus staged runtime libraries inside the
      # package so libinfo.find_lib_path() can find libmxnet under mxnet/ at
      # install time. data_files goes to <sysprefix>/mxnet/ which find_lib_path()
      # doesn't search.
      package_data=_package_data(),
      exclude_package_data=_exclude_package_data(),
      include_package_data=True,
      url='https://github.com/apache/mxnet',
      ext_modules=config_cython(),
      classifiers=[
          # https://pypi.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Cython',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      **kwargs)
