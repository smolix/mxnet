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

# pylint: disable=undefined-all-variable, not-callable, cell-var-from-loop
"""Operators that fallback to official NumPy implementation."""

import sys
from functools import wraps
import numpy as onp

fallbacks = [
    '__version__',
    '_NoValue',
    'allclose',
    'alltrue',
    'apply_along_axis',
    'apply_over_axes',
    'argpartition',
    'argwhere',
    'array_equal',
    'array_equiv',
    'choose',
    'compress',
    'corrcoef',
    'correlate',
    'count_nonzero',
    'cov',
    'cumprod',
    'digitize',
    'divmod',
    'dtype',
    'extract',
    'float_power',
    'frexp',
    'heaviside',
    'histogram2d',
    'histogram_bin_edges',
    'histogramdd',
    'i0',
    'in1d',
    'intersect1d',
    'isclose',
    'isin',
    'ix_',
    'lexsort',
    'min_scalar_type',
    'mirr',
    'modf',
    'msort',
    'nanargmax',
    'nanargmin',
    'nancumprod',
    'nancumsum',
    'nanmax',
    'nanmedian',
    'nanmin',
    'nanpercentile',
    'nanprod',
    'nanquantile',
    'nanstd',
    'nansum',
    'nanvar',
    'ndim',
    'npv',
    'packbits',
    'partition',
    'piecewise',
    'pmt',
    'poly',
    'polyadd',
    'polydiv',
    'polyfit',
    'polyint',
    'polymul',
    'polysub',
    'positive',
    'ppmt',
    'promote_types',
    'ptp',
    'pv',
    'rate',
    'real',
    'roots',
    'searchsorted',
    'select',
    'setdiff1d',
    'setxor1d',
    'signbit',
    'size',
    'spacing',
    'take_along_axis',
    'trapz',
    'tril_indices_from',
    'trim_zeros',
    'union1d',
    'unpackbits',
    'unwrap',
    'vander',
]

fallback_mod = sys.modules[__name__]

# Drop fallbacks that don't actually work in the installed NumPy. NumPy 1.20
# removed the financial functions (mirr/npv/pmt/...) per NEP 32; the symbols
# still exist but raise RuntimeError when called. Some others (alltrue/
# sometrue/msort) have also since been retired. Filter by actually trying to
# fetch the attribute and call its repr — anything that raises is dropped.
import warnings as _w

_RETIRED = {'mirr', 'npv', 'pmt', 'ppmt', 'pv', 'rate', 'fv', 'ipmt', 'nper',
            'alltrue', 'sometrue', 'msort', 'product', 'cumproduct', 'round_'}

def _available(name):
    if name in {'__version__', '_NoValue'}:
        return True
    if name in _RETIRED:
        return False
    with _w.catch_warnings():
        _w.simplefilter("ignore", DeprecationWarning)
        if not hasattr(onp, name):
            return False
        obj = getattr(onp, name)
    if not callable(obj):
        return True
    # NEP-32-expired stubs raise RuntimeError on call rather than at attribute
    # access. Trigger the check via the wrapper's `__signature__` or similar
    # attribute that the expired-shim doesn't override; if we can't get a
    # docstring, treat as unavailable.
    try:
        with _w.catch_warnings():
            _w.simplefilter("error", DeprecationWarning)
            _ = obj.__doc__  # touches the wrapper; on expired stubs may warn
    except Exception:
        return False
    return True

fallbacks = [name for name in fallbacks if _available(name)]
# Explicitly drop names known to be RuntimeError-on-call shims in modern NumPy
fallbacks = [name for name in fallbacks if name not in _RETIRED]

def get_func(obj, doc):
    """Get new numpy function with object and doc"""
    @wraps(obj)
    def wrapper(*args, **kwargs):
        return obj(*args, **kwargs)
    wrapper.__doc__ = doc
    return wrapper

for obj_name in fallbacks:
    onp_obj = getattr(onp, obj_name)
    if callable(onp_obj):
        new_fn_doc = onp_obj.__doc__
        if obj_name in {'divmod', 'float_power', 'frexp', 'heaviside', 'modf', 'signbit', 'spacing'}:
            # remove reference of kwargs doc and the reference to ufuncs
            new_fn_doc = new_fn_doc.replace("**kwargs\n    For other keyword-only arguments, see the"
                                            + "\n    :ref:`ufunc docs <ufuncs.kwargs>`.", '')
        elif obj_name == 'trapz':
            # remove unused reference
            new_fn_doc = new_fn_doc.replace(
                '.. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule', '')
        elif obj_name == "i0":
            # replace broken link
            new_fn_doc = new_fn_doc.replace(
                '.. [3] http://kobesearch.cpan.org/htdocs/Math-Cephes/Math/Cephes.html',
                '.. [3] https://metacpan.org/pod/distribution/Math-Cephes/lib/Math/Cephes.pod \
                    #i0:-Modified-Bessel-function-of-order-zero')
        setattr(fallback_mod, obj_name, get_func(onp_obj, new_fn_doc))
    else:
        setattr(fallback_mod, obj_name, onp_obj)

__all__ = fallbacks
