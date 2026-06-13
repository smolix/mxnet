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

"""Register backend ops in mxnet.ndarray namespace"""
import os as _os
import ctypes
import numpy as _np  # pylint: disable=unused-import

from .ndarray import get_dtype_name
from ._internal import NDArrayBase, _imperative_invoke # pylint: disable=unused-import
from ..ndarray_doc import _build_doc

from ..base import mx_uint, check_call, _LIB, py_str, _init_op_module, _Null, _is_np_op, _output_is_list, MXNetError  # pylint: disable=unused-import
from ..util import use_np_shape, _check_same_device  # pylint: disable=unused-import


def _verify_all_np_ndarrays(op_name, func_name, args, out):
    """Verify if all the arrays are numpy ndarrays.

    Parameters
    ----------
    op_name : str
        Operator full name registered in backend.
    func_name : str
        Operator name exposed to users. This is usually the name by stripping off
        the prefix of the full operator names registered in backend.
    args : list of arrays
        Input ndarray arguments to be checked.
    out : ndarray or None or list of ndarrays
        User-provided output ndarrays.
    """
    from ..numpy import ndarray as np_ndarray
    for arr in args:
        if (arr is not None) and (not isinstance(arr, np_ndarray)):
            raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. '
                            'This is a numpy operator which can only accept '
                            'MXNet numpy ndarrays, while received a legacy ndarray. '
                            'Please ensure that you have activated numpy semantics by calling '
                            '`npx.set_np()` in your code. If you still see this error with numpy '
                            'semantics activated, please call `as_np_ndarray()` upon the legacy '
                            'ndarray to convert it to an MXNet numpy ndarray, and then feed the '
                            'converted array to this operator.'
                            .format(op_name, func_name))
    if out is None:
        return
    if not isinstance(out, (list, tuple)):
        out = [out]
    for arr in out:
        if (arr is not None) and (not isinstance(arr, np_ndarray)):
            raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. '
                            'This is a numpy operator which can only accept '
                            'MXNet numpy ndarrays, while received a legacy ndarray. '
                            'Please ensure that you have activated numpy semantics by calling '
                            '`npx.set_np()` in your code. If you still see this error with numpy '
                            'semantics activated, please call `as_np_ndarray()` upon the legacy '
                            'ndarray to convert it to an MXNet numpy ndarray, and then feed the '
                            'converted array to this operator.'
                            .format(op_name, func_name))


def _verify_all_legacy_ndarrays(op_name, func_name, args, out):
    """Verify if all the arrays are legacy ndarrays.

    Parameters
    ----------
    op_name : str
        Operator full name registered in backend.
    func_name : str
        Operator name exposed to users. This is usually the name by stripping off
        the prefix of the full operator names registered in backend.
    args : list of arrays
        Input ndarray arguments to be checked.
    out : ndarray or None or list of ndarrays
        User-provided output ndarrays.
    """
    from ..numpy import ndarray as np_ndarray
    for arr in args:
        if (arr is not None) and (isinstance(arr, np_ndarray)):
            raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. '
                            'This is a legacy operator which can only accept '
                            'legacy ndarrays, while received an MXNet numpy ndarray. '
                            'Please call `as_nd_ndarray()` upon the numpy ndarray to '
                            'convert it to a legacy ndarray, and then feed the converted '
                            'array to this operator.'
                            .format(op_name, func_name))
    if out is None:
        return
    if not isinstance(out, (list, tuple)):
        out = [out]
    for arr in out:
        if (arr is not None) and (isinstance(arr, np_ndarray)):
            raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. '
                            'This is a legacy operator which can only write to '
                            'legacy ndarrays, while received an MXNet numpy ndarray. '
                            'Please call `as_nd_ndarray()` upon the numpy ndarray to '
                            'convert it to a legacy ndarray, and then feed the converted '
                            'array to this operator.'
                            .format(op_name, func_name))



def _is_true_param(value):
    return value is True or str(value).lower() == "true"


def _to_int_param(value, name):
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be an integer".format(name)) from None
    if isinstance(value, float) and result != value:
        raise ValueError("{} must be an integer".format(name))
    return result


def _tuple_param(value, name):
    if isinstance(value, _np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _validate_positive_int(value, name):
    value = _to_int_param(value, name)
    if value <= 0:
        raise ValueError("{} must be greater than 0".format(name))
    return value


def _validate_size_param(value, name="size", allow_single=True):
    values = _tuple_param(value, name)
    valid_lengths = (1, 2) if allow_single else (2,)
    if len(values) not in valid_lengths:
        raise ValueError("{} must contain {} positive dimension values".format(
            name, "one or two" if allow_single else "two"))
    return tuple(_validate_positive_int(v, "{} dimension".format(name)) for v in values)


def _validate_float_pair(value, name, lower=None, upper=None, strictly_positive=False):
    values = _tuple_param(value, name)
    if len(values) != 2:
        raise ValueError("{} range must contain two values".format(name))
    try:
        low, high = float(values[0]), float(values[1])
    except (TypeError, ValueError):
        raise ValueError("{} range values must be numeric".format(name)) from None
    if low > high:
        raise ValueError("{} range lower bound must not exceed upper bound".format(name))
    if strictly_positive and low <= 0:
        raise ValueError("{} range values must be greater than 0".format(name))
    if lower is not None and low < lower:
        raise ValueError("{} range values must be at least {}".format(name, lower))
    if upper is not None and high > upper:
        raise ValueError("{} range values must be at most {}".format(name, upper))
    return low, high


def _validate_image_shape(data, op_name):
    ndim = len(data.shape)
    if ndim not in (3, 4):
        raise ValueError("{} expects input image dimension to be 3 or 4, but got {}".format(
            op_name, ndim))


def _validate_image_crop(data, x, y, width, height):
    _validate_image_shape(data, "image crop")
    x = _to_int_param(x, "x offset")
    y = _to_int_param(y, "y offset")
    width = _validate_positive_int(width, "width")
    height = _validate_positive_int(height, "height")
    if x < 0:
        raise ValueError("x offset must be non-negative")
    if y < 0:
        raise ValueError("y offset must be non-negative")
    src_h = data.shape[-3]
    src_w = data.shape[-2]
    if x + width > src_w:
        raise ValueError("x offset plus width exceeds input width")
    if y + height > src_h:
        raise ValueError("y offset plus height exceeds input height")


def _validate_interp_param(interp):
    interp_id = _to_int_param(interp, "interp")
    if interp_id not in (0, 1, 2, 3, 4):
        raise ValueError("Unknown interp method {}".format(interp))
    return interp_id


def _validate_image_random_crop_params(get_param):
    interp = get_param("interp")
    if interp is not None:
        _validate_interp_param(interp)
    _validate_size_param((get_param("width"), get_param("height")),
                         "crop size", allow_single=False)
    _validate_float_pair(get_param("xrange", (0.0, 1.0)), "xrange", lower=0.0, upper=1.0)
    _validate_float_pair(get_param("yrange", (0.0, 1.0)), "yrange", lower=0.0, upper=1.0)


def _validate_sequence_length(data, sequence_length, axis, op_name):
    if len(data.shape) <= 1:
        raise ValueError("{} data shape must have rank 2 or greater".format(op_name))
    axis = _to_int_param(axis, "axis")
    if op_name in ("SequenceReverse", "_npx_sequence_reverse") and axis != 0:
        raise ValueError("SequenceReverse only supports axis 0")
    if axis not in (0, 1):
        raise ValueError("{} axis must be 0 or 1".format(op_name))
    expected_batch = data.shape[0] if axis else data.shape[1]
    max_length = data.shape[axis]
    if len(sequence_length.shape) != 1 or sequence_length.shape[0] != expected_batch:
        raise ValueError("sequence_length shape must be ({},), got {}".format(
            expected_batch, sequence_length.shape))
    lengths = sequence_length.asnumpy()
    if lengths.size and ((lengths <= 0).any() or (lengths > max_length).any()):
        raise ValueError("sequence_length values must be in range [1, {}]".format(max_length))


def _validate_image_random_resized_crop_params(get_param):
    size = get_param("size")
    if size is not None:
        _validate_size_param(size, "resize crop size")
    _validate_float_pair(get_param("area", (0.08, 1.0)), "area",
                         lower=0.0, upper=1.0, strictly_positive=True)
    _validate_float_pair(get_param("ratio", (3.0 / 4.0, 4.0 / 3.0)), "ratio",
                         strictly_positive=True)
    interp = get_param("interp")
    if interp is not None:
        _validate_interp_param(interp)
    max_trial = get_param("max_trial")
    if max_trial is not None:
        # max_trial == 0 is valid: it forces the deterministic center-crop
        # fallback in the native op, so only reject negative values.
        if _to_int_param(max_trial, "max_trial") < 0:
            raise ValueError("max_trial must be non-negative")

def _imperative_invoke_checked(handle, ndargs, param_keys, param_vals,
                               out, is_np_op, output_is_list, op_name):
    def get_param(name, default=None):
        try:
            return param_vals[param_keys.index(name)]
        except ValueError:
            return default

    # NOTE: Python-side validation of the image ops (resize/crop/random_crop/
    # random_resized_crop) was removed: it rejected valid inputs (e.g. the size
    # formats the Gluon vision transforms pass, max_trial=0, interp 9/10) and its
    # only benefit was a slightly nicer error for invalid interp, which OpenCV
    # already raises as a catchable exception. The native ops handle validation.
    if op_name in ("SequenceLast", "SequenceReverse", "_npx_sequence_last",
                   "_npx_sequence_reverse"):
        use_sequence_length = get_param("use_sequence_length", False)
        if _is_true_param(use_sequence_length):
            if len(ndargs) < 2:
                raise ValueError("{} requires sequence_length when use_sequence_length=True".format(
                    op_name))
            _validate_sequence_length(ndargs[0], ndargs[1], get_param("axis", 0), op_name)

    if op_name == "_npx_gammaln" and ndargs and _np.issubdtype(ndargs[0].dtype, _np.integer):
        ndargs = list(ndargs)
        ndargs[0] = ndargs[0].astype("float32")

    if op_name == "_contrib_box_encode" and len(ndargs) >= 2 and 0 in ndargs[1].shape:
        raise ValueError("refs input for box_encode must not be empty; got shape {}".format(ndargs[1].shape))

    if op_name == "SequenceMask" and len(ndargs) >= 2:
        use_sequence_length = get_param("use_sequence_length", False)
        if use_sequence_length in (True, "True", "true"):
            axis = int(get_param("axis", 0))
            axis_size = ndargs[0].shape[axis]
            lengths = ndargs[1].asnumpy()
            if lengths.size and ((lengths < 0).any() or (lengths > axis_size).any()):
                raise ValueError("sequence_length values must be in range [0, {}]".format(axis_size))

    if op_name == "_contrib_boolean_mask" and ndargs and 0 in ndargs[0].shape:
        raise ValueError("boolean_mask does not support empty input data")

    if op_name in ("_contrib_interleaved_matmul_selfatt_qk", "_npx_interleaved_matmul_selfatt_qk"):
        heads = get_param("heads")
        if heads is not None and int(heads) <= 0:
            raise ValueError("heads must be positive")

    restore_float16 = False
    if op_name in ("sort", "argsort") and ndargs and ndargs[0].dtype == _np.dtype("float16"):
        ndargs = list(ndargs)
        ndargs[0] = ndargs[0].astype("float32")
        restore_float16 = op_name == "sort"

    if op_name in ("_contrib_arange_like", "_npx_arange_like"):
        repeat = get_param("repeat", 1)
        if int(repeat) <= 0:
            raise ValueError("repeat must be positive")

    _check_same_device(ndargs, func_name=op_name)

    try:
        result = _imperative_invoke(handle, ndargs, param_keys, param_vals,
                                    out, is_np_op, output_is_list)
    except MXNetError as err:
        message = str(err)
        if " expects " in message and " inputs, but got " in message and " instead" in message:
            raise TypeError(message) from None
        raise

    if restore_float16:
        result = result.astype("float16")

    if op_name == "_sparse_elemwise_mul" and getattr(result, "stype", None) == "csr":
        result = result.tostype("default").tostype("csr")

    if op_name in ("max", "min") and ndargs and out is None and get_param("axis") is None:
        from . import op as _op  # pylint: disable=import-outside-toplevel
        data = ndargs[0]
        mask = _op.broadcast_equal(data, result).astype(data.dtype)
        result = _op.sum(data * mask / _op.sum(mask))

    if op_name == "prod" and ndargs and out is None and get_param("axis") is None:
        data = ndargs[0]
        if (data.asnumpy() == 0).sum() > 1:
            from . import op as _op  # pylint: disable=import-outside-toplevel
            result = _op.sum(data) * 0

    if op_name == "topk":
        ret_typ = get_param("ret_typ", "indices")
        if ret_typ == "both" and isinstance(result, (list, tuple)) and len(result) >= 2:
            result = list(result)
            result[1] = result[1].astype("int64")
        elif ret_typ == "indices":
            result = result.astype("int64")
    return result


# pylint: disable=too-many-locals
def _generate_ndarray_function_code(handle, op_name, func_name, signature_only=False):
    """Generate function for ndarray op by handle and function op_name."""
    real_name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    key_var_num_args = ctypes.c_char_p()
    ret_type = ctypes.c_char_p()

    check_call(_LIB.MXSymbolGetAtomicSymbolInfo(
        handle, ctypes.byref(real_name), ctypes.byref(desc),
        ctypes.byref(num_args),
        ctypes.byref(arg_names),
        ctypes.byref(arg_types),
        ctypes.byref(arg_descs),
        ctypes.byref(key_var_num_args),
        ctypes.byref(ret_type)))
    narg = int(num_args.value)
    arg_names = [py_str(arg_names[i]) for i in range(narg)]
    arg_types = [py_str(arg_types[i]) for i in range(narg)]
    key_var_num_args = py_str(key_var_num_args.value)
    ret_type = py_str(ret_type.value) if ret_type.value is not None else ''
    doc_str = _build_doc(op_name,
                         py_str(desc.value),
                         arg_names,
                         arg_types,
                         [py_str(arg_descs[i]) for i in range(narg)],
                         key_var_num_args,
                         ret_type)

    dtype_name = None
    arr_name = None
    ndsignature = []
    signature = []
    ndarg_names = []
    kwarg_names = []
    for i in range(narg):
        name, atype = arg_names[i], arg_types[i]
        if name == 'dtype':
            dtype_name = name
            signature.append(f'{name}=_Null')
        elif atype.startswith('NDArray') or atype.startswith('Symbol'):
            assert not arr_name, \
                "Op can only have one argument with variable " \
                "size and it must be the last argument."
            if atype.endswith('[]'):
                ndsignature.append(f'*{name}')
                arr_name = name
            else:
                ndsignature.append(f'{name}=None')
                ndarg_names.append(name)
        else:
            signature.append(f'{name}=_Null')
            kwarg_names.append(name)
    signature.append('out=None')
    signature.append('name=None')
    signature.append('**kwargs')
    signature = ndsignature + signature

    code = []
    is_np_op = _is_np_op(op_name)
    output_is_list = _output_is_list(op_name)
    doc_str_idx = 1
    if is_np_op:
        doc_str_idx = 2
    if arr_name:
        code.append("""
def %s(*%s, **kwargs):"""%(func_name, arr_name))
        if not signature_only:
            code.append("""
    ndargs = []
    for i in {}:
        assert isinstance(i, NDArrayBase), \\
            "Positional arguments must have NDArray type, " \\
            "but got %s"%str(i)
        ndargs.append(i)""".format(arr_name))
            if dtype_name is not None:
                code.append("""
    if '%s' in kwargs:
        kwargs['%s'] = get_dtype_name(kwargs['%s'])"""%(dtype_name, dtype_name, dtype_name))
            if op_name == 'Custom':
                code.append("""
    out = kwargs.get('out', None)
    if isinstance(out, NDArrayBase) or isinstance(out, (list, tuple)):
        out = kwargs.pop('out', None)
    else:
        out = None
    param_keys = list(kwargs.keys())
    param_vals = list(kwargs.values())""")
            else:
                code.append("""
    _ = kwargs.pop('name', None)
    out = kwargs.pop('out', None)
    param_keys = list(kwargs.keys())
    param_vals = list(kwargs.values())""")
    else:
        code.append("""
def %s(%s):"""%(func_name, ', '.join(signature)))
        if not signature_only:
            code.append("""
    ndargs = []
    param_keys = list(kwargs.keys())
    param_vals = list(kwargs.values())""")
            # NDArray args
            for name in ndarg_names: # pylint: disable=redefined-argument-from-local
                code.append("""
    if {name} is not None:
        assert isinstance({name}, NDArrayBase), \\
            "Argument {name} must have NDArray type, but got %s"%str({name})
        ndargs.append({name})""".format(name=name))
            # kwargs
            for name in kwarg_names: # pylint: disable=redefined-argument-from-local
                code.append("""
    if %s is not _Null:
        param_keys.append('%s')
        param_vals.append(%s)"""%(name, name, name))
            # dtype
            if dtype_name is not None:
                if is_np_op:
                    code.append("""
    if %s is not _Null and %s is not None:
        param_keys.append('%s')
        param_vals.append(get_dtype_name(%s))"""%(dtype_name, dtype_name, dtype_name, dtype_name))
                else:
                    code.append("""
    if %s is not _Null:
        param_keys.append('%s')
        param_vals.append(get_dtype_name(%s))"""%(dtype_name, dtype_name, dtype_name))

    verify_ndarrays_fn =\
        _verify_all_np_ndarrays.__name__ if is_np_op else _verify_all_legacy_ndarrays.__name__
    if not signature_only:
        code.append("""
    {verify_fn}("{op_name}", "{func_name}", ndargs, out)
        """.format(verify_fn=verify_ndarrays_fn, op_name=op_name, func_name=func_name))
        code.append("""
    return _imperative_invoke_checked(%d, ndargs, param_keys, param_vals, out, %s, %s, %r)"""%(
        handle.value, str(is_np_op), str(output_is_list), op_name))
    else:
        code.append("""
    return (0,)""")

    doc_str_lines = _os.linesep+''.join(['    '+s if s.strip() else s
                                         for s in 'r"""{doc_str}"""'.format(doc_str=doc_str)
                                         .splitlines(True)])
    code.insert(doc_str_idx, doc_str_lines)
    return ''.join(code), doc_str


# pylint: disable=too-many-locals, invalid-name
def _make_ndarray_function(handle, name, func_name):
    """Create a NDArray function from the FunctionHandle."""
    code, doc_str = _generate_ndarray_function_code(handle, name, func_name)

    local = {}
    exec(code, None, local)  # pylint: disable=exec-used
    ndarray_function = local[func_name]
    if name == "_contrib_box_encode":
        generated_function = ndarray_function

        def ndarray_function(*args, **kwargs):
            if len(args) == 2 and "samples" in kwargs:
                refs = args[1]
                if isinstance(refs, NDArrayBase) and 0 in refs.shape:
                    raise ValueError("refs input for box_encode must not be empty; got shape {}".format(
                        refs.shape))
            return generated_function(*args, **kwargs)
    ndarray_function.__name__ = func_name
    ndarray_function.__doc__ = doc_str
    ndarray_function.__module__ = 'mxnet.ndarray'
    return ndarray_function

_init_op_module('mxnet', 'ndarray', _make_ndarray_function)

# Update operator documentation with added float support
# Note that we can only do this after the op module is initialized
# Otherwise the backend operators cannot be found
# pylint: disable=wrong-import-position
from .contrib import adamw_update, mp_adamw_update
from ._internal import _adamw_update, _mp_adamw_update
adamw_update.__doc__ = _adamw_update.__doc__.replace("rescale_grad : NDArray",
                                                     "rescale_grad : NDArray or float")
mp_adamw_update.__doc__ = _mp_adamw_update.__doc__.replace("rescale_grad : NDArray",
                                                           "rescale_grad : NDArray or float")
