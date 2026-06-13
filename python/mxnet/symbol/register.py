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

# pylint: disable=unused-import
"""Register backend ops in mxnet.symbol namespace."""
import os as _os
import ctypes
import numpy as _np

from . import _internal
from .. import name as _name, attribute
from ._internal import SymbolBase, _symbol_creator
from ..base import mx_uint, check_call, _LIB, py_str, SymbolHandle, MXNetError
from ..symbol_doc import _build_doc
from ..base import _Null, _init_op_module, _is_np_op, _output_is_list
from ..name import NameManager
from ..profiler import _current_scope as _profiler_scope
from ..ndarray import get_dtype_name
# pylint: enable=unused-import


def _verify_np_symbol(op_name, func_name, sym):
    """Verify if the sym is a numpy symbol.

    Parameters
    ----------
    op_name : str
        Operator full name registered in backend.
    func_name : str
        Operator name exposed to users. This is usually the name by stripping off
        the prefix of the full operator names registered in backend.
    sym : symbol to be verified
    """
    from .numpy._symbol import _Symbol as np_symbol
    if not isinstance(sym, np_symbol):
        raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. '
                        'This is a numpy operator which can only accept '
                        'MXNet numpy ndarrays, while received a legacy ndarray. '
                        'Please ensure that you have activated numpy semantics by calling '
                        '`npx.set_np()` in your code. If you still see this error with numpy '
                        'semantics activated, please call `as_np_ndarray()` upon the legacy '
                        'ndarray to convert it to an MXNet numpy ndarray, and then feed the '
                        'converted array to this operator.'
                        .format(op_name, func_name))


def _verify_legacy_symbol(op_name, func_name, sym):
    """Verify if the sym is a legacy symbol.

    Parameters
    ----------
    op_name : str
        Operator full name registered in backend.
    func_name : str
        Operator name exposed to users. This is usually the name by stripping off
        the prefix of the full operator names registered in backend.
    sym : symbol to be verified
    """
    from .numpy._symbol import _Symbol as np_symbol
    if isinstance(sym, np_symbol):
        raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. '
                        'This is a legacy operator which can only accept '
                        'legacy ndarrays, while received an MXNet numpy ndarray. '
                        'Please call `as_nd_ndarray()` upon the numpy ndarray to '
                        'convert it to a legacy ndarray, and then feed the converted '
                        'array to this operator.'
                        .format(op_name, func_name))



_SYMBOL_VALIDATION_CLASS = None


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


def _validate_interp_param(interp):
    interp_id = _to_int_param(interp, "interp")
    if interp_id not in (0, 1, 2, 3, 4):
        raise ValueError("Unknown interp method {}".format(interp))
    return interp_id


def _validate_symbol_image_params(op_name, keys, vals):
    def get_param(name, default=None):
        try:
            return vals[keys.index(name)]
        except ValueError:
            return default

    # Symbol-side image-op validation removed (mirrors the ndarray side): it
    # rejected valid inputs and only pre-empted OpenCV's own catchable error for
    # invalid interp. The native ops validate.
    return


def _is_image_validation_error(err):
    message = str(err).lower()
    return "image" in message and any(token in message for token in (
        "dimension", "size", "width", "height", "range", "xrange", "yrange", "crop"))


def _validate_symbol_params(op_name, keys, vals):
    def get_param(name, default=None):
        try:
            return vals[keys.index(name)]
        except ValueError:
            return default

    if op_name in ("_contrib_arange_like", "_npx_arange_like"):
        repeat = get_param("repeat", 1)
        if int(repeat) <= 0:
            raise ValueError("repeat must be positive")

    if op_name in ("_contrib_interleaved_matmul_selfatt_qk",
                   "_npx_interleaved_matmul_selfatt_qk"):
        heads = get_param("heads")
        if heads is not None and int(heads) <= 0:
            raise ValueError("heads must be positive")


def _first_symbol_arg_name(symbol):
    if symbol is None:
        return None
    args = symbol.list_arguments()
    return args[0] if len(args) == 1 else None


def _sequence_validation_spec(op_name, sym_kwargs, keys, vals):
    def get_param(name, default=None):
        try:
            return vals[keys.index(name)]
        except ValueError:
            return default

    if op_name not in ("SequenceMask", "SequenceLast", "SequenceReverse",
                       "_npx_sequence_last", "_npx_sequence_reverse"):
        return None
    if not _is_true_param(get_param("use_sequence_length", False)):
        return None
    data_name = _first_symbol_arg_name(sym_kwargs.get("data"))
    length_name = _first_symbol_arg_name(sym_kwargs.get("sequence_length"))
    if data_name is None or length_name is None:
        return None
    return {
        "kind": "sequence",
        "op_name": op_name,
        "data_name": data_name,
        "length_name": length_name,
        "axis": get_param("axis", 0),
    }


def _symbol_validation_spec(op_name, sym_kwargs, keys, vals):
    spec = {
        "op_name": op_name,
        "image": op_name in ("_image_resize", "_image_crop", "_image_random_crop",
                             "_image_random_resized_crop"),
        "sequence": _sequence_validation_spec(op_name, sym_kwargs, keys, vals),
        "box_encode_refs": None,
    }
    if op_name == "_contrib_box_encode":
        spec["box_encode_refs"] = _first_symbol_arg_name(sym_kwargs.get("refs"))
    if spec["image"] or spec["sequence"] is not None or spec["box_encode_refs"] is not None:
        return spec
    return None


def _array_values(array):
    return array.asnumpy() if hasattr(array, "asnumpy") else _np.asarray(array)


def _validate_bound_sequence_length(spec, executor, kwargs):
    arg_dict = executor.arg_dict
    data = kwargs.get(spec["data_name"], arg_dict.get(spec["data_name"]))
    sequence_length = kwargs.get(spec["length_name"], arg_dict.get(spec["length_name"]))
    if data is None or sequence_length is None:
        return
    axis = _to_int_param(spec["axis"], "axis")
    op_name = spec["op_name"]
    if op_name in ("SequenceReverse", "_npx_sequence_reverse") and axis != 0:
        raise ValueError("SequenceReverse only supports axis 0")
    if axis not in (0, 1):
        raise ValueError("{} axis must be 0 or 1".format(op_name))
    lengths = _array_values(sequence_length)
    if op_name == "SequenceMask":
        max_length = data.shape[axis]
        if lengths.size and ((lengths < 0).any() or (lengths > max_length).any()):
            raise ValueError("sequence_length values must be in range [0, {}]".format(max_length))
        return
    expected_batch = data.shape[0] if axis else data.shape[1]
    max_length = data.shape[axis]
    if len(sequence_length.shape) != 1 or sequence_length.shape[0] != expected_batch:
        raise ValueError("sequence_length shape must be ({},), got {}".format(
            expected_batch, sequence_length.shape))
    if lengths.size and ((lengths <= 0).any() or (lengths > max_length).any()):
        raise ValueError("sequence_length values must be in range [1, {}]".format(max_length))


def _attach_executor_validation(executor, spec):
    sequence_spec = spec.get("sequence")
    if sequence_spec is None:
        return executor
    forward = executor.forward

    def checked_forward(is_train=False, **kwargs):
        _validate_bound_sequence_length(sequence_spec, executor, kwargs)
        return forward(is_train=is_train, **kwargs)

    executor.forward = checked_forward
    return executor


def _get_symbol_validation_class():
    global _SYMBOL_VALIDATION_CLASS
    if _SYMBOL_VALIDATION_CLASS is None:
        from .symbol import Symbol  # pylint: disable=import-outside-toplevel

        class _ValidationSymbol(Symbol):
            __slots__ = ["_mxnet_validation_spec"]

            def infer_shape(self, *args, **kwargs):
                try:
                    return super(_ValidationSymbol, self).infer_shape(*args, **kwargs)
                except MXNetError as err:
                    if (self._mxnet_validation_spec.get("image") and
                            _is_image_validation_error(err)):
                        raise ValueError(str(err)) from None
                    raise

            def infer_shape_partial(self, *args, **kwargs):
                try:
                    return super(_ValidationSymbol, self).infer_shape_partial(*args, **kwargs)
                except MXNetError as err:
                    if (self._mxnet_validation_spec.get("image") and
                            _is_image_validation_error(err)):
                        raise ValueError(str(err)) from None
                    raise

            def _simple_bind(self, *args, **kwargs):
                refs_name = self._mxnet_validation_spec.get("box_encode_refs")
                if refs_name is not None:
                    refs_shape = kwargs.get(refs_name)
                    if refs_shape is not None and 0 in refs_shape:
                        raise ValueError("refs input for box_encode must not be empty; got shape {}".format(
                            refs_shape))
                try:
                    executor = super(_ValidationSymbol, self)._simple_bind(*args, **kwargs)
                except MXNetError as err:
                    if (self._mxnet_validation_spec.get("image") and
                            _is_image_validation_error(err)):
                        raise ValueError(str(err)) from None
                    raise
                return _attach_executor_validation(executor, self._mxnet_validation_spec)

        _SYMBOL_VALIDATION_CLASS = _ValidationSymbol
    return _SYMBOL_VALIDATION_CLASS


def _wrap_validation_symbol(symbol, spec):
    cls = _get_symbol_validation_class()
    handle = SymbolHandle()
    check_call(_LIB.MXShallowCopySymbol(symbol.handle, ctypes.byref(handle)))
    wrapped = cls(handle)
    wrapped._mxnet_validation_spec = spec
    return wrapped


def _symbol_creator_checked(handle, sym_args, sym_kwargs, keys, vals, name,
                            is_np_op, output_is_list, op_name):
    _validate_symbol_image_params(op_name, keys, vals)
    _validate_symbol_params(op_name, keys, vals)
    symbol = _symbol_creator(handle, sym_args, sym_kwargs, keys, vals,
                             name, is_np_op, output_is_list)
    spec = _symbol_validation_spec(op_name, sym_kwargs, keys, vals)
    if spec is not None:
        return _wrap_validation_symbol(symbol, spec)
    return symbol

def _generate_symbol_function_code(handle, op_name, func_name, signature_only=False):
    """Generate function for symbol op by handle and function name."""
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
    #signature.append('is_train=False')
    signature.append('name=None')
    signature.append('attr=None')
    signature.append('out=None')
    signature.append('**kwargs')
    signature = ndsignature + signature

    is_np_op = _is_np_op(op_name)
    output_is_list = _output_is_list(op_name)
    verify_symbol_fn = _verify_np_symbol.__name__ if is_np_op else _verify_legacy_symbol.__name__
    code = []
    if arr_name:
        code.append("""
def %s(*%s, **kwargs):"""%(func_name, arr_name))
        if not signature_only:
            code.append("""
    sym_args = []
    for i in {}:
        assert isinstance(i, SymbolBase), \\
            "Positional arguments must be Symbol instances, " \\
            "but got %s"%str(i)
        {}('{}', '{}', i)
        sym_args.append(i)""".format(arr_name, verify_symbol_fn, op_name, func_name))
            if dtype_name is not None:
                code.append("""
    if '%s' in kwargs:
        kwargs['%s'] = get_dtype_name(kwargs['%s'])"""%(dtype_name, dtype_name, dtype_name))
            code.append("""
    attr = kwargs.get('attr', None)
    if isinstance(attr, SymbolBase):
        attr = None
    else:
        attr = kwargs.pop('attr', None)
    kwargs.update(attribute.current().get(attr))
    name = kwargs.get('name', None)
    if isinstance(name, SymbolBase):
        name = None
    else:
        name = kwargs.pop('name', None)
    name = _name.current().get(name, '%s')
    if not isinstance(kwargs.get('out', None), SymbolBase):
        _ = kwargs.pop('out', None)
    if not sym_args and "op_type" not in kwargs and "data" in kwargs and "arg0" not in kwargs and isinstance(kwargs["data"], SymbolBase):
        kwargs["arg0"] = kwargs.pop("data")
    if not sym_args and "op_type" not in kwargs and "weight" in kwargs and "arg1" not in kwargs and isinstance(kwargs["weight"], SymbolBase):
        kwargs["arg1"] = kwargs.pop("weight")
    keys = []
    vals = []
    sym_kwargs = dict()
    for k, v in kwargs.items():
        if isinstance(v, SymbolBase):
            sym_kwargs[k] = v
            %s('%s', '%s', v)
        else:
            keys.append(k)
            vals.append(v)"""%(func_name.lower(), verify_symbol_fn, op_name, func_name))
            if key_var_num_args: # pylint: disable=using-constant-test
                code.append("""
    if '%s' not in kwargs:
        keys.append('%s')
        vals.append(len(sym_args) + len(sym_kwargs))"""%(
            key_var_num_args, key_var_num_args))

            code.append("""
    if 'profiler_scope' not in keys:
        keys.append('profiler_scope')
        vals.append(_profiler_scope.get())
    return _symbol_creator_checked(%d, sym_args, sym_kwargs, keys, vals, name, %s, %s, %r)"""%(
                handle.value, str(is_np_op), str(output_is_list), op_name))
    else:
        code.append("""
def %s(%s):"""%(func_name, ', '.join(signature)))
        if not signature_only:
            code.append("""
    kwargs.update(attribute.current().get(attr))
    sym_kwargs = dict()
    _keys = []
    _vals = []
    for _k, _v in kwargs.items():
        if isinstance(_v, SymbolBase):
            sym_kwargs[_k] = _v
            {}('{}', '{}', _v)
        else:
            _keys.append(_k)
            _vals.append(_v)""".format(verify_symbol_fn, op_name, func_name))
            # NDArray args
            for name in ndarg_names: # pylint: disable=redefined-argument-from-local
                code.append("""
    if {name} is not None:
        assert isinstance({name}, SymbolBase), \\
            "Argument {name} must be Symbol instances, but got %s"%str({name})
        sym_kwargs['{name}'] = {name}""".format(name=name))
                code.append("""
        {}('{}', '{}', {name})
                """.format(verify_symbol_fn, op_name, func_name, name=name))
            # kwargs
            for name in kwarg_names: # pylint: disable=redefined-argument-from-local
                code.append("""
    if %s is not _Null:
        _keys.append('%s')
        _vals.append(%s)"""%(name, name, name))
            # dtype
            if dtype_name is not None:
                if is_np_op:
                    code.append("""
    if %s is not _Null and %s is not None:
        _keys.append('%s')
        _vals.append(get_dtype_name(%s))"""%(dtype_name, dtype_name, dtype_name, dtype_name))
                else:
                    code.append("""
    if %s is not _Null:
        _keys.append('%s')
        _vals.append(get_dtype_name(%s))"""%(dtype_name, dtype_name, dtype_name))

            code.append("""
    name = _name.current().get(name, "%s")
    if "profiler_scope" not in _keys:
        _keys.append("profiler_scope")
        _vals.append(_profiler_scope.get())
    return _symbol_creator_checked(%d, None, sym_kwargs, _keys, _vals, name, %s, %s, %r)"""%(
        func_name.lower(), handle.value, str(is_np_op), str(output_is_list), op_name))

    if signature_only:
        code.append("""
    return (0,)""")

    doc_str_lines = _os.linesep+''.join(['    '+s if s.strip() else s
                                         for s in 'r"""{doc_str}"""'.format(doc_str=doc_str)
                                         .splitlines(True)])
    code.insert(1, doc_str_lines)
    return ''.join(code), doc_str


def _make_symbol_function(handle, name, func_name):
    """Create a symbol function by handle and function name."""
    code, doc_str = _generate_symbol_function_code(handle, name, func_name)

    local = {}
    exec(code, None, local)  # pylint: disable=exec-used
    symbol_function = local[func_name]
    symbol_function.__name__ = func_name
    symbol_function.__doc__ = doc_str
    symbol_function.__module__ = 'mxnet.symbol'
    return symbol_function

_init_op_module('mxnet', 'symbol', _make_symbol_function)

# Update operator documentation with added float support
# Note that we can only do this after the op module is initialized
# Otherwise the backend operators cannot be found
# pylint: disable=wrong-import-position
from .contrib import adamw_update, mp_adamw_update
from ._internal import _adamw_update, _mp_adamw_update
adamw_update.__doc__ = _adamw_update.__doc__.replace("rescale_grad : Symbol",
                                                     "rescale_grad : Symbol or float")
mp_adamw_update.__doc__ = _mp_adamw_update.__doc__.replace("rescale_grad : Symbol",
                                                           "rescale_grad : Symbol or float")
