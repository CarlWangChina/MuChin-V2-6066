from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
import logging
import os
import re
import six
from builtins import bytes
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.tensor_shape_pb2 import TensorShapeProto

def _make_unique_name(seen, name, min_version=0):
    assert name is not None
    i = min_version
    x = '%s_%d' % (name, i) if i else name
    while x in seen:
        i += 1
        x = '%s_%d' % (name, i)
    seen.add(x)
    return x

def _rename_tensorflow_style(shapes, blob_name_tracker, ops):
    WEIGHT = re.compile(r"(_w)$")
    WEIGHT_ = re.compile(r"(_w_)")
    BN = re.compile(r"(_bn)$")
    BN_ = re.compile(r"(_bn_)")
    BIAS = re.compile(r"(_b)$")
    BIAS_ = re.compile(r"(_b_)")
    SCALE = re.compile(r"(_s)$")
    SCALE_ = re.compile(r"(_s_)")
    SUM = re.compile(r"(_sum)$")
    SUM_ = re.compile(r"(_sum_)")
    BRANCH = re.compile(r"(_branch)")
    def f(name):
        inter_name = WEIGHT_.sub('/weight_', WEIGHT.sub('/weight', name))
        inter_name = BN_.sub('/batchnorm_', BN.sub('/batchnorm', inter_name))
        inter_name = BIAS_.sub('/bias_', BIAS.sub('/bias', inter_name))
        inter_name = SCALE_.sub('/scale_', SCALE.sub('/scale', inter_name))
        inter_name = SUM_.sub('/sum_', SUM.sub('/sum', inter_name))
        new_name = BRANCH.sub('/branch', inter_name)
        return new_name
    _rename_all(shapes, blob_name_tracker, ops, f)

def _convert_to_ssa(shapes, blob_name_tracker, ops):
    ir = core.IR(ops)
    seen = set()
    versioned = {}
    new_shapes = {}
    new_blob_name_tracker = {}
    def ssa_name(name, versions):
        assert name in versions
        version = versions[name]
        if (name, version) in versioned:
            return versioned[(name, version)]
        new_name = _make_unique_name(seen, name, min_version=version)
        versioned[(name, version)] = new_name
        if name in shapes:
            new_shapes[new_name] = shapes[name]
        if blob_name_tracker and name in blob_name_tracker:
            new_blob_name_tracker[new_name] = blob_name_tracker[name]
        return new_name
    for (op, ssa) in zip(ops, ir.ssa):
        assert op is ssa.op
        inputs = list(op.input)
        outputs = list(op.output)
        del op.input[:]
        del op.output[:]
        op.input.extend(ssa_name(name, ssa.in_versions) for name in inputs)
        op.output.extend(ssa_name(name, ssa.out_versions) for name in outputs)
    shapes.clear()
    shapes.update(new_shapes)
    if blob_name_tracker:
        blob_name_tracker.clear()
        blob_name_tracker.update(new_blob_name_tracker)

def _get_blob_names(ops):
    names = set()
    for op in ops:
        names.update(op.input)
        names.update(op.output)
    return {name: name for name in names}

def _remap_keys(old_dict, rename_fn):
    new_dict = {rename_fn(key): value for key, value in six.iteritems(old_dict)}
    old_dict.clear()
    old_dict.update(new_dict)

def _rename_all(shapes, blob_name_tracker, ops, rename_fn):
    seen = set()
    renamed = {}
    def g(name):
        if name is None:
            return None
        if name in renamed:
            return renamed[name]
        new_name = _make_unique_name(seen, rename_fn(name))
        renamed[name] = new_name
        return new_name
    for op in ops:
        inputs = list(op.input)
        outputs = list(op.output)
        del op.input[:]
        del op.output[:]
        op.input.extend(g(name) for name in inputs)
        op.output.extend(g(name) for name in outputs)
    _remap_keys(shapes, g)
    if blob_name_tracker:
        _remap_keys(blob_name_tracker, g)
    seen.clear()
    renamed.clear()
    for op in ops:
        op.name = g(op.name)

def _add_gradient_scope(shapes, blob_name_tracker, ops):
    def f(name):
        if '_grad' in name:
            return 'GRADIENTS/{}'.format(name)
        else:
            return name
    _rename_all(shapes, blob_name_tracker, ops, f)

def _replace_colons(shapes, blob_name_tracker, ops, repl):
    def f(name):
        return name.replace(':', repl)
    _rename_all(shapes, blob_name_tracker, ops, f)

def _fill_missing_operator_names(ops):
    seen = set()
    for op in ops:
        seen.update(op.input)
        seen.update(op.output)
    for op in ops:
        if op.name:
            name = op.name
        elif op.output or op.input:
            name_list = [os.path.dirname(name) for name in op.output or op.input]
            scope = os.path.commonprefix(name_list)
            name = os.path.join(scope, op.type)
        else:
            name = op.type
        assert(name)
        op.name = _make_unique_name(seen, name)

def _tf_device(device_option):
    if not device_option.HasField("device_type"):
        return ""
    if device_option.device_type == caffe2_pb2.CPU or device_option.device_type == caffe2_pb2.MKLDNN:
        return "/cpu:*"
    if device_option.device_type == caffe2_pb2.CUDA:
        return "/gpu:{}".format(device_option.device_id)
    raise Exception("Unhandled device", device_option)

def _add_tf_shape(attr_dict, ints):
    shape_proto = TensorShapeProto()
    for i in ints:
        dim = TensorShapeProto.Dim()
        dim.size = i
        shape_proto.dim.extend([dim])
    attr_dict['_output_shapes'].list.shape.extend([shape_proto])

def _set_tf_attr(attr_dict, arg):
    k = arg.name
    if k == 'shape' and arg.ints:
        _add_tf_shape(attr_dict, arg.ints)
        return
    if arg.HasField("f"):
        attr_dict[k].f = arg.f
        return
    if arg.HasField("i"):
        attr_dict[k].i = arg.i
        return
    if arg.HasField("s"):
        attr_dict[k].s = (
            arg.s if isinstance(arg.s, bytes) else str(arg.s).encode('utf-8')
        )
        return
    if arg.floats:
        attr_dict[k].list.f.extend(arg.floats)
        return
    if arg.ints:
        attr_dict[k].list.i.extend(arg.ints)
        return
    if arg.strings:
        attr_dict[k].list.s.extend(
            s if isinstance(s, bytes) else str(s).encode('utf-8')
            for s in arg.strings
        )
        return
    attr_dict[k].list.s.extend([])

def _operator_to_node(shapes, op):
    assert op.name, op
    n = NodeDef()
    n.name = op.name
    n.input.extend(op.input)
    n.op = op.type
    n.device = _tf_device(op.device_option)
    if shapes:
        for output in op.output:
            if output not in shapes:
                break
            _add_tf_shape(n.attr, shapes[output])
    for arg in op.arg:
        _set_tf_attr(n.attr, arg)
    return n

def _operator_to_node_simp(op, inter_blobs, seen):
    assert op
    nodes = []
    outputs = [o for o in op.output if o not in inter_blobs]
    seen.update(outputs)
    len_outputs = len(outputs)
    if len_outputs == 1:
        n = NodeDef()
        n.name = outputs[0]
        n.input.extend(op.input)
        n.op = op.type
        n.device = _tf_device(op.device_option)
        for arg in op.arg:
            _set_tf_attr(n.attr, arg)
        nodes.append(n)
    elif len_outputs > 1:
        if op.name:
            name = op.name
        else:
            name_list = [name for name in outputs]
            scope = os.path.commonprefix(name_list)
            name = os.path.join(scope, op.type)
        assert(name)
        op.name = _make_unique_name(seen, name)
        device = _tf_device(op.device_option)
        for output in outputs:
            n = NodeDef()
            n.name = output
            n.input.extend([op.name])
            n.op = 'Blob'
            n.device = device
            nodes.append(n)
        n = NodeDef()
        n.name = op.name
        n.input.extend(op.input)
        n.op = op.type
        n.device = device
        for arg in op.arg:
            _set_tf_attr(n.attr, arg)
        nodes.append(n)
    return nodes

def _blob_to_node(producing_ops, shapes, name):
    assert name
    n = NodeDef()
    n.name = name
    produced_by = producing_ops.get(name, [])
    if len(produced_by) > 0:
        n.op = 'Blob'
    else:
        n.op = 'Placeholder'
    n.input.extend('%s:%d' % (p_op.name, i) for p_op, i in produced_by)
    if produced_by:
        device = produced_by[0][0].device_option
        if (all(producer[0].device_option == device for producer in produced_by)):
            n.device = _tf_device(device)
    if shapes and name in shapes:
        _add_tf_shape(n.attr, shapes[name])
    return n

def _clear_debug_info(ops, perform_clear):
    if not perform_clear:
        return
    for op in ops:
        if op.HasField('debug_info'):
            op.ClearField('debug_info')

def _check_if_forward(blob):
    return (blob.find('__m') < 0 or blob.find('grad') < 0)

def _check_if_cpu(blob):
    return not blob.startswith('_gpu')

def _compute_in_out(ops):
    in_blobs = set()
    out_blobs = set()
    for op in ops:
        for input_blob in op.input:
            in_blobs.add(input_blob)
        for output_blob in op.output:
            out_blobs.add(output_blob)
    input_blobs = list(in_blobs.difference(out_blobs))
    output_blobs = list(out_blobs.difference(in_blobs))
    inter_blobs = {b for b in output_blobs if b.startswith('_')}
    output_blobs = [b for b in output_blobs if b not in inter_blobs]
    return input_blobs, inter_blobs, output_blobs

def _filter_ops(ops, filter_fn, perform_filter):
    if not perform_filter:
        return ops
    new_ops = []
    for op in ops:
        inputs = list(op.input)
        outputs = list(op.output)
        del op.input[:]
        del op.output[:]
        new_inputs = [i for i in inputs if filter_fn(i)]
        new_outputs = [o for o in outputs if filter_fn(o)]
        if new_outputs:
            op.input.extend(new_inputs)
            op.output.extend(new_outputs)
            new_ops.append(op)
    return new_ops

def _operators_to_graph_def(
    shapes,
    ops,
    colon_replacement='$',
    with_ssa=True,
    with_gradient_scope=True,
    blob_name_tracker=None,
    show_simplified=False,
    custom_rename=None,
):
    if blob_name_tracker is not None:
        blob_name_tracker.clear()
    else:
        blob_name_tracker = {}
    blob_name_tracker.update(_get_blob_names(ops))
    _clear_debug_info(ops, show_simplified)
    ops = _filter_ops(ops, _check_if_forward, show_simplified)
    ops = _filter_ops(ops, _check_if_cpu, show_simplified)
    if custom_rename:
        _rename_all(shapes, blob_name_tracker, ops, custom_rename)
    if colon_replacement:
        _replace_colons(shapes, blob_name_tracker, ops, colon_replacement)
    if with_ssa:
        _convert_to_ssa(shapes, blob_name_tracker, ops)
    if with_gradient_scope:
        _add_gradient_scope(shapes, blob_name_tracker, ops)
    _fill_missing_operator_names(ops)
    if show_simplified:
        _rename_tensorflow_style(shapes, blob_name_tracker, ops)
    producing_ops = {}
    blobs = []
    input_blobs, inter_blobs, _ = _compute_in_out(ops)
    current_graph = GraphDef()
    seen = set(input_blobs)
    for op in ops:
        nodes_from_op = _operator_to_node_simp(op, inter_blobs, seen) if \
            show_simplified else \
            [_operator_to_node(shapes, op)]
        current_graph.node.extend(nodes_from_op)
        for input_blob in op.input:
            blobs.append(input_blob)
        for i, output_blob in enumerate(op.output):
            blobs.append(output_blob)
            producing_ops.setdefault(output_blob, []).append((op, i))
    if show_simplified:
        blobs = input_blobs
    for blob in blobs:
        current_graph.node.extend([_blob_to_node(producing_ops, {}, blob)])
    return current_graph

def _propagate_device_option(net_def):
    if not net_def.HasField("device_option"):
        return
    for op in net_def.op:
        if not op.HasField("device_option"):
            op.device_option.CopyFrom(net_def.device_option)

def _try_get_shapes(nets):
    try:
        shapes, _ = workspace.InferShapesAndTypes(nets)
        return shapes
    except Exception as e:
        logging.warning('Failed to compute shapes: %s', e)
        return {}

def model_to_graph_def(model, **kwargs):
    nets = [model.param_init_net, model.net]
    return nets_to_graph_def(nets, **kwargs)

def nets_to_graph_def(nets, shapes=None, **kwargs):
    shapes = {} if shapes is None else shapes
    nets = [copy.deepcopy(net.Proto()) for net in nets]
    shapes = copy.deepcopy(shapes)
    return protos_to_graph_def(nets, shapes, **kwargs)

def protos_to_graph_def(net_defs, shapes=None, **kwargs):
    for net in net_defs:
        _propagate_device_option(net)
    shapes = copy.deepcopy(shapes or {})
    ops = [op for net_def in net_defs for op in net_def.op]
    return _operators_to_graph_def(shapes, ops, **kwargs)