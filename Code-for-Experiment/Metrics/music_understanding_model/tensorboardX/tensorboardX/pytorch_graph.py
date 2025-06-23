import logging
import time
import numpy as np
from collections import OrderedDict
from .proto.attr_value_pb2 import AttrValue
from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.step_stats_pb2 import RunMetadata, StepStats, DeviceStepStats, NodeExecStats, AllocatorMemoryUsed
from .proto.tensor_shape_pb2 import TensorShapeProto
from .proto.versions_pb2 import VersionDef
from Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tensorboardX.proto_graph import node_proto

methods_OP = ['attributeNames', 'hasMultipleOutputs', 'hasUses', 'inputs', 'kind', 'outputs', 'outputsSize', 'scopeName']
methods_IO = ['node', 'offset', 'debugName']
backward_mode = False

class NodeBase(object):
    def __init__(self, debugName=None, inputs=None, scope=None, tensor_size=None, op_type='UnSpecified', attributes=''):
        self.debugName = debugName
        self.inputs = inputs
        self.tensor_size = tensor_size
        self.kind = op_type
        self.attributes = attributes
        if scope is not None:
            self.scope = scope

    def __repr__(self):
        repr = []
        repr.append(str(type(self)))
        for m in dir(self):
            if '__' not in m:
                repr.append(m + ': ' + str(getattr(self, m)) + str(type(getattr(self, m))))
        return '\n'.join(repr) + '\n\n'

class NodePy(NodeBase):
    def __init__(self, node_cpp, valid_methods):
        super(NodePy, self).__init__(node_cpp)
        valid_methods = valid_methods[:]
        self.inputs = []
        global backward_mode
        for m in valid_methods:
            if m == 'inputs' or m == 'outputs':
                list_of_node = list(getattr(node_cpp, m)())
                io_unique_names = []
                io_tensor_sizes = []
                for n in list_of_node:
                    if backward_mode:
                        io_unique_names.append(n.uniqueName())
                    else:
                        io_unique_names.append(n.debugName())
                    if n.type().kind() == 'CompleteTensorType':
                        io_tensor_sizes.append(n.type().sizes())
                    else:
                        io_tensor_sizes.append(None)
                setattr(self, m, io_unique_names)
                setattr(self, m + 'tensor_size', io_tensor_sizes)
            else:
                if m == 'debugName' and backward_mode:
                    setattr(self, m, getattr(node_cpp, 'uniqueName')())
                else:
                    setattr(self, m, getattr(node_cpp, m)())

class NodePyIO(NodePy):
    def __init__(self, node_cpp, input_or_output=None):
        super(NodePyIO, self).__init__(node_cpp, methods_IO)
        try:
            tensor_size = node_cpp.type().sizes()
        except RuntimeError:
            tensor_size = [1, ]
        self.tensor_size = tensor_size
        self.kind = 'Parameter'
        if input_or_output:
            self.input_or_output = input_or_output
            self.kind = 'IO Node'

class NodePyOP(NodePy):
    def __init__(self, node_cpp):
        super(NodePyOP, self).__init__(node_cpp, methods_OP)
        self.attributes = str({k: node_cpp[k] for k in node_cpp.attributeNames()}).replace("'", ' ')
        self.kind = node_cpp.kind()

class GraphPy(object):
    def __init__(self):
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.unique_name_to_scoped_name = {}
        self.shallowest_scope_name = 'default'
        self.scope_name_appeared = []

    def append(self, x):
        if isinstance(x, NodePyIO):
            self.nodes_io[x.debugName] = x
        if isinstance(x, NodePyOP):
            self.nodes_op.append(x)
            for node_output, outputSize in zip(x.outputs, x.outputstensor_size):
                self.scope_name_appeared.append(x.scopeName)
                self.nodes_io[node_output] = NodeBase(node_output, x.inputs, x.scopeName, outputSize, op_type=x.kind, attributes=x.attributes)

    def printall(self):
        print('all nodes')
        for node in self.nodes_op:
            print(node)
        for key in self.nodes_io:
            print(self.nodes_io[key])

    def find_common_root(self):
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowest_scope_name = fullscope.split('/')[0]

    def populate_namespace_from_OP_to_IO(self):
        for node in self.nodes_op:
            for input_node_id in node.inputs:
                self.unique_name_to_scoped_name[input_node_id] = node.scopeName + '/' + input_node_id
        for key, node in self.nodes_io.items():
            if type(node) == NodeBase:
                self.unique_name_to_scoped_name[key] = node.scope + '/' + node.debugName
            if hasattr(node, 'input_or_output'):
                self.unique_name_to_scoped_name[key] = node.input_or_output + '/' + node.debugName
            if hasattr(node, 'scope'):
                if node.scope == '' and self.shallowest_scope_name:
                    self.unique_name_to_scoped_name[node.debugName] = self.shallowest_scope_name + '/' + node.debugName
        for key, node in self.nodes_io.items():
            self.nodes_io[key].inputs = [self.unique_name_to_scoped_name[node_input_id] for node_input_id in node.inputs]
            if node.debugName in self.unique_name_to_scoped_name:
                self.nodes_io[key].debugName = self.unique_name_to_scoped_name[node.debugName]

    def to_proto(self):
        import Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tests.test_numpy as test_numpy
        nodes = []
        node_stats = []
        for v in self.nodes_io.values():
            nodes.append(node_proto(v.debugName, input=v.inputs, outputsize=v.tensor_size, op=v.kind, attributes=v.attributes))
            if v.tensor_size and len(v.tensor_size) > 0:
                node_stats.append(NodeExecStats(node_name=v.debugName, all_start_micros=int(time.time() * 1e7), all_end_rel_micros=42, memory=[AllocatorMemoryUsed(allocator_name="cpu", total_bytes=int(np.prod(v.tensor_size)) * 4)]))
        return nodes, node_stats

def parse(graph, args=None, omit_useless_nodes=True):
    import torch
    n_inputs = len(args)
    nodes_py = GraphPy()
    for i, node in enumerate(graph.inputs()):
        global backward_mode
        if not backward_mode:
            try:
                node.debugName()
            except:
                backward_mode = True
        if omit_useless_nodes:
            if len(node.uses()) == 0:
                continue
        if i < n_inputs:
            nodes_py.append(NodePyIO(node, 'input'))
        else:
            nodes_py.append(NodePyIO(node))
    for node in graph.nodes():
        nodes_py.append(NodePyOP(node))
    for node in graph.outputs():
        nodes_py.append(NodePyIO(node, 'output'))
    nodes_py.find_common_root()
    nodes_py.populate_namespace_from_OP_to_IO()
    return nodes_py.to_proto()

def graph(model, args, verbose=False, **kwargs):
    import torch
    with torch.onnx.set_training(model, False):
        try:
            trace = torch.jit.trace(model, args)
            graph = trace.graph
        except RuntimeError as e:
            print(e)
            print('Error occurs, No graph saved')
            raise e
    if verbose:
        print(graph)
    list_of_nodes, node_stats = parse(graph, args)
    stepstats = RunMetadata(step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0", node_stats=node_stats)]))
    return GraphDef(node=list_of_nodes, versions=VersionDef(producer=22)), stepstats