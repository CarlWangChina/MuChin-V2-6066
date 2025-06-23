from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from tensorboardX import SummaryWriter
import os
import unittest
import numpy as np
import caffe2.python.brew as brew
import caffe2.python.cnn as cnn
import Code_for_Experiment.Targeted_Training.audio_quality_screening.encodec.quantization.core_vq
import caffe2.python.model_helper as model_helper
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace
import Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tensorboardX.caffe2_graph as tb
from tensorboardX import x2num
from .expect_reader import compare_proto, write_proto

class Caffe2Test(unittest.TestCase):
    def test_caffe2_np(self):
        workspace.FeedBlob("testBlob", np.random.randn(1, 3, 64, 64).astype(np.float32))
        assert isinstance(x2num.make_np('testBlob'), np.ndarray)

    def test_that_operators_gets_non_colliding_names(self):
        op = caffe2_pb2.OperatorDef()
        op.type = 'foo'
        op.input.extend(['foo'])
        tb._fill_missing_operator_names([op])
        self.assertEqual(op.input[0], 'foo')
        self.assertEqual(op.name, 'foo_1')

    def test_that_replacing_colons_gives_non_colliding_names(self):
        op = caffe2_pb2.OperatorDef()
        op.name = 'foo:0'
        op.input.extend(['foo:0', 'foo$0'])
        shapes = {'foo:0': [1]}
        blob_name_tracker = tb._get_blob_names([op])
        tb._replace_colons(shapes, blob_name_tracker, [op], '$')
        self.assertEqual(op.input[0], 'foo$0')
        self.assertEqual(op.input[1], 'foo$0_1')
        self.assertEqual(op.name, 'foo$0')
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes['foo$0'], [1])
        self.assertEqual(len(blob_name_tracker), 2)
        self.assertEqual(blob_name_tracker['foo$0'], 'foo:0')
        self.assertEqual(blob_name_tracker['foo$0_1'], 'foo$0')

    def test_that_adding_gradient_scope_does_no_fancy_renaming(self):
        op = caffe2_pb2.OperatorDef()
        op.name = 'foo_grad'
        op.input.extend(['foo_grad', 'foo_grad_1'])
        shapes = {'foo_grad': [1]}
        blob_name_tracker = tb._get_blob_names([op])
        tb._add_gradient_scope(shapes, blob_name_tracker, [op])
        self.assertEqual(op.input[0], 'GRADIENTS/foo_grad')
        self.assertEqual(op.input[1], 'GRADIENTS/foo_grad_1')
        self.assertEqual(op.name, 'GRADIENTS/foo_grad')
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes['GRADIENTS/foo_grad'], [1])
        self.assertEqual(len(blob_name_tracker), 2)
        self.assertEqual(blob_name_tracker['GRADIENTS/foo_grad'], 'foo_grad')
        self.assertEqual(blob_name_tracker['GRADIENTS/foo_grad_1'], 'foo_grad_1')

    def test_that_auto_ssa_gives_non_colliding_names(self):
        op1 = caffe2_pb2.OperatorDef()
        op1.output.extend(['foo'])
        op2 = caffe2_pb2.OperatorDef()
        op2.input.extend(['foo'])
        op2.output.extend(['foo'])
        op2.output.extend(['foo_1'])
        shapes = {'foo': [1], 'foo_1': [2]}
        blob_name_tracker = tb._get_blob_names([op1, op2])
        tb._convert_to_ssa(shapes, blob_name_tracker, [op1, op2])
        self.assertEqual(op1.output[0], 'foo')
        self.assertEqual(op2.input[0], 'foo')
        self.assertEqual(op2.output[0], 'foo_1')
        self.assertEqual(op2.output[1], 'foo_1_1')
        self.assertEqual(len(shapes), 3)
        self.assertEqual(shapes['foo'], [1])
        self.assertEqual(shapes['foo_1'], [1])
        self.assertEqual(shapes['foo_1_1'], [2])
        self.assertEqual(len(blob_name_tracker), 3)
        self.assertEqual(blob_name_tracker['foo'], 'foo')
        self.assertEqual(blob_name_tracker['foo_1'], 'foo')
        self.assertEqual(blob_name_tracker['foo_1_1'], 'foo_1')

    def test_renaming_tensorflow_style(self):
        op1 = caffe2_pb2.OperatorDef()
        op1.input.extend(['foo_w'])
        op1.output.extend(['foo_w_2'])
        op2 = caffe2_pb2.OperatorDef()
        op2.input.extend(['foo_bn'])
        op2.output.extend(['foo_bn_2'])
        op3 = caffe2_pb2.OperatorDef()
        op3.input.extend(['foo_b'])
        op3.output.extend(['foo_b_2'])
        op4 = caffe2_pb2.OperatorDef()
        op4.input.extend(['foo_s'])
        op4.output.extend(['foo_s_2'])
        op5 = caffe2_pb2.OperatorDef()
        op5.input.extend(['foo_sum'])
        op5.output.extend(['foo_sum_2'])
        op6 = caffe2_pb2.OperatorDef()
        op6.input.extend(['foo_branch'])
        op6.input.extend(['test_branch_2'])
        op6.output.extend(['foo_branch_3'])
        op6.output.extend(['test_branch4'])
        shapes = {
            'foo_w': [1], 'foo_w_2': [2], 'foo_bn': [3], 'foo_bn_2': [4],
            'foo_b': [5], 'foo_b_2': [6], 'foo_s': [7], 'foo_s_2': [8],
            'foo_sum': [9], 'foo_sum_2': [10], 'foo_branch': [11],
            'test_branch_2': [12], 'foo_branch_3': [13], 'test_branch4': [14],
        }
        ops = [op1, op2, op3, op4, op5, op6]
        blob_name_tracker = tb._get_blob_names(ops)
        tb._rename_tensorflow_style(shapes, blob_name_tracker, ops)
        self.assertEqual(blob_name_tracker['foo/weight'], 'foo_w')
        self.assertEqual(blob_name_tracker['foo/weight_2'], 'foo_w_2')
        self.assertEqual(blob_name_tracker['foo/batchnorm'], 'foo_bn')
        self.assertEqual(blob_name_tracker['foo/batchnorm_2'], 'foo_bn_2')
        self.assertEqual(blob_name_tracker['foo/bias'], 'foo_b')
        self.assertEqual(blob_name_tracker['foo/bias_2'], 'foo_b_2')
        self.assertEqual(blob_name_tracker['foo/scale'], 'foo_s')
        self.assertEqual(blob_name_tracker['foo/scale_2'], 'foo_s_2')
        self.assertEqual(blob_name_tracker['foo/sum'], 'foo_sum')
        self.assertEqual(blob_name_tracker['foo/sum_2'], 'foo_sum_2')
        self.assertEqual(blob_name_tracker['foo/branch'], 'foo_branch')
        self.assertEqual(blob_name_tracker['test/branch_2'], 'test_branch_2')
        self.assertEqual(blob_name_tracker['foo/branch_3'], 'foo_branch_3')
        self.assertEqual(blob_name_tracker['test/branch4'], 'test_branch4')
        self.assertEqual(shapes['foo/weight'], [1])
        self.assertEqual(shapes['foo/batchnorm_2'], [4])
        self.assertEqual(shapes['foo/sum'], [9])
        self.assertEqual(shapes['test/branch_2'], [12])
        self.assertEqual(op1.input[0], 'foo/weight')
        self.assertEqual(op1.output[0], 'foo/weight_2')
        self.assertEqual(op2.input[0], 'foo/batchnorm')
        self.assertEqual(op2.output[0], 'foo/batchnorm_2')
        self.assertEqual(op3.input[0], 'foo/bias')
        self.assertEqual(op3.output[0], 'foo/bias_2')
        self.assertEqual(op4.input[0], 'foo/scale')
        self.assertEqual(op4.output[0], 'foo/scale_2')
        self.assertEqual(op5.input[0], 'foo/sum')
        self.assertEqual(op5.output[0], 'foo/sum_2')
        self.assertEqual(op6.input[0], 'foo/branch')
        self.assertEqual(op6.input[1], 'test/branch_2')
        self.assertEqual(op6.output[0], 'foo/branch_3')
        self.assertEqual(op6.output[1], 'test/branch4')

    def test_filter_ops(self):
        op1 = caffe2_pb2.OperatorDef()
        op1.input.extend(['remove_this'])
        op1.output.extend(['random_output'])
        op2 = caffe2_pb2.OperatorDef()
        op2.input.extend(['leave_this'])
        op2.output.extend(['leave_this_also'])
        op3 = caffe2_pb2.OperatorDef()
        op3.input.extend(['random_input'])
        op3.output.extend(['remove_this_also'])
        def filter_fn(blob):
            return 'remove' not in str(blob)
        op_set1 = [op1, op2, op3]
        op_set2 = [op1, op2, op3]
        result_ops1 = tb._filter_ops(op_set1, filter_fn, True)
        new_op1, new_op2 = result_ops1[0], result_ops1[1]
        self.assertEqual(len(new_op1.input), 0)
        self.assertEqual(new_op1.output, ['random_output'])
        self.assertEqual(new_op2.input, ['leave_this'])
        self.assertEqual(new_op2.output, ['leave_this_also'])
        self.assertEqual(len(result_ops1), 2)
        result_ops2 = tb._filter_ops(op_set2, filter_fn, False)
        self.assertEqual(result_ops2, op_set2)

    def test_simple_cnnmodel(self):
        model = cnn.CNNModelHelper("NCHW", name="overfeat")
        workspace.FeedBlob("data", np.random.randn(1, 3, 64, 64).astype(np.float32))
        workspace.FeedBlob("label", np.random.randn(1, 1000).astype(np.int))
        with model_helper.NameScope("conv1"):
            conv1 = model.Conv("data", "conv1", 3, 96, 11, stride=4)
            relu1 = model.Relu(conv1, conv1)
            pool1 = model.MaxPool(relu1, "pool1", kernel=2, stride=2)
        with model_helper.NameScope("classifier"):
            fc = model.FC(pool1, "fc", 4096, 1000)
            pred = model.Softmax(fc, "pred")
            xent = model.LabelCrossEntropy([pred, "label"], "xent")
            loss = model.AveragedLoss(xent, "loss")
        blob_name_tracker = {}
        graph = tb.model_to_graph_def(
            model,
            blob_name_tracker=blob_name_tracker,
            shapes={},
            show_simplified=False,
        )
        compare_proto(graph, self)

    def test_simple_model(self):
        model = model_helper.ModelHelper(name="mnist")
        workspace.FeedBlob("data", np.random.randn(1, 3, 64, 64).astype(np.float32))
        workspace.FeedBlob("label", np.random.randn(1, 1000).astype(np.int))
        with model_helper.NameScope("conv1"):
            conv1 = brew.conv(model, "data", 'conv1', dim_in=1, dim_out=20, kernel=5)
            pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
            conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=100, kernel=5)
            pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
        with model_helper.NameScope("classifier"):
            fc3 = brew.fc(model, pool2, 'fc3', dim_in=100 * 4 * 4, dim_out=500)
            relu = brew.relu(model, fc3, fc3)
            pred = brew.fc(model, relu, 'pred', 500, 10)
            softmax = brew.softmax(model, pred, 'softmax')
            xent = model.LabelCrossEntropy([softmax, "label"], 'xent')
            loss = model.AveragedLoss(xent, "loss")
        model.net.RunAllOnMKL()
        model.param_init_net.RunAllOnMKL()
        model.AddGradientOperators([loss], skip=1)
        blob_name_tracker = {}
        graph = tb.model_to_graph_def(
            model,
            blob_name_tracker=blob_name_tracker,
            shapes={},
            show_simplified=False,
        )
        compare_proto(graph, self)

if __name__ == "__main__":
    unittest.main()