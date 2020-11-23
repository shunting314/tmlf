from caffe2.python import brew, workspace
from caffe2.python.model_helper import ModelHelper
import numpy as np

workspace.FeedBlob('input', np.random.randn(2, 16).astype(np.float32))
workspace.FeedBlob('label', np.array([0, 1]).astype(np.float32))

helper = ModelHelper("sample_model")
fc = brew.fc(helper, "input", "fc", dim_in=16, dim_out=8)
relu = helper.Relu(fc, 'relu')
fc2 = brew.fc(helper, relu, "fc2", dim_in=8, dim_out=1)
pred = helper.Sigmoid(fc2, "pred")
pred_sq = helper.Squeeze(pred, "pred_sq", dims=[1])
xent = helper.SigmoidCrossEntropyWithLogits([pred_sq, 'label'], 'xent')
loss = helper.AveragedLoss(xent, 'loss')
helper.AddGradientOperators([loss])

ONE = helper.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
LR = helper.param_init_net.ConstantFill([], "LR", shape=[1], value=-0.03)

for param in helper.params:
    param_grad = helper.param_to_grad[param]
    helper.WeightedSum([param, ONE, param_grad, LR], param)

workspace.RunNetOnce(helper.param_init_net)
workspace.RunNetOnce(helper.net)

import pdb; pdb.set_trace()
