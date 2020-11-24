from caffe2.python import brew, workspace, optimizer
from caffe2.python.model_helper import ModelHelper
import numpy as np
import click

@click.command()
@click.option(
    "--opt_name",
    default="adam",
    help="Optimizer name",
)
def main(opt_name):
    workspace.FeedBlob('input', np.random.randn(2, 16).astype(np.float32))
    workspace.FeedBlob('label', np.array([0, 1]).astype(np.float32))
    
    helper = ModelHelper("sample_model")
    fc = brew.fc(helper, "input", "fc", dim_in=16, dim_out=8)
    relu = helper.Relu(fc, 'relu')
    fc2 = brew.fc(helper, relu, "fc2", dim_in=8, dim_out=1)
    label_ex = helper.ExpandDims("label", "label_ex", dims=[1])
    xent = helper.SigmoidCrossEntropyWithLogits([fc2, label_ex], 'xent')
    loss = helper.AveragedLoss(xent, 'loss')
    helper.AddGradientOperators([loss])
   
    if opt_name == "manual":
        ONE = helper.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
        LR = helper.param_init_net.ConstantFill([], "LR", shape=[1], value=-0.03)
   
        for param in helper.params:
            param_grad = helper.param_to_grad[param]
            helper.WeightedSum([param, ONE, param_grad, LR], param)
    elif opt_name == "sgd":
        optimizer.build_sgd(helper, 0.03)
    elif opt_name == "adagrad":
        optimizer.build_adagrad(helper, 0.03)
    # caffe2 does not support rowwise adagrad for dense parameters
    # caffe2 seems not have lamb support yet
    elif opt_name == "adam":
        optimizer.build_adam(helper, 0.03)
    else:
        assert False, f"Unsupported optimizer {opt_name}"
    
    workspace.RunNetOnce(helper.param_init_net)
    workspace.RunNetOnce(helper.net)
    
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
