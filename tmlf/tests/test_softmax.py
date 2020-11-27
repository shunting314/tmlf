from tmlf.python import workspace, model_builder
import numpy as np
import math

workspace.feed_tensor("sm_in", np.array(
    [
        [ 3, -5, 8],
        [-6, 4, 7],
    ], dtype=np.float32))
net = model_builder.Net()
net.softmax("sm_in", "sm_out")
model_builder.run_net(net)
sm_out = workspace.fetch_tensor("sm_out")
exp_sum1 = math.exp(3) + math.exp(-5) + math.exp(8)
exp_sum2 = math.exp(-6) + math.exp(4) + math.exp(7)
sm_out_expected = np.array(
    [
        [math.exp(3) / exp_sum1, math.exp(-5) / exp_sum1, math.exp(8) / exp_sum1],
        [math.exp(-6) / exp_sum2, math.exp(4) / exp_sum2, math.exp(7) / exp_sum2],
    ], dtype=np.float32)
np.testing.assert_almost_equal(sm_out_expected, sm_out)

# backward
net.add_backward_ops()
sm_out_grad = np.array(
    [
        [ 0.2, -0.2, 0.3],
        [ -0.5, 0.4, 0.7],
    ], dtype=np.float32,
)
workspace.feed_tensor("sm_out_grad", sm_out_grad)
model_builder.run_net(net)
sm_in_grad = workspace.fetch_tensor("sm_in_grad")

# calculate the expected grad. Numpy broadcast a vector by row to a matrix
sm_in_grad_expected = (sm_out_grad.transpose() - (sm_out_grad * sm_out).sum(axis=1)).transpose() * sm_out

np.testing.assert_almost_equal(sm_in_grad_expected, sm_in_grad)
