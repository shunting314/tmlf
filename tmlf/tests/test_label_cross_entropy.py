import numpy as np
from tmlf.python import workspace, model_builder
import math

workspace.feed_tensor("pred", np.array(
    [
        [ 0.1, 0.2, 0.7, ],
        [ 0.2, 0.1, 0.7, ],
        [ 0.3, 0.4, 0.3, ],
    ], dtype=np.float32,
))
workspace.feed_tensor("label", np.array(
    [
        2, 0, 1,
    ], dtype=np.float32,
))

net = model_builder.Net()
net.label_cross_entropy(['pred', 'label'], 'xent')
model_builder.run_net(net)
xent = workspace.fetch_tensor('xent').reshape(-1)
xent_expected = np.array(
    [ -math.log(0.7), -math.log(0.2), -math.log(0.4), ], dtype=np.float32,
)
np.testing.assert_almost_equal(xent_expected, xent)

# backward
net.add_backward_ops()
workspace.feed_tensor("xent_grad", np.array(
    [ 0.2, 0.3, 0.4 ], dtype=np.float32,
))
model_builder.run_net(net)
pred_grad = workspace.fetch_tensor("pred_grad")
pred_grad_expected = np.array(
    [
        [ 0, 0, -0.2 / 0.7],
        [-0.3 / 0.2, 0, 0],
        [0, -0.4 / 0.4, 0],
    ], dtype=np.float32,
)
np.testing.assert_almost_equal(pred_grad_expected, pred_grad)
