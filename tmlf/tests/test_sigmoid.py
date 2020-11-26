from tmlf.python import workspace
import numpy as np
from tmlf.python import model_builder
import math

workspace.feed_tensor("in", np.array(
    [ -0.25, 0, 0.25], np.float32,
))
net = model_builder.Net()
net.sigmoid("in", "out")
model_builder.run_net(net)
out = workspace.fetch_tensor("out").reshape(-1)
aux = 1.0 / (1 + math.exp(0.25)); # for -0.25
np.testing.assert_array_equal(out, np.array([aux, 0.5, 1 - aux], np.float32))

# backward
net.add_backward_ops()
workspace.feed_tensor("out_grad", np.array(
    [2, 3, 4],
    dtype=np.float32))
model_builder.run_net(net)
in_grad = workspace.fetch_tensor("in_grad").reshape(-1)
expected_grad = np.array(
    [2 * aux * (1 - aux), 3 * 0.25, 4 * aux * (1 - aux)],
    dtype=np.float32)
np.testing.assert_almost_equal(expected_grad, in_grad)
