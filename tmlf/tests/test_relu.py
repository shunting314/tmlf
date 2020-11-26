from tmlf.python import workspace
import numpy as np
from tmlf.python import model_builder, workspace

workspace.feed_tensor("fc", np.array(
    [ -1, -0.5, 0.5, 1], np.float32,
))
net = model_builder.Net()
net.relu("fc", "relu")
model_builder.run_net(net)
out = workspace.fetch_tensor("relu").reshape(-1)
np.testing.assert_array_equal(out, np.array([0, 0, 0.5, 1], np.float32))

# backward
net.add_backward_ops()
workspace.feed_tensor("relu_grad", np.array(
    [ 2, 3, 4, 5], dtype=np.float32))
model_builder.run_net(net)
fc_grad = workspace.fetch_tensor("fc_grad").reshape(-1)
fc_grad_expected = np.array(
    [ 0, 0, 4, 5], dtype=np.float32)
np.testing.assert_almost_equal(fc_grad_expected, fc_grad)
