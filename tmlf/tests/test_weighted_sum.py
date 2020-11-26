from tmlf.python import model_builder, workspace
import numpy as np

net = model_builder.Net()
workspace.feed_tensor("v0", np.array(
    [ 1, 2], dtype=np.float32,
))
workspace.feed_tensor("w0", np.array(
    [ 0.2], dtype=np.float32,
))
workspace.feed_tensor("v1", np.array(
    [3, 4,], dtype=np.float32,
))
workspace.feed_tensor("w1", np.array(
    [ 0.4], dtype=np.float32,
))
net.weighted_sum(
    ["v0", "w0", "v1", "w1"],
    ["v0"],
)
model_builder.run_net(net)
expected = np.array(
    [1.4, 2.0], dtype=np.float32,
)
np.testing.assert_almost_equal(expected,
    workspace.fetch_tensor("v0").reshape(-1))
