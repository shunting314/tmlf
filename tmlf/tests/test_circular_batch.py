import numpy as np
from tmlf.python import workspace, model_builder

workspace.feed_tensor('X', np.array(
    [
        [2, 3],
        [4, 5],
        [6, 7],
    ], dtype=np.float32))
workspace.feed_tensor('y', np.array(
    [ 0, 1, 0], dtype=np.float32))
workspace.feed_tensor('cursor', np.array(
    [ 0], dtype=np.float32))

net = model_builder.Net()
net.circular_batch(['X', 'y', 'cursor'], ["X_sub", "y_sub", 'cursor'], batch_size=2)

model_builder.run_net(net)
np.testing.assert_almost_equal(workspace.fetch_tensor("X_sub"), np.array(
    [
        [2, 3],
        [4, 5],
    ], dtype=np.float32
))
np.testing.assert_almost_equal(workspace.fetch_tensor("y_sub").reshape(-1), np.array(
    [ 0, 1], dtype=np.float32,
))
np.testing.assert_almost_equal(workspace.fetch_tensor("cursor").reshape(-1), np.array(
    [ 2], dtype=np.float32,
))

model_builder.run_net(net)
np.testing.assert_almost_equal(workspace.fetch_tensor("X_sub"), np.array(
    [
        [ 6, 7],
        [ 2, 3],
    ], dtype=np.float32,
))
np.testing.assert_almost_equal(workspace.fetch_tensor("y_sub").reshape(-1), np.array(
    [ 0, 0], dtype=np.float32,
))
np.testing.assert_almost_equal(workspace.fetch_tensor("cursor").reshape(-1), np.array(
    [ 1], dtype=np.float32,
))
