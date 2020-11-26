from tmlf.python import model_builder
from tmlf.python import tmlf_pybind, workspace
import numpy as np

feat = np.array([
  [ 1, 2, 3],
  [ 1, 1, 1],
], dtype=np.float32)
w = np.array([
  [2, 2],
  [2, 2],
  [2, 2],
], dtype=np.float32)
b = np.array([
  [1], [-1],
], dtype=np.float32)
expected = np.array([
  [ 13, 11, ],
  [ 7, 5, ],
], dtype=np.float32)

workspace.feed_tensor("feat", feat)
workspace.feed_tensor("w", w)
workspace.feed_tensor("b", b)

net = model_builder.Net()
net.fc(["feat", "w", "b"], ["out"])
model_builder.run_net(net)

out = workspace.fetch_tensor('out')
np.testing.assert_array_equal(out, expected)

# backward
net.add_backward_ops()
workspace.feed_tensor("out_grad", np.array(
    [
        [ 2, 4, ],
        [ 2, 8, ],
    ],
    dtype=np.float32,
))
model_builder.run_net(net)
feat_grad = workspace.fetch_tensor("feat_grad")
w_grad = workspace.fetch_tensor("w_grad")
b_grad = workspace.fetch_tensor("b_grad").reshape(-1)

# check feat_grad
feat_grad_expected = np.matmul(
    workspace.fetch_tensor("out_grad"),
    w.transpose(),
)
np.testing.assert_almost_equal(feat_grad_expected, feat_grad)

# check w_grad
w_grad_expected = np.matmul(
    feat.transpose(),
    workspace.fetch_tensor("out_grad"),
)
np.testing.assert_almost_equal(w_grad_expected, w_grad)

# check b_grad
b_grad_expected = np.array(
    [ 4, 12],
    dtype=np.float32,
)
np.testing.assert_almost_equal(b_grad_expected, b_grad);
