from tmlf.python import model_builder
from tmlf.python import tmlf_pybind
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

tmlf_pybind.feed_tensor("feat", tmlf_pybind.Tensor(feat))
tmlf_pybind.feed_tensor("w", tmlf_pybind.Tensor(w))
tmlf_pybind.feed_tensor("b", tmlf_pybind.Tensor(b))

net = model_builder.Net()
net.fc(["feat", "w", "b"], ["out"])
model_builder.run_net(net)

out = np.array(tmlf_pybind.fetch_tensor('out'))
np.testing.assert_array_equal(out, expected)
