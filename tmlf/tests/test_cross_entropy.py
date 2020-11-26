from tmlf.python import workspace, model_builder
import numpy as np
import math

workspace.feed_tensor("pred", np.array([0.25, 0.85], dtype=np.float32))
workspace.feed_tensor("label", np.array([0, 1], dtype=np.float32))
net = model_builder.Net()
net.cross_entropy(["pred", "label"], "loss")
model_builder.run_net(net)
loss = workspace.fetch_tensor("loss")
expected = np.array([
    [-math.log(0.75)],
    [-math.log(0.85)],
], dtype=np.float32)
print(f"expect: {expected}")
print(f"actual: {loss}")
np.testing.assert_almost_equal(expected, loss)
