from tmlf.python import workspace
import numpy as np
from tmlf.python import model_builder

workspace.feed_tensor("fc", np.array(
    [[ -1, -0.5, 0.5, 1]], np.float32,
))
net = model_builder.Net()
net.relu("fc", "relu")
model_builder.run_net(net)
out = workspace.fetch_tensor("relu")
np.testing.assert_array_equal(out, np.array([[0, 0, 0.5, 1]], np.float32))
