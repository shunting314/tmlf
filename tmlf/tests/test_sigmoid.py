from tmlf.python import workspace
import numpy as np
from tmlf.python import model_builder
import math

workspace.feed_tensor("in", np.array(
    [[ -0.25, 0, 0.25]], np.float32,
))
net = model_builder.Net()
net.sigmoid("in", "out")
model_builder.run_net(net)
out = workspace.fetch_tensor("out")
aux = 1.0 / (1 + math.exp(0.25)); # for -0.25
np.testing.assert_array_equal(out, np.array([[aux, 0.5, 1 - aux]], np.float32))
