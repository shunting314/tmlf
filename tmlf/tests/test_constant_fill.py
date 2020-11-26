from tmlf.python import model_builder
from tmlf.python import tmlf_pybind
import numpy as np

net = model_builder.Net()
net.constant_fill([], ["out"], shape=(3, 5), value=3.7)
model_builder.run_net(net)

out = tmlf_pybind.fetch_tensor('out')
out_np = np.array(out)
np.testing.assert_array_equal(out_np, np.full([3, 5], 3.7, dtype=np.float32))
