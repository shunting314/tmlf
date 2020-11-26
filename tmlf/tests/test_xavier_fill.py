from tmlf.python import model_builder
from tmlf.python import tmlf_pybind
import numpy as np

net = model_builder.Net()
net.xavier_fill([], ["out"], shape=(300, 10000))
model_builder.run_net(net)

out = tmlf_pybind.fetch_tensor('out')
out_np = np.array(out)
meanval = out_np.mean()
print(meanval)
assert abs(meanval) <= 0.0001
