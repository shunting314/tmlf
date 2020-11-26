from tmlf.python import tmlf_pybind
import numpy as np

def fetch_tensor(name):
    return np.array(tmlf_pybind.fetch_tensor(name))

# convert single dimension np array to a column vector with dim (x * 1)
def feed_tensor(name, ar):
    if len(ar.shape) == 1:
        ar = ar.reshape(ar.shape[0], 1)
    tmlf_pybind.feed_tensor(name, tmlf_pybind.Tensor(ar))
