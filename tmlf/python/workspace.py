from tmlf.python import tmlf_pybind
import numpy as np

def fetch_tensor(name):
    return np.array(tmlf_pybind.fetch_tensor(name))

def feed_tensor(name, ar):
    tmlf_pybind.feed_tensor(name, tmlf_pybind.Tensor(ar))
