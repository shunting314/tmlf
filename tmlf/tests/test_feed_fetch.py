from tmlf.python import tmlf_pybind
import numpy as np

ar = np.array([[1, 2], [3, 5]], dtype=np.float32)
tmlf_pybind.feed_tensor('fib', tmlf_pybind.Tensor(ar))
out_ar = np.array(tmlf_pybind.fetch_tensor('fib'))
print(ar)
print(out_ar)
np.testing.assert_array_equal(ar, out_ar)
