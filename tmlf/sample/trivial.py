from tmlf.python import model_builder
from tmlf.python import tmlf_pybind
import numpy as np

tmlf_pybind.feed_tensor("features", tmlf_pybind.Tensor(
    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)))
# workspace.feed_tensor("labels", np.array([0, 1]))

model = model_builder.Model()
model.fc("features", "fc", in_dim=3, out_dim=8)
model.relu("fc", "relu")
# fc2 = model.fc(relu, "fc2", dim_in=8, dim_out=1)
# sigmoid = model.sigmoid(fc2, "sigmoid")
# xent = model.cross_entropy([sigmoid, "labels"], "xent")
# loss = model.averaged_loss(xent, "loss")
# model.add_backward_ops(loss)

# TODO apply optimizers

model.do_init()
model.do_train()
print("bye")
