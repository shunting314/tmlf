from tmlf.python import model_builder
from tmlf.python import tmlf_pybind, workspace, optimizer
import numpy as np

workspace.feed_tensor("features", np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
workspace.feed_tensor("labels", np.array([0, 1], dtype=np.float32))

model = model_builder.Model()
model.fc("features", "fc", in_dim=3, out_dim=8)
model.relu("fc", "relu")
model.fc("relu", "fc2", in_dim=8, out_dim=1)
model.sigmoid("fc2", "sigmoid")
model.cross_entropy(["sigmoid", "labels"], "xent")
model.averaged_loss("xent", "loss")
model.add_backward_ops("loss")

print(f"Param to grad map: {model.get_param_to_grad_map()}")
optimizer.build_sgd(model)

print(model.net.get_proto())

model.do_init()
model.do_train()
print("bye")
