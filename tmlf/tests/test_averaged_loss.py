from tmlf.python import workspace, model_builder
import numpy as np

# fwd
workspace.feed_tensor("xent", np.array(
    [ 0.2, 0.3, 0.4], dtype=np.float32,
))
net = model_builder.Net()
net.averaged_loss("xent", "loss")
model_builder.run_net(net)
loss = workspace.fetch_tensor("loss").reshape([-1])
print(loss)
np.testing.assert_almost_equal(loss, np.array([0.3], dtype=np.float32))

# bwd
net.add_backward_ops()
workspace.feed_tensor("loss_grad", np.array(
    [1.0], dtype=np.float32,
))
model_builder.run_net(net)
xent_grad = workspace.fetch_tensor("xent_grad").reshape([-1])
np.testing.assert_almost_equal(xent_grad, np.full(3, 1.0 / 3.0, dtype=np.float32))
