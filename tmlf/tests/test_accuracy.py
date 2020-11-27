import numpy as np
from tmlf.python import workspace, model_builder

workspace.feed_tensor("pred", np.array(
    [
        [ 0.1, 0.2, 0.7, ],
        [ 0.2, 0.1, 0.7, ],
        [ 0.3, 0.4, 0.3, ],
    ], dtype=np.float32,
))
workspace.feed_tensor("label", np.array(
    [
        2, 0, 1,
    ], dtype=np.float32,
))

net = model_builder.Net()
net.accuracy(['pred', 'label'], 'acc')
model_builder.run_net(net)
acc = workspace.fetch_tensor('acc').reshape(-1)
acc_expected = np.array(
    [ 2.0 / 3.0], dtype=np.float32,
)
np.testing.assert_almost_equal(acc_expected, acc)
