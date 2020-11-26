from tmlf.proto import tmlf_pb2
from tmlf.python import gradopmaker
from tmlf.python.proto_utils import to_op_proto

class Net:
    def __init__(self):
        self.seen_names = set()
        self.op_list = []
        self.tensor_to_grad_map = None

    def next_name(self, name):
        orig_name = name
        seq = 0
        while name in self.seen_names:
            name = f"{orig_name}_{seq}"
            seq += 1
        self.seen_names.add(name)
        return name

    def __getattr__(self, name):
        def add_op(in_tensors, out_tensors, **kwargs):
            self.op_list.append(to_op_proto(name, in_tensors, out_tensors, **kwargs))
        return add_op

    def get_proto(self):
        net = tmlf_pb2.Net()
        net.ops.extend(self.op_list)
        return net

    def add_backward_ops(self, loss_tkey):
        ghelper = gradopmaker.GradOpMaker()

        grad_ops = []
        tensor_to_grad_map = {}
        grad_ops.append(ghelper.make_grad_op_for_loss(loss_tkey))
        for op in self.op_list[::-1]:
            grad_op, extra_map = ghelper.make_grad_op(op)
            grad_ops.append(grad_op)
            tensor_to_grad_map.update(extra_map)

        self.op_list.extend(grad_ops)
        self.tensor_to_grad_map = tensor_to_grad_map

def run_net(net):
    from tmlf.python import tmlf_pybind
    tmlf_pybind.run_net(net.get_proto().SerializeToString())

class Model:
    def __init__(self):
        self.init_net = Net()
        self.net = Net()

    def fc(self, in_tensors, out_tensors, in_dim, out_dim):
        w = self.init_net.next_name("w")
        b = self.init_net.next_name("b")
        self.init_net.xavier_fill([], w, shape=[in_dim, out_dim])
        self.init_net.constant_fill([], b, shape=[out_dim])
        self.net.fc([in_tensors, w, b], out_tensors)

    def __getattr__(self, name):
        return getattr(self.net, name)

    def add_backward_ops(self, loss_tkey):
        self.net.add_backward_ops(loss_tkey)

    def do_init(self):
        run_net(self.init_net)

    def do_train(self):
        run_net(self.net)
