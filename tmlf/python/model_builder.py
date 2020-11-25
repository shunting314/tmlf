from tmlf.proto import tmlf_pb2

def to_proto_args(kwargs):
    kwlist = []
    for k, v in kwargs.items():
        arg = tmlf_pb2.Arg()
        arg.key = k;
        arg.value = str(v)
        kwlist.append(arg)
    return kwlist

class Net:
    def __init__(self):
        self.seen_names = set()
        self.op_list = []

    def next_name(self, name):
        orig_name = name
        seq = 0
        while name in self.seen_names:
            name = f"{orig_name}_seq"
            seq += 1
        self.seen_names.add(name)
        return name

    def __getattr__(self, name):
        def add_op(in_tensors, out_tensors, **kwargs):
            self.op_list.append((name, in_tensors, out_tensors, kwargs))
        return add_op

    def get_proto(self):
        net = tmlf_pb2.Net()
        for op in self.op_list:
            name, in_tensors, out_tensors, kwargs = op
            op_proto = net.ops.add()
            op_proto.type = name
            op_proto.in_tensors.extend(in_tensors)
            op_proto.out_tensors.extend(out_tensors)
            op_proto.args.extend(to_proto_args(kwargs))
        return net

def run_net(net):
    print(net.get_proto())
    assert False

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

    def do_init(self):
        run_net(self.init_net)

    def do_train(self):
        run_net(self.net)
