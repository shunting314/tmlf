class Net:
    def __init__(self):
        self.seen_names = set()

    def next_name(self, name):
        orig_name = name
        seq = 0
        while name in self.seen_names:
            name = f"{orig_name}_seq"
            seq += 1
        self.seen_names.add(name)
        return name

def run_net(net):
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
