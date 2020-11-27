from tmlf.python.proto_utils import to_op_proto

class Rule:
    def __init__(self, op_type, in_sig_list=[], out_sig_list=[]):
        self.op_type = op_type
        self.in_sig_list = in_sig_list
        self.out_sig_list = out_sig_list

RULE_LIST = {
    "fc": Rule(
        "fc_grad",
        ["i0", "i1", "go0",], # feat, w, b
        ["gi0", "gi1", "gi2"], # fc
    ),
    "relu": Rule(
        "relu_grad",
        ["i0", "go0"],
        ["gi0"],
    ),
    "sigmoid": Rule(
        "sigmoid_grad",
        ["o0", "go0"],
        ["gi0"],
    ),
    "cross_entropy": Rule(
        "cross_entropy_grad",
        ["i0", "i1", "go0"],
        ["gi0"],
    ),
    "averaged_loss": Rule(
        "averaged_loss_grad",
        ["i0", "go0"],
        ["gi0"],
    ),
    "label_cross_entropy": Rule(
        "label_cross_entropy_grad",
    ),
    "softmax": Rule(
        "softmax_grad",
    ),

    # ops to skip gradient
    "circular_batch": None,
    "accuracy": None,
}

def sig_to_tkey(sig, orig_op, tensor_to_grad_map):
    is_grad = False
    if sig[0] == 'g':
        is_grad = True
        sig = sig[1:]
    assert len(sig) == 2 # don't expect more than 10 input/output
    assert sig[0] in 'io'
    is_input = (sig[0] == 'i')
    index = int(sig[1:])
    
    tkey = None
    if is_grad:
        if is_input:
            tkey = f"{orig_op.in_tensors[index]}_grad"
            tensor_to_grad_map[orig_op.in_tensors[index]] = tkey
        else:
            tkey = f"{orig_op.out_tensors[index]}_grad"
    else:
        if is_input:
            tkey = orig_op.in_tensors[index]
        else:
            tkey = orig_op.out_tensors[index]

    return tkey

class GradOpMaker:
    def __init__(self):
        pass

    def make_grad_op(self, orig_op):
        if orig_op.type not in RULE_LIST:
            raise RuntimeError(f"No gradient rule defined for {orig_op.type}")
        rule = RULE_LIST[orig_op.type]
        if rule is None: # skip gradient
            return None, {}
        tensor_to_grad_map = {}
        in_tensors = []
        out_tensors = []

        for sig in rule.in_sig_list:
            in_tensors.append(sig_to_tkey(sig, orig_op, tensor_to_grad_map))

        for sig in rule.out_sig_list:
            out_tensors.append(sig_to_tkey(sig, orig_op, tensor_to_grad_map))

        return to_op_proto(
            rule.op_type,
            in_tensors,
            out_tensors,
        ), tensor_to_grad_map

    def make_grad_op_for_loss(self, loss_tkey):
        return to_op_proto(
            "constant_fill",
            loss_tkey,
            f"{loss_tkey}_grad",
            value=1.0,
        )
