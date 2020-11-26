from tmlf.proto import tmlf_pb2
import six

def to_proto_args(kwargs):
    kwlist = []
    for k, v in kwargs.items():
        arg = tmlf_pb2.Arg()
        arg.key = k;
        arg.value = str(v)
        kwlist.append(arg)
    return kwlist

def to_op_proto(name, in_tensors, out_tensors, **kwargs):
    op_proto = tmlf_pb2.Op()
    op_proto.type = name
    if isinstance(in_tensors, six.string_types):
        in_tensors = [in_tensors]
    op_proto.in_tensors.extend(in_tensors)
    if isinstance(out_tensors, six.string_types):
        out_tensors = [out_tensors]
    op_proto.out_tensors.extend(out_tensors)
    op_proto.args.extend(to_proto_args(kwargs))
    return op_proto
