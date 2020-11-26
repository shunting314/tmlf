#include "tmlf/core/Operator.h"
#include <glog/logging.h>

using namespace tmlf;

class ConstantFillOp : public Operator {
 public:
  explicit ConstantFillOp(const proto::Op& op_proto) : Operator(op_proto) {
  }
  void run() override {
    std::vector<int64_t> shape;
    if (op_proto_.in_tensors().size() > 0) {
      Tensor in_tensor = ws_->get_tensor(op_proto_.in_tensors()[0]);
      shape.push_back(in_tensor.mat().rows());
      shape.push_back(in_tensor.mat().cols());
    } else {
      shape = arg_to_ints(getarg("shape"));
    }
    if (shape.size() == 1) {
      shape.push_back(1); // column vector
    }
    Tensor out(shape[0], shape[1]);
    std::string valstr = getarg("value", "0");
    float val = strtof(valstr.c_str(), nullptr);
    out.mat() = Tensor::MatType::Constant(shape[0], shape[1], val);
    ws_->add_tensor(op_proto_.out_tensors()[0], out);
  }
};

REGISTER_OPERATOR(constant_fill, ConstantFillOp);
