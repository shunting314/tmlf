#include "tmlf/core/Operator.h"
#include <glog/logging.h>

using namespace tmlf;

class SigmoidOp : public Operator {
 public:
  explicit SigmoidOp(const proto::Op& op_proto) : Operator(op_proto) {
  }
  void run() override {
    auto tin = ws_->get_tensor(in(0));
    Tensor tout((1 / ((tin.mat().array() * (-1.0)).exp() + 1)).matrix());
    ws_->add_tensor(out(0), tout);
  }
};

REGISTER_OPERATOR(sigmoid, SigmoidOp);

class SigmoidGradOp : public Operator {
 public:
  using Operator::Operator;
};

REGISTER_OPERATOR(sigmoid_grad, SigmoidGradOp);
