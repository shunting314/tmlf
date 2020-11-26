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
  void run() override {
    auto out_ar = ws_->get_tensor(in(0)).arr();
    auto gout_ar = ws_->get_tensor(in(1)).arr();
    auto gin_ar = out_ar * (1 - out_ar) * gout_ar;
    ws_->add_tensor(out(0), Tensor(gin_ar.matrix()));
  }
};

REGISTER_OPERATOR(sigmoid_grad, SigmoidGradOp);
