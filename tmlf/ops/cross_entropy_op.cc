#include "tmlf/core/Operator.h"
#include <glog/logging.h>

using namespace tmlf;

class CrossEntropyOp : public Operator {
 public:
  explicit CrossEntropyOp(const proto::Op& op_proto) : Operator(op_proto) {
  }
  void run() override {
    Tensor pred = ws_->get_tensor(in(0));
    Tensor label = ws_->get_tensor(in(1));

    Tensor ce((label.mat().array() > 0).select(
      (-pred.arr().log()).matrix(),
      (-(1 - pred.arr()).log()).matrix()
    ));
    ws_->add_tensor(out(0), ce);
  }
};

REGISTER_OPERATOR(cross_entropy, CrossEntropyOp);

class CrossEntropyGradOp : public Operator {
 public:
  using Operator::Operator;
};

REGISTER_OPERATOR(cross_entropy_grad, CrossEntropyGradOp);
