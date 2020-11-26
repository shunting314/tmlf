#include "tmlf/core/Operator.h"
#include <glog/logging.h>

using namespace tmlf;

class AveragedLossOp : public Operator {
 public:
  explicit AveragedLossOp(const proto::Op& op_proto) : Operator(op_proto) {
  }
  void run() override {
    Tensor xent = ws_->get_tensor(in(0));
    float avg_loss = xent.arr().mean();
    Tensor loss(1, 1);
    loss.mat()(0, 0) = avg_loss;
    ws_->add_tensor(out(0), loss);
  }
};

REGISTER_OPERATOR(averaged_loss, AveragedLossOp);

class AveragedLossGradOp : public Operator {
 public:
  using Operator::Operator;
};

REGISTER_OPERATOR(averaged_loss_grad, AveragedLossGradOp);
