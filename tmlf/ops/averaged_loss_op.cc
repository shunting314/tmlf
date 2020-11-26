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
  void run() override {
    Tensor xent = ws_->get_tensor(in(0));
    Tensor gloss = ws_->get_tensor(in(1));
    assert(gloss.size() == 1);
    float loss_scalar = gloss.mat()(0, 0);
    size_t xent_size = xent.size();
    auto gxent_mat = Tensor::MatType::Constant(
        xent.rows(),
        xent.cols(),
        loss_scalar / (float) xent_size);
    ws_->add_tensor(out(0), Tensor(gxent_mat));
  }
};

REGISTER_OPERATOR(averaged_loss_grad, AveragedLossGradOp);
