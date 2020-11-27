#include "tmlf/core/Operator.h"
#include <glog/logging.h>
#include <math.h>

using namespace tmlf;

class LabelCrossEntropyOp : public Operator {
 public:
  using Operator::Operator;
  void run() override {
    auto pred = ws_->get_tensor(in(0)).mat();
    auto label = ws_->get_tensor(in(1)).mat();

    auto xent = Tensor(pred.rows(), 1);
    for (int i = 0; i < pred.rows(); ++i) {
      int loc = (int) label(i, 0);
      float p = pred(i, loc);
      xent.mat()(i, 0) = -log(p);
    }
    ws_->add_tensor(out(0), xent);
  }
};

REGISTER_OPERATOR(label_cross_entropy, LabelCrossEntropyOp);

class LabelCrossEntropyGradOp : public Operator {
 public:
  using Operator::Operator;
  void run() override {
    auto pred = ws_->get_tensor(in(0)).mat();
    auto label = ws_->get_tensor(in(1)).mat();
    auto gxent = ws_->get_tensor(in(2)).mat();

    Tensor::MatType gpred_mat = Tensor::MatType::Constant(
        pred.rows(),
        pred.cols(),
        0.0);
    for (int i = 0; i < pred.rows(); ++i) {
      int loc = (int) label(i, 0);
      gpred_mat(i, loc) = -gxent(i, 0) / pred(i, loc);
    }
    ws_->add_tensor(out(0), Tensor(gpred_mat));
  }
};

REGISTER_OPERATOR(label_cross_entropy_grad, LabelCrossEntropyGradOp);
