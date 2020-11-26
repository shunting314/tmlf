#include "tmlf/core/Operator.h"
#include <glog/logging.h>

using namespace tmlf;

class ReluOp : public Operator {
 public:
  explicit ReluOp(const proto::Op& op_proto) : Operator(op_proto) {
  }
  void run() override {
    auto tin = ws_->get_tensor(in(0));
    ws_->add_tensor(out(0), Tensor(tin.mat().cwiseMax(0)));
  }
};

REGISTER_OPERATOR(relu, ReluOp);

class ReluGradOp : public Operator {
 public:
  using Operator::Operator;
  void run() override {
    auto _in = ws_->get_tensor(in(0)).arr();
    auto _gout = ws_->get_tensor(in(1)).arr();

    auto rows = _in.rows();
    auto cols = _in.cols();
    auto _gin = (_in > 0).select(
        Tensor::ArrType::Constant(rows, cols, 1.0),
        Tensor::ArrType::Constant(rows, cols, 0.0)
    ) * _gout;
    
    ws_->add_tensor(out(0), Tensor(_gin.matrix()));
  }
};

REGISTER_OPERATOR(relu_grad, ReluGradOp);
