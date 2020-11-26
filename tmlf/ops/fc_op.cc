#include "tmlf/core/Operator.h"
#include <glog/logging.h>

using namespace tmlf;

class FCOp : public Operator {
 public:
  explicit FCOp(const proto::Op& op_proto) : Operator(op_proto) {
  }
  void run() override {
    Tensor feat = ws_->get_tensor(in(0));
    Tensor w = ws_->get_tensor(in(1));
    Tensor b = ws_->get_tensor(in(2));

    // auto prod = feat.mat() * w.mat(); // auto does not work here. Seems because prod will be a rvalue but assign below needs lval
    Tensor::MatType prod = feat.mat() * w.mat();
    prod.rowwise() += b.vec().transpose();
    ws_->add_tensor(out(0), Tensor(prod));
  }
};

REGISTER_OPERATOR(fc, FCOp);

class FCGradOp : public Operator {
 public:
  using Operator::Operator;
  void run() override {
    auto feat = ws_->get_tensor(in(0)).mat();
    auto w = ws_->get_tensor(in(1)).mat();
    auto gfc = ws_->get_tensor(in(2)).mat();

    auto gfeat_mat = gfc * w.transpose();
    auto gw = feat.transpose() * gfc;
    auto gb = gfc.transpose().rowwise().sum();

    ws_->add_tensor(out(0), Tensor(gfeat_mat));
    ws_->add_tensor(out(1), Tensor(gw));
    ws_->add_tensor(out(2), Tensor(gb));
  }
};

REGISTER_OPERATOR(fc_grad, FCGradOp);
