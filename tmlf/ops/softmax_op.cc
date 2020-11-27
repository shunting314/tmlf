#include "tmlf/core/Operator.h"
#include <glog/logging.h>

using namespace tmlf;

class SoftmaxOp : public Operator {
 public:
  using Operator::Operator;
  void run() override {
    auto sm_in = ws_->get_tensor(in(0)).arr();
    Tensor::ArrType sm_out = sm_in.exp();
    auto rowsum = sm_out.rowwise().sum();
    sm_out.colwise() /= rowsum;
    ws_->add_tensor(out(0), Tensor(sm_out.matrix()));
  }
};

REGISTER_OPERATOR(softmax, SoftmaxOp);

class SoftmaxGradOp : public Operator {
 public:
  using Operator::Operator;
  void run() override {
    auto sm_out = ws_->get_tensor(in(0)).arr();
    auto gsm_out = ws_->get_tensor(in(1)).arr();

    Tensor::ArrType gsm_in = gsm_out;
    auto dotvec = (sm_out * gsm_out).rowwise().sum();
    gsm_in.colwise() -= dotvec;
    gsm_in *= sm_out;

    ws_->add_tensor(out(0), Tensor(gsm_in.matrix()));
  }
};

REGISTER_OPERATOR(softmax_grad, SoftmaxGradOp);
