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
