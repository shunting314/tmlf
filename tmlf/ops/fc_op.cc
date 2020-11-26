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
