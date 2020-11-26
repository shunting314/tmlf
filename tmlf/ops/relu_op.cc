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
