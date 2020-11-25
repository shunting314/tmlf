#include "tmlf/core/Operator.h"
#include <glog/logging.h>

using namespace tmlf;

class ConstantFillOp : public Operator {
 public:
  explicit ConstantFillOp(const proto::Op& op_proto) : Operator(op_proto) {
  }
  void run() override {
    LOG(FATAL) << "ni";
  }
};

REGISTER_OPERATOR(constant_fill, ConstantFillOp);
