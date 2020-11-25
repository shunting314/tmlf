#include "tmlf/core/Operator.h"
#include <glog/logging.h>

using namespace tmlf;

class XavierFillOp : public Operator {
 public:
  explicit XavierFillOp(const proto::Op& op_proto) : Operator(op_proto) {
  }
  void run() override {
    LOG(FATAL) << "ni";
  }
};

REGISTER_OPERATOR(xavier_fill, XavierFillOp);
