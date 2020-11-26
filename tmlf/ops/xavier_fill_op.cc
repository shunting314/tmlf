#include "tmlf/core/Operator.h"
#include <glog/logging.h>
#include "tmlf/core/Tensor.h"

using namespace tmlf;

class XavierFillOp : public Operator {
 public:
  explicit XavierFillOp(const proto::Op& op_proto) : Operator(op_proto) {
  }
  void run() override {
    auto shape = arg_to_ints(getarg("shape"));
    assert(shape.size() == 2);
    Tensor out(shape[0], shape[1]);
    ws_->add_tensor(op_proto_.out_tensors()[0], out);
    LOG(FATAL) << "\n" << ws_->get_tensor(op_proto_.out_tensors()[0]);
  }
};

REGISTER_OPERATOR(xavier_fill, XavierFillOp);
