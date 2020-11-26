#include "tmlf/core/Operator.h"
#include <glog/logging.h>
#include "tmlf/core/Tensor.h"
#include <random>

using namespace tmlf;

static float uniform_rand(float minv, float maxv) {
  static std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(minv, maxv);
  return distribution(generator);
}

class XavierFillOp : public Operator {
 public:
  explicit XavierFillOp(const proto::Op& op_proto) : Operator(op_proto) {
  }
  void run() override {
    auto shape = arg_to_ints(getarg("shape"));
    assert(shape.size() == 2);
    Tensor out(shape[0], shape[1]);
    auto dim_in = shape[0];
    auto scale = sqrt(3 / dim_in);
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        out.mat()(i, j) = uniform_rand(-scale, scale);
      }
    }
    ws_->add_tensor(op_proto_.out_tensors()[0], out);
  }
};

REGISTER_OPERATOR(xavier_fill, XavierFillOp);
