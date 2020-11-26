#include "tmlf/core/Operator.h"
#include <glog/logging.h>

using namespace tmlf;

class WeightedSumOp : public Operator {
 public:
  using Operator::Operator;
  void run() override {
    assert(num_in() == 4); // only support 2 pairs so far
    assert(num_out() == 1);
    auto v0 = ws_->get_tensor(in(0)).arr();
    auto w0_ar = ws_->get_tensor(in(1)).arr();
    auto v1 = ws_->get_tensor(in(2)).arr();
    auto w1_ar = ws_->get_tensor(in(3)).arr();

    assert(w0_ar.size() == 1);
    assert(w1_ar.size() == 1);
    float w0 = w0_ar(0, 0);
    float w1 = w1_ar(0, 0);

    auto out_ar = v0 * w0 + v1 * w1;
    ws_->add_tensor(out(0), Tensor(out_ar.matrix()));
  }
};

REGISTER_OPERATOR(weighted_sum, WeightedSumOp);
