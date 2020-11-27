#include "tmlf/core/Operator.h"
#include <glog/logging.h>

using namespace tmlf;

class AccuracyOp : public Operator {
 public:
  using Operator::Operator;
  void run() override {
    auto pred = ws_->get_tensor(in(0)).mat();
    auto label = ws_->get_tensor(in(1)).mat();
    int pos = 0;
    auto rowmax = pred.rowwise().maxCoeff();
    for (int i = 0; i < pred.rows(); ++i) {
      pos += (pred(i, (int) label(i, 0)) >= rowmax(i));
    }
    auto acc = Tensor::MatType::Constant(1, 1, (float) pos / pred.rows());
    ws_->add_tensor(out(0), Tensor(acc));
  }
};

REGISTER_OPERATOR(accuracy, AccuracyOp);
