#include "tmlf/core/Operator.h"
#include <glog/logging.h>
#include <algorithm>

using namespace tmlf;

class CircularBatchOp : public Operator {
 public:
  using Operator::Operator;
  void run() override {
    auto X = ws_->get_tensor(in(0)).mat();
    auto y = ws_->get_tensor(in(1)).mat();
    auto cursor_mat = ws_->get_tensor(in(2)).mat();
    int cursor = int(cursor_mat(0, 0) + 0.1);
    size_t batch_size = strtoll(getarg("batch_size").c_str(), nullptr, 10);

    assert(X.rows() == y.rows());
    assert(X.rows() > 0);
    assert(y.cols() == 1);
    assert(cursor_mat.size() == 1);
    assert(cursor >= 0 && cursor < X.rows());
    assert(X.rows() >= batch_size);

    Tensor::MatType X_bat(batch_size, X.cols());
    Tensor::MatType y_bat(batch_size, 1);

    size_t len1 = std::min((size_t) batch_size, (size_t)(X.rows() - cursor));
    X_bat.block(0, 0, len1, X.cols())
        = X.block(cursor, 0, len1, X.cols());
    y_bat.block(0, 0, len1, y.cols())
        = y.block(cursor, 0, len1, y.cols());

    int new_cursor = cursor + len1;
  
    if (len1 < batch_size) {
      X_bat.block(len1, 0, batch_size - len1, X.cols())
          = X.block(0, 0, batch_size - len1, X.cols());
      y_bat.block(len1, 0, batch_size - len1, y.cols())
          = y.block(0, 0, batch_size - len1, y.cols());
      new_cursor = batch_size - len1;
    }
    if (new_cursor == X.rows()) {
      new_cursor = 0;
    }

    auto new_cursor_mat = Tensor::MatType::Constant(1, 1, new_cursor);
    ws_->add_tensor(out(0), Tensor(X_bat));
    ws_->add_tensor(out(1), Tensor(y_bat));
    ws_->add_tensor(out(2), Tensor(new_cursor_mat));
  }
};

REGISTER_OPERATOR(circular_batch, CircularBatchOp);
