#pragma once

#include <vector>
#include <Eigen/Dense>
#include <iostream>

namespace tmlf {

namespace {
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXf;
// Eigen broadcast op requires a compile time vector
// (not matrix with 1 column at run time)
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VecXf;
}

class Tensor {
 public:
  using MatType = MatXf;
  using VecType = VecXf;
  explicit Tensor(int64_t w, int64_t h);
  explicit Tensor(const MatXf& mat) : mat_(std::make_shared<MatXf>(mat)) { }

  // TODO maybe it's better to not expose MatXf at all
  MatXf& mat() { return *mat_; }

  // TODO support row vector as well
  VecXf vec() { assert(mat_->cols() == 1); return Eigen::Map<VecXf>(mat_->data(), mat_->rows() * mat_->cols()); }
 private:
  std::shared_ptr<MatXf> mat_;
  friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
};


}
