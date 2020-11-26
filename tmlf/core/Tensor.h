#pragma once

#include <vector>
#include <Eigen/Dense>
#include <iostream>

namespace tmlf {

namespace {
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXf;
}

class Tensor {
 public:
  explicit Tensor(int64_t w, int64_t h);

  // TODO maybe it's better to not expose MatXf at all
  MatXf& mat() { return *mat_; }
 private:
  std::shared_ptr<MatXf> mat_;
  friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
};


}
