#include "tmlf/core/Tensor.h"
#include <glog/logging.h>

namespace tmlf {

Tensor::Tensor(int64_t w, int64_t h) {
  mat_ = std::make_shared<MatXf>(w, h);
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  os << *(tensor.mat_);
  return os;
}

}
