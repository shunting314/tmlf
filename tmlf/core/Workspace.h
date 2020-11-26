#pragma once
#include "tmlf/core/Tensor.h"
#include <unordered_map>

namespace tmlf {

class Workspace {
 public:
  static Workspace* get_ptr();
  Tensor get_tensor(const std::string& name);
  void add_tensor(const std::string& name, Tensor tensor);
 private:
  Workspace() {}
  std::unordered_map<std::string, Tensor> str2tensor_;
};

}
