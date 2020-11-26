#include "tmlf/core/Workspace.h"
#include <glog/logging.h>

namespace tmlf {
Workspace* Workspace::get_ptr() {
  static Workspace ws;
  return &ws;
}

void Workspace::add_tensor(const std::string& name, Tensor tensor) {
  auto ret = str2tensor_.emplace(name, tensor);
  // already exist
  if (!ret.second) {
    ret.first->second = tensor;
  }
}

Tensor Workspace::get_tensor(const std::string& name) {
  auto itr = str2tensor_.find(name);
  if (itr == str2tensor_.end()) {
    LOG(FATAL) << "Tensor not found in workspace: " << name;
  }
  return itr->second;
}

}
