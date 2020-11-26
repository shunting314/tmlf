#include "tmlf/core/Operator.h"
#include <glog/logging.h>
#include <stdlib.h>

namespace tmlf {

std::unique_ptr<Operator> create_operator(const proto::Op& op_proto) {
  return OperatorRegistry::get().create_operator(op_proto);
}

std::unique_ptr<Operator> OperatorRegistry::create_operator(const proto::Op& op_proto) {
  auto creator_itr = str_to_creator_.find(op_proto.type());
  if (creator_itr == str_to_creator_.end()) {
    LOG(FATAL) << "Unregistered operator: " << op_proto.DebugString();
  }
  return creator_itr->second(op_proto);
}

std::string Operator::getarg(const std::string& name) {
  for (const auto& arg : op_proto_.args()) {
    if (arg.key() == name) {
      return arg.value();
    }
  }
  LOG(FATAL) << "Argument not found " << name;
  return "";
}

std::string Operator::getarg(const std::string& name, const std::string& def) {
  for (const auto& arg : op_proto_.args()) {
    if (arg.key() == name) {
      return arg.value();
    }
  }
  return def;
}

// TODO only unsigied int so far
std::vector<int64_t> arg_to_ints(const std::string& str) {
  std::vector<int64_t> ret;
  const char* ptr = str.c_str();
  while (*ptr) {
    while (!isdigit(*ptr) && *ptr) {
      ++ptr;
    }
    if (!*ptr) {
      break;
    }
    int64_t v;
    char* endptr;
    v = strtoll(ptr, &endptr, 10);
    ret.push_back(v);
    ptr = endptr;
  }
  return ret;
}

}
