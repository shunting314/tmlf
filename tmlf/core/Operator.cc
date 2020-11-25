#include "tmlf/core/Operator.h"
#include <glog/logging.h>

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

}
