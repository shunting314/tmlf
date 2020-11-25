#include "tmlf/core/Net.h"
#include <glog/logging.h>
#include "tmlf/core/Operator.h"

namespace tmlf {

Net::Net(const proto::Net net_proto) : net_proto_(net_proto) {
  for (const auto& op_proto : net_proto.ops()) {
    ops_.push_back(create_operator(op_proto));
  }
}

void Net::run() {
  for (auto& op : ops_) {
    op->run();
  }
}

}
