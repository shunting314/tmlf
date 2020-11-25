#pragma once

#include "tmlf/proto/tmlf.pb.h"
#include "tmlf/core/Operator.h"

namespace tmlf {

class Net {
 public:
  explicit Net(const proto::Net net_proto);
  void run();
 private:
  proto::Net net_proto_;
  std::vector<std::unique_ptr<Operator>> ops_;
};

}
