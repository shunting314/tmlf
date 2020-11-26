#pragma once

#include <memory>
#include "tmlf/proto/tmlf.pb.h"
#include <glog/logging.h>
#include "tmlf/core/Workspace.h"

namespace tmlf {

class Operator;

class OperatorRegistry {
 public:
  typedef std::unique_ptr<Operator> (*OperatorCreator)(const proto::Op& op_proto);
  static OperatorRegistry& get() {
    static OperatorRegistry registry;
    return registry;
  }
  std::unique_ptr<Operator> create_operator(const proto::Op& op_proto);
 private:
  OperatorRegistry() {}
  std::unordered_map<std::string, OperatorCreator> str_to_creator_;
  friend class OperatorRegisterer;
};

class OperatorRegisterer {
 public:
  explicit OperatorRegisterer(const std::string& name, OperatorRegistry::OperatorCreator creator) {
    auto& registry = OperatorRegistry::get();
    if (registry.str_to_creator_.count(name) > 0) {
      LOG(FATAL) << "Operator " << name << " already registered";
    }
    registry.str_to_creator_.emplace(name, creator);
  }
};

class Operator {
 public:
  explicit Operator(const proto::Op& op_proto) : op_proto_(op_proto) {
    // only support single workspace so far
    ws_ = Workspace::get_ptr();
  }
  virtual ~Operator() {}
  virtual void run() = 0;
  std::string getarg(const std::string& name);
  std::string getarg(const std::string& name, const std::string& def);
  const std::string& in(int i) const { return op_proto_.in_tensors()[i]; }
  const std::string& out(int i) const { return op_proto_.out_tensors()[i]; }
 protected:
  proto::Op op_proto_;
  Workspace* ws_;
};

std::unique_ptr<Operator> create_operator(const proto::Op& op_proto);

#define REGISTER_OPERATOR(op_type, op_class) \
  static auto reg ## op_type = OperatorRegisterer(#op_type, [](const proto::Op& op_proto) -> std::unique_ptr<Operator> { \
    return std::make_unique<op_class>(op_proto); \
  })

// arg utils
std::vector<int64_t> arg_to_ints(const std::string& str);

}
