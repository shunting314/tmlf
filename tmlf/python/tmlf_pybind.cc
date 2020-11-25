#include <pybind11/pybind11.h>
#include <string>
#include <assert.h>

namespace py = pybind11;

std::string ping(void) {
  return "Hello, this is tmlf_pybind";
}

void run_net(const std::string& net_proto_ser) {
  // LOG(FATAL) << "ni"; // TODO
  assert(false);
}

PYBIND11_MODULE(tmlf_pybind, m) {
  m.def("ping", &ping);
  m.def("run_net", &run_net);
}
