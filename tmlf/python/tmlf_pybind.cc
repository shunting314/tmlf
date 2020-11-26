#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <assert.h>
#include <glog/logging.h>
#include "tmlf/proto/tmlf.pb.h"
#include "tmlf/core/Net.h"
#include "tmlf/core/Workspace.h"

namespace py = pybind11;

std::string ping(void) {
  return "Hello, this is tmlf_pybind";
}

void run_net(const std::string& net_proto_ser) {
  tmlf::proto::Net net_proto;
  net_proto.ParseFromString(net_proto_ser);

  tmlf::Net net(net_proto);
  net.run(); 
}

/*
 * refer to https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#f2
 * for how to make tensor works with numpy
 */
tmlf::Tensor fetch_tensor(const std::string& name) {
  auto wsptr = tmlf::Workspace::get_ptr();
  auto tensor = wsptr->get_tensor(name);
  return tensor;
}

PYBIND11_MODULE(tmlf_pybind, m) {
  google::InitGoogleLogging("tmlf_pybind");
  m.def("ping", &ping);
  m.def("run_net", &run_net);
  m.def("fetch_tensor", &fetch_tensor);

  // define Tensor class in python
  py::class_<tmlf::Tensor>(m, "Tensor", py::buffer_protocol())
    .def_buffer([](tmlf::Tensor& tensor) -> py::buffer_info {
      return py::buffer_info(
        tensor.mat().data(),
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        { tensor.mat().rows(), tensor.mat().cols() },
        { tensor.mat().cols() * sizeof(float), sizeof(float) } // strides in bytes
      );
    });
}
