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

tmlf::Tensor fetch_tensor(const std::string& name) {
  auto wsptr = tmlf::Workspace::get_ptr();
  auto tensor = wsptr->get_tensor(name);
  return tensor;
}

void feed_tensor(const std::string& name, tmlf::Tensor tensor) {
  auto wsptr = tmlf::Workspace::get_ptr();
  wsptr->add_tensor(name, tensor);
}

PYBIND11_MODULE(tmlf_pybind, m) {
  google::InitGoogleLogging("tmlf_pybind");
  m.def("ping", &ping);
  m.def("run_net", &run_net);
  m.def("fetch_tensor", &fetch_tensor);
  m.def("feed_tensor", &feed_tensor);

  /*
   * refer to https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#f2
   * for how to make tensor works with numpy
   */
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
    }).def(py::init([](py::buffer b) {
      typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;
      py::buffer_info info = b.request();
      if (info.format != py::format_descriptor<float>::format()) {
        LOG(FATAL) << "Incompatible format";
      }
      if (info.ndim != 2) {
        LOG(FATAL) << "Incompatible buffer dimension! Got " << info.ndim;
      }
      auto strides = Strides(
        info.strides[0] / sizeof(float),
        info.strides[1] / sizeof(float)
      );
      auto map = Eigen::Map<tmlf::Tensor::MatType, 0, Strides>(
          static_cast<float *>(info.ptr), info.shape[0], info.shape[1], strides
      );
      return tmlf::Tensor(map);
    }));
}
