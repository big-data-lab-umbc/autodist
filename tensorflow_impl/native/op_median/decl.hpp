/**
 * @section DESCRIPTION
 *
 * Median GAR, declarations.
**/

#pragma once

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <common.hpp>

using namespace tensorflow;
using namespace tensorflow::shape_inference;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// -------------------------------------------------------------------------- //
// Op configuration

#define OP_NAME Median
#define OP_TEXT TO_STRING(OP_NAME)

// -------------------------------------------------------------------------- //
// Kernel declaration
namespace OP_NAME {

template<class Device, class T> class Kernel: public Static {
public:
    static void process(OpKernelContext&, size_t const, size_t const, Tensor const&, Tensor&);
};

}
