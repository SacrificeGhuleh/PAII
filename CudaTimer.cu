#include "CudaTimer.cuh"
#include <cudaDefs.h>

namespace nscommon {
  CudaTimer::CudaTimer() : elapsed_{0}, running_{false} {
    cudaEventCreate(&startEvent_);
    cudaEventCreate(&stopEvent_);
  }

  CudaTimer::~CudaTimer() {
    cudaEventDestroy(startEvent_);
    cudaEventDestroy(stopEvent_);
  }

  void CudaTimer::start(cudaStream_t stream) {
    if (running_) {
      throw std::runtime_error("Timer is already running");
    }
    running_ = true;
    cudaEventRecord(startEvent_, stream);
  }

  float CudaTimer::stop(cudaStream_t stream) {
    if (!running_) {
      throw std::runtime_error("Timer is not running");
    }
    running_ = false;
    cudaEventRecord(stopEvent_, stream);
    cudaEventSynchronize(stopEvent_);
    cudaEventElapsedTime(&elapsed_, startEvent_, stopEvent_);
    return elapsed_;
  }
}
