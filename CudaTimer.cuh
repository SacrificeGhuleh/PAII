#include <cuda_runtime.h>

namespace nscommon {
  class CudaTimer {
  public:
    CudaTimer();
    ~CudaTimer();
    void start(cudaStream_t stream = 0);
    float stop(cudaStream_t stream = 0);
  private:
    cudaEvent_t startEvent_;
    cudaEvent_t stopEvent_;
    bool running_;
    float elapsed_;
  };
}
