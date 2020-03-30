#include "CV02.cuh"
#include "CudaTimer.cuh"

#include <cudaDefs.h>
#include <ctime>
#include <cmath>
#include <functional>
#include <random>

namespace nscv02 {

  cudaError_t error = cudaSuccess;
  cudaDeviceProp deviceProp = cudaDeviceProp();

  __global__ void fillData(const unsigned int pitch,
                           const unsigned int rows,
                           const unsigned int cols,
                           float* __restrict__ data) {

    const uint16_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const uint16_t y = blockDim.y * blockIdx.y + threadIdx.y;

    const uint16_t pitchInElements = pitch / sizeof(float);
    const uint16_t index = y * pitchInElements + x;

    if ((x < rows) && (x < cols)) {
      data[index] = x * rows + y;
    }
  }

  int cv02(int argc, char* argv[]) {
    initializeCUDA(deviceProp);

    uint16_t rows = 5;
    uint16_t cols = 5;
    float* devPtr;

    //Allocate on gpu and check if error occured
    size_t pitch;
    checkCudaErrors(cudaMallocPitch(&devPtr, &pitch, rows * sizeof(float), cols));

    printf("pitch: %llu\n", pitch);

    const int threadsPerBlock = 8;

    dim3 myGrid(((rows + threadsPerBlock - 1) / threadsPerBlock), (cols + threadsPerBlock - 1) / threadsPerBlock, 1);
    dim3 myBlock(threadsPerBlock, threadsPerBlock, 1);

    nscommon::CudaTimer timer;
    timer.start(0);

    fillData<<<myGrid, myBlock>>>(pitch, rows, cols, devPtr);

    float elapsedTime = timer.stop(0);

    float* hostPtr = new float[pitch * cols];

    for (int i = 0; i < pitch * cols; i++) {
      hostPtr[i] = 0;
    }

    cudaMemcpy2D(hostPtr, rows * sizeof(float), devPtr, pitch, rows * sizeof(float), rows,
                 cudaMemcpyKind::cudaMemcpyDeviceToHost);

    //for (int i = 0; i < pitch * cols; i++) {
    //}

    checkHostMatrix(hostPtr, rows * sizeof(float), cols, rows, "\t%0.1f", "Host data");

    //for (int x = 0; x < rows; x++) {
    //  for (int y = 0; y < cols; y++) {
    //    const uint16_t index = x * rows + y;
    //    printf("\t%0.1f ", hostPtr[index]);
    //  }
    //  printf("\n");
    //}

    printf("\n");

    printf("Elapsed time: %f ms\n", elapsedTime);
    // Free memory

    SAFE_DELETE_CUDA(devPtr);

    return 0;
  }

}
