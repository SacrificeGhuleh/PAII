#include <cudaDefs.h>
#include <random>
#include <functional>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

constexpr uint32_t arrLength = 1000;

class CudaTimer {
public:
  CudaTimer() : elapsed_{0}, running_{false} {
    cudaEventCreate(&startEvent_);
    cudaEventCreate(&stopEvent_);
  }

  ~CudaTimer() {
    cudaEventDestroy(startEvent_);
    cudaEventDestroy(stopEvent_);
  }

  void start(cudaStream_t stream = 0) {
    if (running_) {
      throw std::runtime_error("Timer is already running");
    }
    running_ = true;
    cudaEventRecord(startEvent_, stream);
  }

  float stop(cudaStream_t stream = 0) {
    if (!running_) {
      throw std::runtime_error("Timer is not running");
    }
    running_ = false;
    cudaEventRecord(stopEvent_, stream);
    cudaEventSynchronize(stopEvent_);
    cudaEventElapsedTime(&elapsed_, startEvent_, stopEvent_);
    return elapsed_;
  }

private:

  cudaEvent_t startEvent_;
  cudaEvent_t stopEvent_;
  bool running_;
  float elapsed_;
};

/**
 * @brief Kernel to add two vectors
 * @param a             First value
 * @param b             Second value
 * @param c             Output value
 * @param length        Lenth of vectors
 * @note  __restrict__  Avoids pointers aliasing
 */
__global__ void add(
  const uint32_t* __restrict__ a,
  const uint32_t* __restrict__ b,
  uint32_t* __restrict__ c,
  const uint32_t length) {
  /** Thread index */
  uint16_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint16_t skip = blockDim.x * gridDim.x;
  while (tid < length) {
    c[tid] = a[tid] + b[tid];
    tid += skip;
  }
}

typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<float> Distribution;

auto uniform_generator = std::bind(Distribution(0.0f, 1.0f), Engine(1));

float random(const float range_min, const float range_max) {
  return static_cast<float>(uniform_generator()) * (range_max - range_min) + range_min;
}

void cv01() {
  initializeCUDA(deviceProp);

  uint32_t* arrA = new uint32_t[arrLength];
  uint32_t* arrB = new uint32_t[arrLength];
  uint32_t* arrC = new uint32_t[arrLength];

  for (int i = 0; i < arrLength; i++) {
    arrA[i] = i; //random(0, 1000);
    arrB[i] = i; //random(0, 1000);

    arrC[i] = 0;
  }

  uint32_t* da; //device ptr a
  uint32_t* db; //device ptr b
  uint32_t* dc; //device ptr c

  //Allocate on gpu and check if error occured

  checkCudaErrors(cudaMalloc(&da, arrLength * sizeof(uint32_t)));
  checkCudaErrors(cudaMalloc(&db, arrLength * sizeof(uint32_t)));
  checkCudaErrors(cudaMalloc(&dc, arrLength * sizeof(uint32_t)));

  //Copy data to GPU to CPU
  cudaMemcpy(da, arrA, arrLength * sizeof(uint32_t), cudaMemcpyKind::cudaMemcpyHostToDevice);
  cudaMemcpy(db, arrB, arrLength * sizeof(uint32_t), cudaMemcpyKind::cudaMemcpyHostToDevice);

  dim3 myBlock(512, 1, 1);
  dim3 myGrid(MINIMUM(getNumberOfParts(arrLength, 512), 64), 1, 1);

  cudaEvent_t startEvent, stopEvent;

  //cudaEventCreate(&startEvent);
  //cudaEventCreate(&stopEvent);

  //cudaEventRecord(startEvent, 0);

  CudaTimer timer;
  timer.start(0);

  add<<<myGrid, myBlock>>>(da, db, dc, arrLength);

  float elapsedTime = timer.stop(0);
  //cudaEventRecord(stopEvent, 0);
  //cudaEventSynchronize(stopEvent);
  //cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

  //Copy data from GPU to CPU
  cudaMemcpy(arrC, dc, arrLength * sizeof(uint32_t), cudaMemcpyKind::cudaMemcpyDeviceToHost);

  checkDeviceMatrix<uint32_t>(dc, arrLength * sizeof(uint32_t), 1, arrLength, "%d ", "Device C");
  /*
  for(int i = 0; i < arrLength; i++)
  {
    printf("%d + %d = %d \n", arrA[i], arrB[i], arrC[i]);
  }*/

  printf("Elapsed time: %f ms\n", elapsedTime);

  // cudaEventDestroy(startEvent);
  // cudaEventDestroy(stopEvent);

  // Free memory
  SAFE_DELETE_ARRAY(arrA);
  SAFE_DELETE_ARRAY(arrB);
  SAFE_DELETE_ARRAY(arrC);

  SAFE_DELETE_CUDA(da);
  SAFE_DELETE_CUDA(db);
  SAFE_DELETE_CUDA(dc);
}

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

void cv02() {
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

  CudaTimer timer;
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
}

int main(int argc, char* argv[]) {
  //cv01();
  cv02();
}
