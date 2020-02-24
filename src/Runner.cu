#include <cudaDefs.h>
#include <random>
#include <functional>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

constexpr uint32_t arrLength = 1000;

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

int main(int argc, char* argv[]) {
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

  add<<<myGrid, myBlock>>>(da, db, dc, arrLength);

  //Copy data from GPU to CPU
  cudaMemcpy(arrC, dc, arrLength * sizeof(uint32_t), cudaMemcpyKind::cudaMemcpyDeviceToHost);

  checkDeviceMatrix<uint32_t>(dc, arrLength * sizeof(uint32_t), 1, arrLength, "%d ", "Device C");
  /*
  for(int i = 0; i < arrLength; i++)
  {
    printf("%d + %d = %d \n", arrA[i], arrB[i], arrC[i]);
  }*/

  // Free memory
  SAFE_DELETE_ARRAY(arrA);
  SAFE_DELETE_ARRAY(arrB);
  SAFE_DELETE_ARRAY(arrC);

  SAFE_DELETE_CUDA(da);
  SAFE_DELETE_CUDA(db);
  SAFE_DELETE_CUDA(dc);
}
