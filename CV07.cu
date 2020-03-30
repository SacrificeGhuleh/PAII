//#include "CV06.cuh"

#include <iostream>

#include <cuda_runtime.h>

#include <cudaDefs.h>
#include <imageManager.h>

#include "imageKernels.cuh"

#include <FreeImage.h>

#include "CudaTimer.cuh"

#define PRINT_MEMORY 0

/**
#if __NVCC__
#else
//defines for visual studio idiotic intellisence
#define __syncthreads() (void)0
#define tex2D(x, y,z) (0)
#endif
*/
namespace nscv07 {
#define BLOCK_DIM 8

  cudaError_t error = cudaSuccess;
  cudaDeviceProp deviceProp = cudaDeviceProp();

  cudaChannelFormatDesc texChannelDesc;

  struct MyImageData {
    unsigned char* dImageData = nullptr;
    unsigned int imageWidth;
    unsigned int imageHeight;
    unsigned int imageBPP; //Bits Per Pixel = 8, 16, 24, or 32 bit
    unsigned int imagePitch;

    size_t texPitch;
    float* dLinearPitchTextureData = nullptr;
    cudaArray* dArrayTextureData = nullptr;

    ~MyImageData() {
      std::cout << "Deleting MyIMageData\n";
      SAFE_DELETE_CUDA(dImageData);
      SAFE_DELETE_CUDA(dLinearPitchTextureData);
      SAFE_DELETE_CUDA(dArrayTextureData);
    }
  };

  MyImageData referenceImageData;
  MyImageData queryImageData;

  texture<float, 2, cudaReadModeElementType> referenceImageTexRef; // declared texture reference must be at file-scope !!!
  texture<float, 2, cudaReadModeElementType> queryImageTexRef; // declared texture reference must be at file-scope !!!

  KernelSetting ks;

  float* dRefOutputData = nullptr;
  float* dQueryOutputData = nullptr;

  void loadSourceImage(const char* imageFileName, MyImageData* imgData) {
    FreeImage_Initialise();
    FIBITMAP* tmp = ImageManager::GenericLoader(imageFileName, 0);

    imgData->imageWidth = FreeImage_GetWidth(tmp);
    imgData->imageHeight = FreeImage_GetHeight(tmp);
    imgData->imageBPP = FreeImage_GetBPP(tmp);
    imgData->imagePitch = FreeImage_GetPitch(tmp); // FREEIMAGE align row data ... You have to use pitch instead of width

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&imgData->dImageData), imgData->imagePitch * imgData->imageHeight * imgData->imageBPP / 8));
    checkCudaErrors(
      cudaMemcpy(imgData->dImageData, FreeImage_GetBits(tmp), imgData->imagePitch * imgData->imageHeight * imgData->imageBPP / 8, cudaMemcpyHostToDevice));

#if PRINT_MEMORY
    checkHostMatrix<unsigned char>(FreeImage_GetBits(tmp), imgData->imagePitch, imgData->imageHeight, imgData->imageWidth, "%6u ",
                                   "Result of Linear Pitch Text");
    checkDeviceMatrix<unsigned char>(imgData->dImageData, imgData->imagePitch, imgData->imageHeight, imgData->imageWidth, "%6u ",
                                     "Result of Linear Pitch Text");
#endif
    FreeImage_Unload(tmp);
    FreeImage_DeInitialise();
  }

  void createTextureFromLinearPitchMemory(MyImageData* imgData, struct textureReference* texRef) {
    // Allocate dLinearPitchTextureData variable memory
    checkCudaErrors(cudaMallocPitch(&imgData->dLinearPitchTextureData, &imgData->texPitch, imgData->imageWidth * sizeof(float), imgData->imageHeight));

    dim3 blockDim(8, 8);
    dim3 gridDim(getNumberOfParts(imgData->imageWidth, blockDim.x), getNumberOfParts(imgData->imageHeight, blockDim.y));

    switch (imgData->imageBPP) {
      // Here call your kernel to convert image into linearPitch memory
      // converts image data into floats.
      // Call the colorToFloat() from  imageKernel.cuh and beware of data alignment, image size and bits per pixel. 
    case 8:
      colorToFloat<8><<<ks.dimGrid, ks.dimBlock>>>(imgData->dImageData, imgData->imageWidth, imgData->imageHeight, imgData->imagePitch,
                                                   imgData->texPitch / sizeof(float), imgData->dLinearPitchTextureData);
      break;
    case 16:
      colorToFloat<16><<<ks.dimGrid, ks.dimBlock>>>(imgData->dImageData, imgData->imageWidth, imgData->imageHeight, imgData->imagePitch,
                                                    imgData->texPitch / sizeof(float), imgData->dLinearPitchTextureData);
      break;
    case 24:
      colorToFloat<24><<<ks.dimGrid, ks.dimBlock>>>(imgData->dImageData, imgData->imageWidth, imgData->imageHeight, imgData->imagePitch,
                                                    imgData->texPitch / sizeof(float), imgData->dLinearPitchTextureData);
      break;
    case 32:
      colorToFloat<32><<<ks.dimGrid, ks.dimBlock>>>(imgData->dImageData, imgData->imageWidth, imgData->imageHeight, imgData->imagePitch,
                                                    imgData->texPitch / sizeof(float), imgData->dLinearPitchTextureData);
      break;
    default:
      throw std::runtime_error("Invalid pits per pixel value");
    }
#if PRINT_MEMORY
    checkDeviceMatrix<float>(imgData->dLinearPitchTextureData, imgData->texPitch, imgData->imageHeight, imgData->imageWidth, "%6.1f ",
                             "Result of Linear Pitch Text");
#endif
    //Define texture (texRef) parameters
    texRef->normalized = false;
    texRef->filterMode = cudaFilterModePoint;
    texRef->addressMode[0] = cudaAddressModeClamp;
    texRef->addressMode[1] = cudaAddressModeClamp;

    //TODO: Define texture channel descriptor (texChannelDesc)
    //We want a floating-point texture.
    //That is why the storage should be 32-bits in a RED channel, 0-bits in green,
    //0-bits in blue and 0-bits in alpha channel, plus the type of the channel format must support floats.
    texChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    //Bind texture
    checkCudaErrors(
      cudaBindTexture2D(nullptr, texRef, imgData->dLinearPitchTextureData, &texChannelDesc, imgData->imageWidth, imgData->imageHeight, imgData->
        texPitch));
  }

  void createTextureFrom2DArray(MyImageData* imgData, struct textureReference* texRef) {
    //TODO: Define texture (texRef) parameters
    texRef->normalized = false;
    texRef->filterMode = cudaFilterModePoint;
    texRef->addressMode[0] = cudaAddressModeClamp;
    texRef->addressMode[1] = cudaAddressModeClamp;

    // Define texture channel descriptor (texChannelDesc)
    //We want a floating-point texture.
    //That is why the storage should be 32-bits in a RED channel, 0-bits in green,
    //0-bits in blue and 0-bits in alpha channel, plus the type of the channel format must support floats.
    texChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    //Converts custom image data to float and stores result in the float_linear_data
    float* dLinearTextureData = nullptr;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dLinearTextureData), imgData->imageWidth * imgData->imageHeight * sizeof(float)));
    switch (imgData->imageBPP) {
      // Here call your kernel to convert image into linear memory (no pitch!!!)
      // converts image data into floats.
      // Call the colorToFloat() from  imageKernel.cuh and beware of data alignment, image size and bits per pixel. 
    case 8:
      colorToFloat<8> <<<ks.dimGrid, ks.dimBlock>>>(imgData->dImageData, imgData->imageWidth, imgData->imageHeight, imgData->imagePitch, imgData->imageWidth,
                                                    dLinearTextureData);
      break;
    case 16:
      colorToFloat<16> <<<ks.dimGrid, ks.dimBlock>>>(imgData->dImageData, imgData->imageWidth, imgData->imageHeight, imgData->imagePitch, imgData->imageWidth,
                                                     dLinearTextureData);
      break;
    case 24:
      colorToFloat<24> <<<ks.dimGrid, ks.dimBlock>>>(imgData->dImageData, imgData->imageWidth, imgData->imageHeight, imgData->imagePitch, imgData->imageWidth,
                                                     dLinearTextureData);
      break;
    case 32:
      colorToFloat<32> <<<ks.dimGrid, ks.dimBlock>>>(imgData->dImageData, imgData->imageWidth, imgData->imageHeight, imgData->imagePitch, imgData->imageWidth,
                                                     dLinearTextureData);
      break;
    default:
      throw std::runtime_error("Invalid pits per pixel value");
    }
    checkCudaErrors(cudaMallocArray(&imgData->dArrayTextureData, &texChannelDesc, imgData->imageWidth, imgData->imageHeight));

    // copy data into cuda array (dArrayTextureData)
    checkCudaErrors(
      cudaMemcpyToArray(imgData->dArrayTextureData, 0, 0, dLinearTextureData, imgData->imageWidth * imgData->imageHeight * sizeof(float),
        cudaMemcpyDeviceToDevice));

    // Bind texture
    checkCudaErrors(cudaBindTextureToArray(texRef, imgData->dArrayTextureData, &texChannelDesc));

    cudaFree(dLinearTextureData);
  }

  void releaseMemory() {
    cudaUnbindTexture(queryImageTexRef);
    cudaUnbindTexture(referenceImageTexRef);

    SAFE_DELETE_CUDA(dRefOutputData);
    SAFE_DELETE_CUDA(dQueryOutputData);
  }

  __global__ void querytexKernel(const unsigned int texWidth, const unsigned int texHeight, float* dst) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //TODO some kernel
    //https://forums.developer.nvidia.com/t/2d-texture-access-how-can-i-access-pixels-from-2d-texture/16721/12
    if (col < texWidth && row < texHeight) {
      dst[col * texWidth + row] = tex2D(queryImageTexRef, col, row);
    }
  }

  __global__ void referencetexKernel(const unsigned int texWidth, const unsigned int texHeight, float* dst) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //TODO some kernel
    //https://forums.developer.nvidia.com/t/2d-texture-access-how-can-i-access-pixels-from-2d-texture/16721/12
    if (col < texWidth && row < texHeight) {
      dst[col * texWidth + row] = tex2D(referenceImageTexRef, col, row);
    }
  }

  __global__ void matchKernel(const unsigned int queryTexWidth,
                              const unsigned int queryTexHeight,
                              const unsigned int refTexWidth,
                              const unsigned int refTexHeight,
                              bool* found,
                              float* queryData,
                              float* refData,
                              int* foundAtRowOffset,
                              int* foundAtColOffset) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //__shared__ int sharedRowOffset = 0;
    //__shared__ int sharedColOffset = 0;
    __shared__ bool sharedKernelFound;

    if (row == 0 && col == 0) {
      sharedKernelFound = true; //set to true, if not found, and with false
    }
    const int maxRowOffset = refTexHeight - queryTexHeight;
    const int maxColOffset = refTexWidth - queryTexWidth;

    for (int rowOffset = 0; rowOffset < maxRowOffset; rowOffset++) {
      for (int colOffset = 0; colOffset < maxColOffset; colOffset++) {
        __syncthreads();
        if ((col < queryTexWidth && row < queryTexHeight) &&
          ((col + colOffset) < refTexWidth && (row + rowOffset) < refTexHeight)) {

          float queryPix = queryData[col * queryTexWidth + row];
          float referencePix = refData[(col + colOffset) * refTexWidth + (row + rowOffset)];
          sharedKernelFound &= (queryPix - referencePix) == 0.f;

          __syncthreads();

          if (sharedKernelFound) {
            if (row == 0 && col == 0) {
              *found = true;
              *foundAtColOffset = colOffset;
              *foundAtRowOffset = rowOffset;
            }
            return;
          }
          if (row == 0 && col == 0) {
            sharedKernelFound = true;
          }

        }

      }
    }
    if (row == 0 && col == 0) {
      *found = false;
      *foundAtColOffset = -1;
      *foundAtRowOffset = -1;
    }
  }

  int cv07(int argc, const char** argv) {
    initializeCUDA(deviceProp);

    {
      //loadSourceImage("query.tif", &queryImageData);
      loadSourceImage("testQuery.png", &queryImageData);

      cudaMalloc(reinterpret_cast<void**>(&dQueryOutputData), queryImageData.imageWidth * queryImageData.imageHeight * sizeof(float));

      ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
      ks.blockSize = BLOCK_DIM * BLOCK_DIM;
      ks.dimGrid = dim3((queryImageData.imageWidth + BLOCK_DIM - 1) / BLOCK_DIM, (queryImageData.imageHeight + BLOCK_DIM - 1) / BLOCK_DIM, 1);

      //Test 2 - texture stored in 2D array
      createTextureFrom2DArray(&queryImageData, &queryImageTexRef);
      querytexKernel<<<ks.dimGrid, ks.dimBlock>>>(queryImageData.imageWidth, queryImageData.imageHeight, dQueryOutputData);
#if PRINT_MEMORY
      checkDeviceMatrix<float>(dQueryOutputData, queryImageData.imageWidth * sizeof(float), queryImageData.imageHeight, queryImageData.imageWidth, "%6.1f ",
                               "dQueryOutputData");
#endif
    }

    {
      //loadSourceImage("reference.tif", &referenceImageData);
      loadSourceImage("testRef.png", &referenceImageData);
      //loadSourceImage("testRefNoMatch.png", &referenceImageData);

      cudaMalloc(reinterpret_cast<void**>(&dRefOutputData), referenceImageData.imageWidth * referenceImageData.imageHeight * sizeof(float));

      //Test 2 - texture stored in 2D array
      createTextureFrom2DArray(&referenceImageData, &referenceImageTexRef);
      referencetexKernel<<<ks.dimGrid, ks.dimBlock>>>(referenceImageData.imageWidth, referenceImageData.imageHeight, dRefOutputData);
#if PRINT_MEMORY
      checkDeviceMatrix<float>(dRefOutputData, referenceImageData.imageWidth * sizeof(float), referenceImageData.imageHeight, referenceImageData.imageWidth,
                               "%6.1f ",
                               "dRefOutputData");
#endif
    }

    bool* deviceFoundMatch;
    bool localFoundMatch;
    cudaMalloc(reinterpret_cast<void**>(&deviceFoundMatch), sizeof(bool));

    int* deviceFoundAtRowOffset;
    int localFoundAtRowOffset;
    cudaMalloc(reinterpret_cast<void**>(&deviceFoundAtRowOffset), sizeof(int));

    int* deviceFoundAtColOffset;
    int localFoundAtColOffset;
    cudaMalloc(reinterpret_cast<void**>(&deviceFoundAtColOffset), sizeof(int));

    nscommon::CudaTimer timer;
    timer.start();
    matchKernel<<<ks.dimGrid, ks.dimBlock>>>(queryImageData.imageWidth,
                                             queryImageData.imageHeight,
                                             referenceImageData.imageWidth,
                                             referenceImageData.imageHeight,
                                             deviceFoundMatch,
                                             dQueryOutputData,
                                             dRefOutputData,
                                             deviceFoundAtRowOffset,
                                             deviceFoundAtColOffset);

    printf("Elapsed: %f\n", timer.stop());
    cudaMemcpy(&localFoundMatch, deviceFoundMatch, sizeof(bool), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(&localFoundAtRowOffset, deviceFoundAtRowOffset, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(&localFoundAtColOffset, deviceFoundAtColOffset, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    printf("Found: %s \n", localFoundMatch ? "true" : "false");
    if (localFoundMatch) {
      printf("  row: %d\n  col: %d\n", localFoundAtRowOffset, localFoundAtColOffset);
    }

    SAFE_DELETE_CUDA(deviceFoundMatch);

    releaseMemory();
    return 0;
  }
}

int main() {
  return nscv07::cv07(0, nullptr);
}
