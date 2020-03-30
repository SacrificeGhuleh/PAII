//#include "CV06.cuh"

#include <iostream>

#include <cuda_runtime.h>

#include <cudaDefs.h>
#include <imageManager.h>

#include "imageKernels.cuh"

#include <FreeImage.h>

namespace nscv06 {
#define BLOCK_DIM 8

  cudaError_t error = cudaSuccess;
  cudaDeviceProp deviceProp = cudaDeviceProp();

  texture<float, 2, cudaReadModeElementType> texRef; // declared texture reference must be at file-scope !!!

  cudaChannelFormatDesc texChannelDesc;

  unsigned char* dImageData = nullptr;
  unsigned int imageWidth;
  unsigned int imageHeight;
  unsigned int imageBPP; //Bits Per Pixel = 8, 16, 24, or 32 bit
  unsigned int imagePitch;

  size_t texPitch;
  float* dLinearPitchTextureData = nullptr;
  cudaArray* dArrayTextureData = nullptr;

  KernelSetting ks;

  float* dOutputData = nullptr;

  void loadSourceImage(const char* imageFileName) {
    FreeImage_Initialise();
    FIBITMAP* tmp = ImageManager::GenericLoader(imageFileName, 0);

    imageWidth = FreeImage_GetWidth(tmp);
    imageHeight = FreeImage_GetHeight(tmp);
    imageBPP = FreeImage_GetBPP(tmp);
    imagePitch = FreeImage_GetPitch(tmp); // FREEIMAGE align row data ... You have to use pitch instead of width

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dImageData), imagePitch * imageHeight * imageBPP / 8));
    checkCudaErrors(cudaMemcpy(dImageData, FreeImage_GetBits(tmp), imagePitch * imageHeight * imageBPP / 8, cudaMemcpyHostToDevice));

    checkHostMatrix<unsigned char>(FreeImage_GetBits(tmp), imagePitch, imageHeight, imageWidth, "%6u ",
                                   "Result of Linear Pitch Text");
    checkDeviceMatrix<unsigned char>(dImageData, imagePitch, imageHeight, imageWidth, "%6u ",
                                     "Result of Linear Pitch Text");

    FreeImage_Unload(tmp);
    FreeImage_DeInitialise();
  }

  void createTextureFromLinearPitchMemory() {
    // Allocate dLinearPitchTextureData variable memory
    checkCudaErrors(cudaMallocPitch(&dLinearPitchTextureData, &texPitch, imageWidth * sizeof(float), imageHeight));

    dim3 blockDim(8, 8);
    dim3 gridDim(getNumberOfParts(imageWidth, blockDim.x), getNumberOfParts(imageHeight, blockDim.y));

    switch (imageBPP) {
      // Here call your kernel to convert image into linearPitch memory
      // converts image data into floats.
      // Call the colorToFloat() from  imageKernel.cuh and beware of data alignment, image size and bits per pixel. 
    case 8:
      colorToFloat<8><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch,
                                                   texPitch / sizeof(float), dLinearPitchTextureData);
      break;
    case 16:
      colorToFloat<16><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch,
                                                    texPitch / sizeof(float), dLinearPitchTextureData);
      break;
    case 24:
      colorToFloat<24><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch,
                                                    texPitch / sizeof(float), dLinearPitchTextureData);
      break;
    case 32:
      colorToFloat<32><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch,
                                                    texPitch / sizeof(float), dLinearPitchTextureData);
      break;
    default:
      throw std::runtime_error("Invalid pits per pixel value");
    }

    checkDeviceMatrix<float>(dLinearPitchTextureData, texPitch, imageHeight, imageWidth, "%6.1f ",
                             "Result of Linear Pitch Text");

    //Define texture (texRef) parameters
    texRef.normalized = false;
    texRef.filterMode = cudaFilterModePoint;
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;

    //TODO: Define texture channel descriptor (texChannelDesc)
    //We want a floating-point texture.
    //That is why the storage should be 32-bits in a RED channel, 0-bits in green,
    //0-bits in blue and 0-bits in alpha channel, plus the type of the channel format must support floats.
    texChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    //Bind texture
    checkCudaErrors(cudaBindTexture2D(nullptr, &texRef, dLinearPitchTextureData, &texChannelDesc, imageWidth, imageHeight, texPitch));
  }

  void createTextureFrom2DArray() {
    //TODO: Define texture (texRef) parameters
    texRef.normalized = false;
    texRef.filterMode = cudaFilterModePoint;
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;

    // Define texture channel descriptor (texChannelDesc)
    //We want a floating-point texture.
    //That is why the storage should be 32-bits in a RED channel, 0-bits in green,
    //0-bits in blue and 0-bits in alpha channel, plus the type of the channel format must support floats.
    texChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    //Converts custom image data to float and stores result in the float_linear_data
    float* dLinearTextureData = nullptr;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dLinearTextureData), imageWidth * imageHeight * sizeof(float)));
    switch (imageBPP) {
      // Here call your kernel to convert image into linear memory (no pitch!!!)
      // converts image data into floats.
      // Call the colorToFloat() from  imageKernel.cuh and beware of data alignment, image size and bits per pixel. 
    case 8:
      colorToFloat<8> <<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, imageWidth, dLinearTextureData);
      break;
    case 16:
      colorToFloat<16> <<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, imageWidth, dLinearTextureData);
      break;
    case 24:
      colorToFloat<24> <<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, imageWidth, dLinearTextureData);
      break;
    case 32:
      colorToFloat<32> <<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, imageWidth, dLinearTextureData);
      break;
    default:
      throw std::runtime_error("Invalid pits per pixel value");
    }
    checkCudaErrors(cudaMallocArray(&dArrayTextureData, &texChannelDesc, imageWidth, imageHeight));

    // copy data into cuda array (dArrayTextureData)
    checkCudaErrors(cudaMemcpyToArray(dArrayTextureData, 0, 0, dLinearTextureData, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToDevice));

    // Bind texture
    checkCudaErrors(cudaBindTextureToArray(&texRef, dArrayTextureData, &texChannelDesc));

    cudaFree(dLinearTextureData);
  }

  void releaseMemory() {
    cudaUnbindTexture(texRef);

    SAFE_DELETE_CUDA(dImageData);
    SAFE_DELETE_CUDA(dLinearPitchTextureData);
    SAFE_DELETE_CUDA(dArrayTextureData);
    SAFE_DELETE_CUDA(dOutputData);
  }

  __global__ void texKernel(const unsigned int texWidth, const unsigned int texHeight, float* dst) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //TODO some kernel
    //https://forums.developer.nvidia.com/t/2d-texture-access-how-can-i-access-pixels-from-2d-texture/16721/12
    if (col < texWidth && row < texHeight) {
      dst[col * texWidth + row] = tex2D(texRef, col, row);
    }

  }

  int cv06(int argc, const char** argv) {
    initializeCUDA(deviceProp);

    loadSourceImage("terrain10x10.tif");

    std::cout << "\n\nTexture loaded\n\n\n";

    cudaMalloc(reinterpret_cast<void**>(&dOutputData), imageWidth * imageHeight * sizeof(float));

    ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
    ks.blockSize = BLOCK_DIM * BLOCK_DIM;
    ks.dimGrid = dim3((imageWidth + BLOCK_DIM - 1) / BLOCK_DIM, (imageHeight + BLOCK_DIM - 1) / BLOCK_DIM, 1);

    std::cout << "\n\nLinear pitch texture\n\n\n";
    ////Test 1 - texture stored in linear pitch memory
    createTextureFromLinearPitchMemory();
    texKernel<<<ks.dimGrid, ks.dimBlock>>>(imageWidth, imageHeight, dOutputData);
    checkDeviceMatrix<float>(dOutputData, imageWidth * sizeof(float), imageHeight, imageWidth, "%6.1f ", "dOutputData");

    std::cout << "\n\n2D texture\n\n\n";

    //Test 2 - texture stored in 2D array
    createTextureFrom2DArray();
    texKernel<<<ks.dimGrid, ks.dimBlock>>>(imageWidth, imageHeight, dOutputData);
    checkDeviceMatrix<float>(dOutputData, imageWidth * sizeof(float), imageHeight, imageWidth, "%6.1f ", "dOutputData");

    releaseMemory();
    return 0;
  }
}


