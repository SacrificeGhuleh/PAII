#include <iostream>
#include <cudaDefs.h>
#include <imageManager.h>
#include <FreeImage.h>
#include <cstdint>
#include <climits>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "arrayUtils.cuh"
#include "CudaTimer.cuh"
#include "imageKernels.cuh"

namespace nscv08 {

#define PRINT_MEMORY 0

#if __NVCC__
#else
  //defines for visual studio idiotic intellisense
#define __syncthreads() (void)0
#define tex2D(x, y,z) (0)
#endif

#define BLOCK_DIM 8
#define TPB_1D 16                   // ThreadsPerBlock in one dimension
#define TPB_2D (TPB_1D * TPB_1D)    // ThreadsPerBlock = 16*16 (2D block)

  cudaError_t error = cudaSuccess;
  cudaDeviceProp deviceProp = cudaDeviceProp();

  texture<float, cudaTextureType2D, cudaReadModeElementType> texRef; // declared texture reference must be at file-scope !!!

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
    /*
    checkHostMatrix<unsigned char>(FreeImage_GetBits(tmp), imagePitch, imageHeight, imageWidth, "%6u ",
                                   "Result of Linear Pitch Text");
    checkDeviceMatrix<unsigned char>(dImageData, imagePitch, imageHeight, imageWidth, "%6u ",
                                     "Result of Linear Pitch Text");
    */
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

    /*checkDeviceMatrix<float>(dLinearPitchTextureData, texPitch, imageHeight, imageWidth, "%6.1f ",
                             "Result of Linear Pitch Text");
*/
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

  void releaseMemory() {
    cudaUnbindTexture(texRef);

    SAFE_DELETE_CUDA(dImageData);
    SAFE_DELETE_CUDA(dLinearPitchTextureData);
    SAFE_DELETE_CUDA(dArrayTextureData);
    SAFE_DELETE_CUDA(dOutputData);
  }

  __global__ void texKernel(const unsigned int texWidth, const unsigned int texHeight, float* dst, float3* dst3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //TODO some kernel
    //https://forums.developer.nvidia.com/t/2d-texture-access-how-can-i-access-pixels-from-2d-texture/16721/12
    if (col < texWidth && row < texHeight) {

      float dx, dy;

      dx =
        (tex2D(texRef, (col - 1), (row - 1))) +
        (2 * tex2D(texRef, (col - 1), (row))) +
        (tex2D(texRef, (col - 1), (row + 1))) +
        (-1 * tex2D(texRef, (col + 1), (row - 1))) +
        (-2 * tex2D(texRef, (col + 1), (row))) +
        (-1 * tex2D(texRef, (col + 1), (row + 1)));

      dy =
        (tex2D(texRef, (col - 1), (row - 1))) +
        (2 * tex2D(texRef, (col), (row - 1))) +
        (tex2D(texRef, (col + 1), (row - 1))) +
        (-1 * tex2D(texRef, (col - 1), (row + 1))) +
        (-2 * tex2D(texRef, (col), (row + 1))) +
        (-1 * tex2D(texRef, (col + 1), (row + 1)));

      float3 result = float3({dx, dy, 1.f / 2.f});

      float lengthOfVector = sqrt((result.x * result.x) + (result.y * result.y) + (result.z * result.z));
      result.x = (result.x / lengthOfVector) * 255;
      result.y = (result.y / lengthOfVector) * 255;
      result.z = (result.z / lengthOfVector) * 255;

      dst[col * texWidth + row] = dx; //sqrt((dx * dx) + (dy * dy)); //tex2D(texRef, col, row);
      //dst3[col * texWidth + row] = {dx, dy, 1.f / 2.f}; //sqrt((dx * dx) + (dy * dy)); //tex2D(texRef, col, row);
      dst3[col * texWidth + row] = result;
    }

  }

  template <class T, FREE_IMAGE_TYPE F>
  void save(const std::string& file_name, T* inData, const uint32_t width, const uint32_t height) {
    FIBITMAP* bitmap = FreeImage_AllocateT(F, width, height, sizeof(T) * 8); // FIT_BITMAP, FIT_BITMAP, FIT_RGBF, FIT_RGBAF
    BYTE* data = (BYTE*)(FreeImage_GetBits(bitmap));
    const int scan_width = FreeImage_GetPitch(bitmap);
    memcpy(data, inData, scan_width * height);
    FreeImage_FlipVertical(bitmap);
    FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(file_name.c_str());
    if (FreeImage_Save(fif, bitmap, file_name.c_str())) {
      printf("Texture has been saved successfully in '%s'.\n", file_name.c_str());
    }
    else {
      printf("Texture failed to save in '%s'.\n", file_name.c_str());
    }
    FreeImage_Unload(bitmap);
    bitmap = nullptr;
    data = nullptr;
  }

  int cv08(int argc, const char** argv) {
    initializeCUDA(deviceProp);

    //loadSourceImage("./Data/terrain10x10.tif");
    loadSourceImage("./Data/terrain3Kx3K.tif");

    cudaMalloc(reinterpret_cast<void**>(&dOutputData), imageWidth * imageHeight * sizeof(float));
    //How many block of the size of [16x16] will process the reference image? 
    //Too much to manage. That's we use a 1D grid of [16x16] blocks that will move down the image.
    //This we need (((ref.width - query.width + 1) + 16 - 1)/16) blocks!!!
    uint32_t noBlocksX = ((imageWidth + 1) + TPB_1D - 1) / TPB_1D;
    uint32_t noBlocksY = ((imageHeight + 1) + TPB_1D - 1) / TPB_1D;
    dim3 block{TPB_1D, TPB_1D, 1};
    dim3 grid{noBlocksX, noBlocksY, 1};

    createTextureFromLinearPitchMemory();
    float3* dOutputData3 = nullptr;
    cudaMalloc(&dOutputData3, imageWidth * imageHeight * sizeof(float3));

    texKernel<<<grid, block>>>(imageWidth, imageHeight, dOutputData, dOutputData3);

    //checkDeviceMatrix<float>(dOutputData, imageWidth * sizeof(float), imageHeight, imageWidth, "%6.1f ", "dOutputData");

    float* hOutputData = nullptr;
    float3* hOutputData3 = nullptr;

    hOutputData = new float[imageWidth * imageHeight];
    hOutputData3 = new float3[imageWidth * imageHeight];

    uchar3* rgbdataSobel = nullptr;
    uchar3* rgbdataNormal = nullptr;
    rgbdataSobel = new uchar3[imageWidth * imageHeight];
    rgbdataNormal = new uchar3[imageWidth * imageHeight];

    cudaMemcpy(hOutputData, dOutputData, sizeof(float) * imageWidth * imageHeight, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(hOutputData3, dOutputData3, sizeof(float3) * imageWidth * imageHeight, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    for (int y = 0; y < imageHeight; y++) {
      for (int x = 0; x < imageWidth; x++) {
        float pixSobel = hOutputData[y * imageWidth + x];
        float3 pixNormal = hOutputData3[y * imageWidth + x];
        // rgbdataSobel[y * imageWidth + x].x = pixSobel /* 255*/;
        // rgbdataSobel[y * imageWidth + x].y = pixSobel /* 255*/;
        // rgbdataSobel[y * imageWidth + x].z = pixSobel /* 255*/;

        rgbdataNormal[y * imageWidth + x].x = pixNormal.x /* 255*/;
        rgbdataNormal[y * imageWidth + x].y = pixNormal.y /* 255*/;
        rgbdataNormal[y * imageWidth + x].z = pixNormal.z /* 255*/;
      }
      //printf("\n");
    }

    // save<uchar3, FIT_BITMAP>("./Data/outSobel.png", rgbdataSobel, imageWidth, imageHeight);
    save<uchar3, FIT_BITMAP>("./Data/outNormal.png", rgbdataNormal, imageWidth, imageHeight);

    SAFE_DELETE_ARRAY(hOutputData);
    SAFE_DELETE_ARRAY(hOutputData3);
    SAFE_DELETE_ARRAY(rgbdataSobel);
    SAFE_DELETE_ARRAY(rgbdataNormal);
    SAFE_DELETE_CUDA(dOutputData3);

    releaseMemory();
    return 0;
  }
}
