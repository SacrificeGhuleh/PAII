#include <cudaDefs.h>
#include <random>
#include <functional>

#include "../CudaTimer.cuh"
#include "../CV02.cuh"
#include "../CV03.cuh"
#include "../CV04.cuh"
#include "../CV06.cuh"
#include "../CV07.cuh"
#include "../CV08.cuh"

int main(int argc, const char** argv) {
  //nscv2::cv02(argc, argv);
  //nscv03::cv03(argc, argv);
  //nscv04::cv04(argc, argv);
  //nscv06::cv06(argc, argv);
  //nscv07::cv07(argc, argv);
  nscv08::cv08(argc, argv);
}
