#include "Random.h"

#include <ctime>

Random::Random() : seed_(time(nullptr)) {}
Random::Random(uint32_t seed) : seed_(seed) {}

LehmerRandom::LehmerRandom() : Random() {
  reset();
}

LehmerRandom::LehmerRandom(uint32_t seed) : Random(seed) {
  reset();
}

uint32_t LehmerRandom::nextUi32(uint32_t min, uint32_t max) {
  return (rnd() % (max - min)) + min;
}

double LehmerRandom::nextDouble(double min, double max) {
  return (static_cast<double>(rnd()) / static_cast<double>(0x7FFFFFFF)) * (max - min) + min;
}

void LehmerRandom::reset() {
  lehmer_ = seed_;
}

uint32_t LehmerRandom::rnd() {
  lehmer_ += 0xe120fc15U;
  uint64_t tmp = static_cast<uint64_t>(seed_) * 0x4a39b70dU;
  const uint32_t m1 = (tmp >> 32) ^ tmp;
  tmp = static_cast<uint64_t>(m1) * 0x12fad5c9U;
  const uint32_t m2 = (tmp >> 32) ^ tmp;
  return m2;
}
