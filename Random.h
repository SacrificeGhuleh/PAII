#pragma once
#include <cstdint>

class Random {
public:
  Random();
  explicit Random(uint32_t seed);
  virtual uint32_t nextUi32(uint32_t min = 0x0, uint32_t max = 0xffffffff) = 0;
  //virtual int32_t nextSi32(int32_t min = 0x80000000, int32_t max = 0x7fffffff) = 0;
  virtual double nextDouble(double min = 0.0, double max = 1.1) = 0;

protected:
  const uint32_t seed_;
};


/**
 * @brief Lehmer random number generator
 * @note https://en.wikipedia.org/wiki/Lehmer_random_number_generator
 */
class LehmerRandom : Random {

  LehmerRandom();
  LehmerRandom(uint32_t seed);

  virtual uint32_t nextUi32(uint32_t min = 0x0, uint32_t max = 0xffffffff) override;
  //virtual int32_t nextSi32(int32_t min = 0x80000000, int32_t max = 0x7fffffff) override;
  virtual double nextDouble(double min = 0.0, double max = 1.1) override;

  void reset();
protected:
  virtual uint32_t rnd();
  uint32_t lehmer_;

};
