#ifndef ANTON_NOISE_NOISE_HPP
#define ANTON_NOISE_NOISE_HPP

#include <stdint.h>

namespace anton {
    // Generates 2D perlin noise.
    //
    // buffer must be a 32 byte aligned size * size big buffer of floats. All elements must be cleared to 0 before this function is called.
    // size is the size of the texture to be generated. The final texture will be size * size large. Must be a power of 2.
    // octaves must be less than 16.
    //
    void perlin_2D(float* buffer, uint64_t seed, uint32_t size, uint32_t octaves);
} // namespace anton

#endif // !ANTON_NOISE_NOISE_HPP
