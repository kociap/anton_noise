#ifndef ANTON_NOISE_NOISE_HPP
#define ANTON_NOISE_NOISE_HPP

#include <stdint.h>

namespace anton {
    // buffer must a 32 byte aligned size*size big buffer of floats.
    // size must be a power of 2.
    // octaves must be less than 16.
    void generate_perlin_noise_texture(float* buffer, uint64_t seed, uint32_t size, uint32_t octaves);
} // namespace anton

#endif // !ANTON_NOISE_NOISE_HPP
