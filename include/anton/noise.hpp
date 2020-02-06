#ifndef ANTON_NOISE_NOISE_HPP
#define ANTON_NOISE_NOISE_HPP

#include <stdint.h>

namespace anton {
    // Generates 2D perlin noise.
    // The output values are normalized to range [0, 1].
    //
    // buffer must be a 32 byte aligned size * size big buffer of floats. All elements must be cleared to 0 before this function is called.
    // size is the size of the texture to be generated. The final texture will be size * size large. Must be a power of 2.
    // start_octave, end_octave define how many layers the final noise will consist of. The function will generate end_octave - start_octave layers.
    //   The maximum number of layers is 15 and maximum value of end_octave is 15.
    // persistence defines the contribution of a layer to the final noise and is computed as pow(persistence, octave - start_octave).
    //
    // Example usage:
    //   // Generate 4k texture with 8 octaves
    //   anton::perlin_2D(aligned_buffer, std::random_device()(), 4096, 0, 8, 0.5f);
    //
    void perlin_2D(float* buffer, uint64_t seed, uint32_t size, uint32_t start_octave, uint32_t end_octave, float persistence);
} // namespace anton

#endif // !ANTON_NOISE_NOISE_HPP
