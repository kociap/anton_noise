
#include <anton/noise.hpp>

#include <random>

#include <immintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>

namespace anton {
    using u8 = uint8_t;
    using i32 = int32_t;
    using u32 = uint32_t;
    using i64 = int64_t;
    using u64 = uint64_t;
    using f32 = float;

    struct vec2 {
        f32 x;
        f32 y;
    };

    struct Gradient_Grid {
        vec2 gradients[16];
        u8 perm_table[128];

        vec2 at(u64 const x, u64 const y) const {
            u8 const index = (y % 128 + x) % 128;
            return gradients[perm_table[index] % 16];
        }
    };

    static Gradient_Grid create_gradient_grid(std::mt19937& random_engine) {
        Gradient_Grid grid{{{0.0f, 1.0f}, {0.382683f, 0.923879f}, {0.707107f, 0.707107f}, {0.923879f, 0.382683f}, {1.0f, 0.0f}, {0.923879f, -0.382683f}, {0.707107f, -0.707107f}, {0.382683f, -0.923879f}, {0.0f, -1.0f}, {-0.382683f, -0.923879f}, {-0.707107f, -0.707107f}, {-0.923879f, -0.382683f}, {-1.0f, 0.0f}, {-0.923879f, 0.382683f}, {-0.707107f, 0.707107f}, {-0.382683f, 0.923879f}}, {}};

        std::uniform_int_distribution<u32> d(0, 255);
        for (int i = 0; i < 128; ++i) {
            grid.perm_table[i] = d(random_engine);
        }

        return grid;
    }

    // TODO: case for size < 8

    void generate_perlin_noise_texture(float* const buffer, u64 const seed, u32 const size, u32 const octaves) {
        std::mt19937 random_engine(seed);
        Gradient_Grid const grid = create_gradient_grid(random_engine);

        f32 amplitude = 1.0f;
        f32 const persistence = 0.5f;
        f32 const size_f32 = size;
        for (u32 octave = 0; octave < octaves; ++octave) {
            amplitude *= persistence;
            u64 const noise_scale = 1 << octave;
            u64 const resample_period = size / noise_scale;
            if (resample_period >= 8) {
                f32 const scale_factor_f32 = (f32)noise_scale / size_f32;
                __m256 const scale_factor = _mm256_set1_ps(scale_factor_f32);
                __m256 const remap_factor = _mm256_set1_ps(amplitude * 0.5f * 1.4142135f);
                for (u64 y = 0; y < size; ++y) {
                    f32 const y_coord = (f32)y * scale_factor_f32;
                    u64 const sample_offset_y = y / resample_period;
                    f32 const y_fractional = y_coord - sample_offset_y;
                    f32 const y_lerp_factor = y_fractional * y_fractional * y_fractional * (y_fractional * (y_fractional * 6 - 15) + 10);
                    __m256 x_coord = _mm256_mul_ps(_mm256_set_ps(-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f), scale_factor);
                    __m256 increment = _mm256_mul_ps(_mm256_set1_ps(8.0f), scale_factor);
                    for (u64 x = 0, sample_offset_x = 0; sample_offset_x < noise_scale; ++sample_offset_x) {
                        vec2 const g00 = grid.at(sample_offset_x, sample_offset_y);
                        vec2 const g10 = grid.at(sample_offset_x + 1, sample_offset_y);
                        vec2 const g01 = grid.at(sample_offset_x, sample_offset_y + 1);
                        vec2 const g11 = grid.at(sample_offset_x + 1, sample_offset_y + 1);
                        __m256 x_floor = _mm256_set1_ps(sample_offset_x);
                        for (u64 i = 0; i < resample_period; i += 8, x += 8) {
                            x_coord = _mm256_add_ps(x_coord, increment);
                            __m256 x_fractional = _mm256_sub_ps(x_coord, x_floor);
                            __m256 x_fractional_less_one = _mm256_sub_ps(x_fractional, _mm256_set1_ps(1.0f));

                            __m256 fac00x = _mm256_mul_ps(x_fractional, _mm256_set1_ps(g00.x));
                            __m256 fac00y = _mm256_set1_ps(g00.y * y_fractional);
                            __m256 fac00 = _mm256_add_ps(fac00x, fac00y);

                            __m256 fac10x = _mm256_mul_ps(x_fractional_less_one, _mm256_set1_ps(g10.x));
                            __m256 fac10y = _mm256_set1_ps(g10.y * y_fractional);
                            __m256 fac10 = _mm256_add_ps(fac10x, fac10y);

                            __m256 fac01x = _mm256_mul_ps(x_fractional, _mm256_set1_ps(g01.x));
                            __m256 fac01y = _mm256_set1_ps(g01.y * (y_fractional - 1.0f));
                            __m256 fac01 = _mm256_add_ps(fac01x, fac01y);

                            __m256 fac11x = _mm256_mul_ps(x_fractional_less_one, _mm256_set1_ps(g11.x));
                            __m256 fac11y = _mm256_set1_ps(g11.y * (y_fractional - 1.0f));
                            __m256 fac11 = _mm256_add_ps(fac11x, fac11y);

                            __m256 x_fractional_cube = _mm256_mul_ps(x_fractional, _mm256_mul_ps(x_fractional, x_fractional));
                            __m256 lerp_factor = _mm256_mul_ps(x_fractional_cube, _mm256_add_ps(_mm256_mul_ps(x_fractional, _mm256_sub_ps(_mm256_mul_ps(x_fractional, _mm256_set1_ps(6)), _mm256_set1_ps(15))), _mm256_set1_ps(10)));

                            __m256 lerp_factor_compl_xmm = _mm256_sub_ps(_mm256_set1_ps(1.0f), lerp_factor);
                            __m256 lerped_x0 = _mm256_add_ps(_mm256_mul_ps(lerp_factor, fac10), _mm256_mul_ps(lerp_factor_compl_xmm, fac00));
                            __m256 lerped_x1 = _mm256_add_ps(_mm256_mul_ps(lerp_factor, fac11), _mm256_mul_ps(lerp_factor_compl_xmm, fac01));

                            __m256 noise_r0 = _mm256_add_ps(_mm256_mul_ps(lerped_x1, _mm256_set1_ps(y_lerp_factor)), _mm256_mul_ps(lerped_x0, _mm256_set1_ps(1.0f - y_lerp_factor)));
                            __m256 noise_r1 = _mm256_add_ps(noise_r0, _mm256_set1_ps(0.7071067f));
                            __m256 noise_r2 = _mm256_mul_ps(noise_r1, remap_factor);
                            __m256 current = _mm256_load_ps(buffer + y * size + x);
                            __m256 noise = _mm256_add_ps(noise_r2, current);
                            _mm256_store_ps(buffer + y * size + x, noise);
                        }
                    }
                }
            } else if (resample_period >= 4) {
                f32 const scale_factor_f32 = (f32)noise_scale / size_f32;
                __m128 const scale_factor = _mm_set1_ps(scale_factor_f32);
                for (u64 y = 0; y < size; ++y) {
                    f32 const y_coord = (f32)y * scale_factor_f32;
                    u64 const sample_offset_y = y / resample_period;
                    f32 const y_fractional = y_coord - sample_offset_y;
                    f32 const y_lerp_factor = y_fractional * y_fractional * y_fractional * (y_fractional * (y_fractional * 6 - 15) + 10);
                    __m128 x_coord = _mm_mul_ps(_mm_set_ps(-1.0f, -2.0f, -3.0f, -4.0f), scale_factor);
                    __m128 increment = _mm_mul_ps(_mm_set1_ps(4.0f), scale_factor);
                    for (u64 x = 0, sample_offset_x = 0; sample_offset_x < noise_scale; ++sample_offset_x) {
                        vec2 const g00 = grid.at(sample_offset_x, sample_offset_y);
                        vec2 const g10 = grid.at(sample_offset_x + 1, sample_offset_y);
                        vec2 const g01 = grid.at(sample_offset_x, sample_offset_y + 1);
                        vec2 const g11 = grid.at(sample_offset_x + 1, sample_offset_y + 1);
                        __m128 x_floor = _mm_set1_ps(sample_offset_x);
                        for (u64 i = 0; i < resample_period; i += 4, x += 4) {
                            x_coord = _mm_add_ps(x_coord, increment);
                            __m128 x_fractional = _mm_sub_ps(x_coord, x_floor);
                            __m128 x_fractional_less_one = _mm_sub_ps(x_fractional, _mm_set1_ps(1.0f));

                            __m128 fac00x = _mm_mul_ps(x_fractional, _mm_set1_ps(g00.x));
                            __m128 fac00y = _mm_set1_ps(g00.y * y_fractional);
                            __m128 fac00 = _mm_add_ps(fac00x, fac00y);

                            __m128 fac10x = _mm_mul_ps(x_fractional_less_one, _mm_set1_ps(g10.x));
                            __m128 fac10y = _mm_set1_ps(g10.y * y_fractional);
                            __m128 fac10 = _mm_add_ps(fac10x, fac10y);

                            __m128 fac01x = _mm_mul_ps(x_fractional, _mm_set1_ps(g01.x));
                            __m128 fac01y = _mm_set1_ps(g01.y * (y_fractional - 1.0f));
                            __m128 fac01 = _mm_add_ps(fac01x, fac01y);

                            __m128 fac11x = _mm_mul_ps(x_fractional_less_one, _mm_set1_ps(g11.x));
                            __m128 fac11y = _mm_set1_ps(g11.y * (y_fractional - 1.0f));
                            __m128 fac11 = _mm_add_ps(fac11x, fac11y);

                            __m128 x_fractional_cube = _mm_mul_ps(x_fractional, _mm_mul_ps(x_fractional, x_fractional));
                            __m128 lerp_factor = _mm_mul_ps(x_fractional_cube, _mm_add_ps(_mm_mul_ps(x_fractional, _mm_sub_ps(_mm_mul_ps(x_fractional, _mm_set1_ps(6)), _mm_set1_ps(15))), _mm_set1_ps(10)));

                            __m128 lerp_factor_compl_xmm = _mm_sub_ps(_mm_set1_ps(1.0f), lerp_factor);
                            __m128 lerped_x0 = _mm_add_ps(_mm_mul_ps(lerp_factor, fac10), _mm_mul_ps(lerp_factor_compl_xmm, fac00));
                            __m128 lerped_x1 = _mm_add_ps(_mm_mul_ps(lerp_factor, fac11), _mm_mul_ps(lerp_factor_compl_xmm, fac01));

                            __m128 noise_r0 = _mm_add_ps(_mm_mul_ps(lerped_x1, _mm_set1_ps(y_lerp_factor)), _mm_mul_ps(lerped_x0, _mm_set1_ps(1.0f - y_lerp_factor)));
                            __m128 noise_r1 = _mm_add_ps(noise_r0, _mm_set1_ps(0.7071067f));
                            __m128 noise_r2 = _mm_mul_ps(noise_r1, _mm_set1_ps(amplitude * 0.5f * 1.4142135f));
                            __m128 current = _mm_load_ps(buffer + y * size + x);
                            __m128 noise = _mm_add_ps(noise_r2, current);
                            _mm_store_ps(buffer + y * size + x, noise);
                        }
                    }
                }
            } else {
                f32 const scale_factor_f32 = (f32)noise_scale / size_f32;
                __m256 scale_factor = _mm256_set1_ps(scale_factor_f32);
                for (u64 y = 0; y < size; ++y) {
                    f32 const y_coord = (f32)y * scale_factor_f32;
                    u64 const sample_y = y_coord;
                    f32 const y_fractional = y_coord - sample_y;
                    __m256 const y_lerp_factor = _mm256_set1_ps(y_fractional * y_fractional * y_fractional * (y_fractional * (y_fractional * 6 - 15) + 10));
                    __m256 const y_lerp_factor_compl = _mm256_sub_ps(_mm256_set1_ps(1.0f), y_lerp_factor);
                    __m256 x_coord = _mm256_mul_ps(_mm256_set_ps(-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f), scale_factor);
                    __m256 increment = _mm256_mul_ps(_mm256_set1_ps(8.0f), scale_factor);
                    for (u64 x = 0; x < size; x += 8) {
                        x_coord = _mm256_add_ps(x_coord, increment);
                        __m256 sample_x = _mm256_floor_ps(x_coord);
                        alignas(32) f32 sample_x_store[8];
                        _mm256_store_ps(sample_x_store, sample_x);

                        vec2 const g00[8] = {
                            grid.at(sample_x_store[0], sample_y),
                            grid.at(sample_x_store[1], sample_y),
                            grid.at(sample_x_store[2], sample_y),
                            grid.at(sample_x_store[3], sample_y),
                            grid.at(sample_x_store[4], sample_y),
                            grid.at(sample_x_store[5], sample_y),
                            grid.at(sample_x_store[6], sample_y),
                            grid.at(sample_x_store[7], sample_y),
                        };
                        vec2 const g10[8] = {
                            grid.at(sample_x_store[0] + 1, sample_y),
                            grid.at(sample_x_store[1] + 1, sample_y),
                            grid.at(sample_x_store[2] + 1, sample_y),
                            grid.at(sample_x_store[3] + 1, sample_y),
                            grid.at(sample_x_store[4] + 1, sample_y),
                            grid.at(sample_x_store[5] + 1, sample_y),
                            grid.at(sample_x_store[6] + 1, sample_y),
                            grid.at(sample_x_store[7] + 1, sample_y),
                        };
                        vec2 const g01[8] = {
                            grid.at(sample_x_store[0], sample_y + 1),
                            grid.at(sample_x_store[1], sample_y + 1),
                            grid.at(sample_x_store[2], sample_y + 1),
                            grid.at(sample_x_store[3], sample_y + 1),
                            grid.at(sample_x_store[4], sample_y + 1),
                            grid.at(sample_x_store[5], sample_y + 1),
                            grid.at(sample_x_store[6], sample_y + 1),
                            grid.at(sample_x_store[7], sample_y + 1),
                        };
                        vec2 const g11[8] = {
                            grid.at(sample_x_store[0] + 1, sample_y + 1),
                            grid.at(sample_x_store[1] + 1, sample_y + 1),
                            grid.at(sample_x_store[2] + 1, sample_y + 1),
                            grid.at(sample_x_store[3] + 1, sample_y + 1),
                            grid.at(sample_x_store[4] + 1, sample_y + 1),
                            grid.at(sample_x_store[5] + 1, sample_y + 1),
                            grid.at(sample_x_store[6] + 1, sample_y + 1),
                            grid.at(sample_x_store[7] + 1, sample_y + 1),
                        };

                        __m256 g00x = _mm256_set_ps(g00[7].x, g00[6].x, g00[5].x, g00[4].x, g00[3].x, g00[2].x, g00[1].x, g00[0].x);
                        __m256 g00y = _mm256_set_ps(g00[7].y, g00[6].y, g00[5].y, g00[4].y, g00[3].y, g00[2].y, g00[1].y, g00[0].y);
                        __m256 g10x = _mm256_set_ps(g10[7].x, g10[6].x, g10[5].x, g10[4].x, g10[3].x, g10[2].x, g10[1].x, g10[0].x);
                        __m256 g10y = _mm256_set_ps(g10[7].y, g10[6].y, g10[5].y, g10[4].y, g10[3].y, g10[2].y, g10[1].y, g10[0].y);
                        __m256 g01x = _mm256_set_ps(g01[7].x, g01[6].x, g01[5].x, g01[4].x, g01[3].x, g01[2].x, g01[1].x, g01[0].x);
                        __m256 g01y = _mm256_set_ps(g01[7].y, g01[6].y, g01[5].y, g01[4].y, g01[3].y, g01[2].y, g01[1].y, g01[0].y);
                        __m256 g11x = _mm256_set_ps(g11[7].x, g11[6].x, g11[5].x, g11[4].x, g11[3].x, g11[2].x, g11[1].x, g11[0].x);
                        __m256 g11y = _mm256_set_ps(g11[7].y, g11[6].y, g11[5].y, g11[4].y, g11[3].y, g11[2].y, g11[1].y, g11[0].y);

                        __m256 x_fractional = _mm256_sub_ps(x_coord, sample_x);
                        __m256 x_fractional_s1 = _mm256_sub_ps(x_fractional, _mm256_set1_ps(1.0f));
                        __m256 fac00 = _mm256_add_ps(_mm256_mul_ps(g00x, x_fractional), _mm256_mul_ps(g00y, _mm256_set1_ps(y_fractional)));
                        __m256 fac10 = _mm256_add_ps(_mm256_mul_ps(g10x, x_fractional_s1), _mm256_mul_ps(g10y, _mm256_set1_ps(y_fractional)));
                        __m256 fac01 = _mm256_add_ps(_mm256_mul_ps(g01x, x_fractional), _mm256_mul_ps(g01y, _mm256_set1_ps(y_fractional - 1.0f)));
                        __m256 fac11 = _mm256_add_ps(_mm256_mul_ps(g11x, x_fractional_s1), _mm256_mul_ps(g11y, _mm256_set1_ps(y_fractional - 1.0f)));

                        __m256 x_fractional_cube = _mm256_mul_ps(x_fractional, _mm256_mul_ps(x_fractional, x_fractional));
                        __m256 lerp_factor = _mm256_mul_ps(x_fractional_cube, _mm256_add_ps(_mm256_mul_ps(x_fractional, _mm256_sub_ps(_mm256_mul_ps(x_fractional, _mm256_set1_ps(6)), _mm256_set1_ps(15))), _mm256_set1_ps(10)));
                        __m256 lerped_x0 = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), lerp_factor), fac00), _mm256_mul_ps(lerp_factor, fac10));
                        __m256 lerped_x1 = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), lerp_factor), fac01), _mm256_mul_ps(lerp_factor, fac11));
                        __m256 noise = _mm256_add_ps(_mm256_mul_ps(y_lerp_factor_compl, lerped_x0), _mm256_mul_ps(y_lerp_factor, lerped_x1));
                        __m256 noise_remapped = _mm256_mul_ps(_mm256_set1_ps(amplitude * 0.5f * 1.4142135f), _mm256_add_ps(_mm256_set1_ps(0.7071067f), noise));
                        __m256 current = _mm256_load_ps(buffer + y * size + x);
                        __m256 value = _mm256_add_ps(noise_remapped, current);
                        _mm256_store_ps(buffer + y * size + x, value);
                    }
                }
            }
        }
    }
} // namespace anton
