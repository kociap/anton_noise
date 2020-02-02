# anton_noise

Noise library aimed at ultra fast generation of 2D and 3D noise. Requires AVX.

Available noise functions:
 - Perlin Noise 2D

## Performance
Compiled with Clang 9 x64 and benchmarked on AMD Ryzen 9 3900X on Arch Linux (A in the table) and Intel i7-8700k on Windows 10 64 bit (I):
### 4k texture 8 octaves (average of 100 runs)
| Function   | AVX  |
|------------|------|
| Perlin (A) | 45ms |
| Perlin (I) | 84ms |

### 16k texture 8 octaves (average of 100 runs)
| Function   | AVX    |
|------------|--------|
| Perlin (A) | 653ms  |
| Perlin (I) | 1267ms |

