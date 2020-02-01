# anton_noise

Noise library aimed at ultra fast generation of 2D and 3D noise. Requires AVX.

Available noise functions:
 - Perlin Noise 2D

## Performance
Code compiled with Clang 9.0.0 x64 and ran on Intel i7-8700k.

### 4k texture 8 octaves (average of 100 runs)
| Function | AVX   |
|----------|-------|
| Perlin   | 113ms |

### 16k texture 8 octaves (average of 100 runs)
| Function | AVX    |
|----------|--------|
| Perlin   | 1563ms |
