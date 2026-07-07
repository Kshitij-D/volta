#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __device__ float __symexpf(float x);
#define BLOCK_SIZE 128
#define HEAD_DIM 64
#define VALUE_DIM 64
#define LQ 16
#define LK 512

__global__ void sdpa_basic(const float *__restrict__ Q, // [LQ, D]
                           const float *__restrict__ K, // [LK, D]
                           const float *__restrict__ V, // [LK, DV]
                           float *__restrict__ O,       // [LQ, DV]
                           float scale)                 // 1/sqrt(D)
{
  // only lane 0 does work; launch with <<<LQ, 1>>>
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  for (int q = 0; q < LQ; ++q) {
    float num[VALUE_DIM];
    for (int dv = 0; dv < VALUE_DIM; ++dv)
      num[dv] = 0.f;
    float denom = 0.f;

    // Accumulate over all keys k
    for (int k = 0; k < LK; ++k) {
      // dot = Q[q] · K[k]
      float dot = 0.f;
      for (int d = 0; d < HEAD_DIM; ++d) {
        dot += Q[q * HEAD_DIM + d] * K[k * HEAD_DIM + d];
      }

      // Causal mask via multiply: weight * mask where mask is 1 or 0
      const float w_raw = __symexpf(dot * scale);
      const float mask = (k <= q) ? 1.0f : 0.0f;
      const float w = w_raw * mask;
      denom += w;

      // num += w * V[k]
      const int vbase = k * VALUE_DIM;
      for (int dv = 0; dv < VALUE_DIM; ++dv) {
        num[dv] += w * V[vbase + dv];
      }
    }

    // Write O[q] = num / denom (guard denom)
    const float inv = 1.f / denom;
    const int obase = q * VALUE_DIM;
    for (int dv = 0; dv < VALUE_DIM; ++dv) {
      O[obase + dv] = num[dv] * inv;
    }
  }
}
