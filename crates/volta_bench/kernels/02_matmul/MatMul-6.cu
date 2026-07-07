// Warptiling SGEMM adapted to match matmul_1D signature and macro pattern
// inlined everything

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define M 4096u
#define N 4096u
#define K 4096u
#define BM 64u
#define BN 64u
#define BK 8u
#define Tile 8u
#define TM 8u

#define CEIL_DIV(MACRO_M, MACRO_N) (((MACRO_M) + (MACRO_N) - 1) / (MACRO_N))

const int WARPSIZE = 32;

// Non-templated kernel with the same signature as matmul_1D
__global__ void sgemm(const float *__restrict__ A, const float *__restrict__ B,
                      float *__restrict__ C, float alpha, float beta) {
  constexpr int NUM_THREADS = 256;
  constexpr int TN = 1;
  constexpr int WM = 32; // per-warp tile M
  constexpr int WN = 16; // per-warp tile N
  constexpr int WNITER = 2;
  constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr int WSUBM = WM / WMITER; // 32
  constexpr int WSUBN = WN / WNITER; // 8

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(8/1)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/8

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const float *A0 = A + cRow * BM * K;
  const float *B0 = B + cCol * BN;
  float *C0 = C + (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Load A to shared (transpose while storing)
    for (uint offset = 0; offset < BM; offset += rowStrideA) {
      const uint row = innerRowA + offset;
      if (row < BM) {
        const float4 tmp =
            reinterpret_cast<const float4 *>(&A0[row * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + row] = tmp.x;
        As[(innerColA * 4 + 1) * BM + row] = tmp.y;
        As[(innerColA * 4 + 2) * BM + row] = tmp.z;
        As[(innerColA * 4 + 3) * BM + row] = tmp.w;
      }
    }

    // Load B to shared
    for (uint offset = 0; offset < BK; offset += rowStrideB) {
      const uint row = innerRowB + offset;
      if (row < BK) {
        reinterpret_cast<float4 *>(&Bs[row * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4 *>(&B0[row * N + innerColB * 4])[0];
      }
    }
    __syncthreads();

    // Compute from shared tiles (warptiling)
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint i = 0; i < TM; ++i) {
          regM[wSubRowIdx * TM + i] =
              As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                 threadRowInWarp * TM + i];
        }
      }
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint i = 0; i < TN; ++i) {
          regN[wSubColIdx * TN + i] =
              Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                 threadColInWarp * TN + i];
        }
      }

      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                            (wSubColIdx * TN) + resIdxN] +=
                  regM[wSubRowIdx * TM + resIdxM] *
                  regN[wSubColIdx * TN + resIdxN];
            }
          }
        }
      }
    }

    A0 += BK;
    B0 += BK * N;
    __syncthreads();
  }

  // Write out results (scalar path; TN==1)
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *C_interim = C0 + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          const uint cOff = (threadRowInWarp * TM + resIdxM) * N +
                            threadColInWarp * TN + resIdxN;
          float cval = C_interim[cOff];
          cval = alpha * threadResults[i] + beta * cval;
          C_interim[cOff] = cval;
        }
      }
    }
  }
}
