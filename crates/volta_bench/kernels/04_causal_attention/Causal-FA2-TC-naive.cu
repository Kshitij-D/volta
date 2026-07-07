#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <mma.h>

using namespace nvcuda;

extern "C" __device__ float __symexpf(float x);

// ======= problem & tiling =======
// Br : query rows per CTA (must be 16 for WMMA m-dim)
// Bc : keys per tile (multiple of 16)
// HEAD_DIM may be 8/16/... (QK^T done with FMAs here)
// VALUE_DIM must be multiple of 16 for WMMA n-dim
#define BLOCKSIZE 128
#define Br 16
#define Bc 128
#define HEAD_DIM 64
#define VALUE_DIM 64
#define LQ 16
#define LK 512
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCKSIZE / WARP_SIZE)

static_assert((BLOCKSIZE % WARP_SIZE) == 0,
              "BLOCKSIZE must be a multiple of the warp size.");

// ----------------------------------------------
// Block-wide reductions over the Bc threads
// (unchanged, used for per-row reductions)
// ----------------------------------------------
__device__ float tile_max(float x, volatile float *s) {
  const unsigned tid = threadIdx.x;
  s[tid] = x;
  __syncthreads();
  for (unsigned stride = (blockDim.x >> 1); stride > 0; stride >>= 1) {
    if (tid < stride)
      s[tid] = fmaxf(s[tid], s[tid + stride]);
    __syncthreads();
  }
  return s[0];
}

__device__ float tile_sum(float x, volatile float *s) {
  const unsigned tid = threadIdx.x;
  s[tid] = x;
  __syncthreads();
  for (unsigned stride = (blockDim.x >> 1); stride > 0; stride >>= 1) {
    if (tid < stride)
      s[tid] += s[tid + stride];
    __syncthreads();
  }
  return s[0];
}

// -------------------------------------------------------
// FlashAttention-v2 style streaming softmax + TC for P·V
// One CTA handles Br=16 query rows (single warp)
// -------------------------------------------------------
__global__ void sdpa_fa2_tc(const float *__restrict__ Q, // [LQ, HEAD_DIM]
                            const float *__restrict__ K, // [LK, HEAD_DIM]
                            const float *__restrict__ V, // [LK, VALUE_DIM]
                            float *__restrict__ O, // [LQ, VALUE_DIM] (final)
                            float scale)           // 1/sqrt(HEAD_DIM)
{
  // Sanity for this simplified WMMA path
  static_assert(Br == 16, "This WMMA path requires Br==16.");
  static_assert((Bc % 16) == 0, "Bc must be multiple of 16.");
  static_assert((VALUE_DIM % 16) == 0, "VALUE_DIM must be multiple of 16.");

  const int tid = threadIdx.x; // 0..BLOCKSIZE-1
  const int warp_id = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;
  const int q_block_start = blockIdx.x * Br;

  // ---- Shared tiles (one K/V tile reused across Br rows) ----
  __shared__ float sK[Bc][HEAD_DIM];  // FP32 for logits
  __shared__ float sV[Bc][VALUE_DIM]; // FP32 -> downcast for TC
  __shared__ float sQ[Br][HEAD_DIM];  // the Br rows in this CTA
  __shared__ float sLogit[Br][Bc];    // logits for current tile
  __shared__ float sW[Br][Bc];        // weights exp(...) for tile
  __shared__ float sTmp[BLOCKSIZE];   // reductions
  __shared__ float sRowMax[Br];       // tile rowmax
  __shared__ float sRowSum[Br];       // tile rowsum of weights
  __shared__ __align__(16) half sA[WARPS_PER_BLOCK * Br * 16];
  __shared__ __align__(16) half sB[WARPS_PER_BLOCK * 16 * 16];
  __shared__ float sC[WARPS_PER_BLOCK * Br * 16];

  // Running state kept on-chip (FA-v2)
  __shared__ float sM[Br];            // running max per row
  __shared__ float sL[Br];            // running denom per row
  __shared__ float sO[Br][VALUE_DIM]; // running O per row

  // ---- Initialize running state once (no HBM ping-pong) ----
  if (tid < Br) {
    sM[tid] = -CUDART_INF_F;
    sL[tid] = 0.f;
  }
  for (int dv = tid; dv < Br * VALUE_DIM; dv += BLOCKSIZE) {
    reinterpret_cast<float *>(sO)[dv] = 0.f;
  }
  __syncthreads();

  // ---- Bring Br rows of Q for this CTA to shared ----
  // Each thread helps load; stride by warp size
  for (int r = tid; r < Br * HEAD_DIM; r += BLOCKSIZE) {
    int rr = r / HEAD_DIM, dd = r % HEAD_DIM;
    int q_idx = q_block_start + rr;
    if (q_idx < LQ)
      sQ[rr][dd] = Q[q_idx * HEAD_DIM + dd];
    else
      sQ[rr][dd] = 0.f;
  }
  __syncthreads();

  // ---- Outer loop over K/V tiles ----
  for (int k_start = 0; k_start < LK; k_start += Bc) {
    const int k_idx = k_start + tid;

    const int valid_cols = max(0, min(Bc, LK - k_start));
    // Load K and V tile (one row per thread)
    if (k_idx < LK) {
      for (int d = 0; d < HEAD_DIM; ++d)
        sK[tid][d] = K[k_idx * HEAD_DIM + d];
      for (int dv = 0; dv < VALUE_DIM; ++dv)
        sV[tid][dv] = V[k_idx * VALUE_DIM + dv];
    } else {
      for (int d = 0; d < HEAD_DIM; ++d)
        sK[tid][d] = 0.f;
      for (int dv = 0; dv < VALUE_DIM; ++dv)
        sV[tid][dv] = 0.f;
    }
    __syncthreads();

    // ---- Compute logits for this tile: [Br x Bc] = Q[Br x d] * K[Bc x d]^T
    for (int r = 0; r < Br; ++r) {
      const int q_idx = q_block_start + r;
      // each thread handles columns c = tid, tid+32, ...
      for (int c = tid; c < Bc; c += BLOCKSIZE) {
        float logit;
        if (c < valid_cols) {
          float dot = 0.f;
          for (int d = 0; d < HEAD_DIM; ++d)
            dot += sQ[r][d] * sK[c][d];
          logit = dot * scale;
        } else {
          // Mask padded columns: exp(-inf) = 0, they vanish from max/sum
          logit = -CUDART_INF_F;
        }
        // Causal mask (branchless): select logit or -inf via ternary
        const int k_idx_c = k_start + c;
        logit = (k_idx_c <= q_idx) ? logit : -CUDART_INF_F;
        sLogit[r][c] = logit;
      }
    }
    __syncthreads();

    // ---- Tile rowmax & online rescale (FA-v2 style) ----
    for (int r = 0; r < Br; ++r) {
      // compute tmax over this tile’s Bc keys for row r
      float local = -CUDART_INF_F;
      for (int c = tid; c < Bc; c += BLOCKSIZE)
        local = fmaxf(local, sLogit[r][c]);
      float tmax = tile_max(local, sTmp); // reduce across warp
      if (tid == 0) {
        sRowMax[r] = tmax;
      }
      __syncthreads();

      if (tid == 0) {
        const float M_old = sM[r];
        const float M_new = fmaxf(M_old, sRowMax[r]);
        const float alpha = __symexpf(M_old - M_new); // rescale factor
        sM[r] = M_new;
        // rescale denom and O for this row (lazy, FA-v2)
        sL[r] *= alpha;
        for (int dv = 0; dv < VALUE_DIM; ++dv)
          sO[r][dv] *= alpha;
      }
      __syncthreads();
    }

    // ---- Compute tile weights w = exp(logit - M_new) and row sums ----
    for (int r = 0; r < Br; ++r) {
      const int q_idx = q_block_start + r;
      float tsum_local = 0.f;
      const float m = sM[r];
      for (int c = tid; c < Bc; c += BLOCKSIZE) {
        float w = __symexpf(sLogit[r][c] - m);
        // Causal mask (branchless): multiply weight by 0/1 mask
        const int k_idx_c = k_start + c;
        const float causal_mask = (k_idx_c <= q_idx) ? 1.0f : 0.0f;
        w = w * causal_mask;
        sW[r][c] = w;
        tsum_local += w;
      }
      float tsum = tile_sum(tsum_local, sTmp);
      if (tid == 0)
        sRowSum[r] = tsum;
      __syncthreads();
      if (tid == 0)
        sL[r] += sRowSum[r];
      __syncthreads();
    }

    // ---- Tensor Core: accumulate (P_tile[Br x Bc]) · (V_tile[Bc x VALUE_DIM])
    // We do Bc in 16-chunks and VALUE_DIM in 16-chunks
    for (int dv_base = 0; dv_base < VALUE_DIM; dv_base += 16) {
      const int tile_owner = (dv_base / 16) % WARPS_PER_BLOCK;
      half *A_warp = &sA[warp_id * Br * 16];
      half *B_warp = &sB[warp_id * 16 * 16];
      float *c_tile_warp = &sC[warp_id * Br * 16];

      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      if (warp_id == tile_owner)
        wmma::fill_fragment(c_frag, 0.0f);
      __syncthreads();

      for (int kk = 0; kk < Bc; kk += 16) {
        if (warp_id == tile_owner) {
          for (int idx = lane; idx < Br * 16; idx += WARP_SIZE) {
            int r = idx / 16, c = idx % 16;
            float w = sW[r][kk + c];
            A_warp[idx] = __float2half(w);
          }
          for (int idx = lane; idx < 16 * 16; idx += WARP_SIZE) {
            int r = idx / 16, c = idx % 16;
            float v = sV[kk + r][dv_base + c];
            B_warp[idx] = __float2half(v);
          }
        }
        __syncthreads();

        if (warp_id == tile_owner) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
              a_frag;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
              b_frag;
          wmma::load_matrix_sync(a_frag, A_warp, 16);
          wmma::load_matrix_sync(b_frag, B_warp, 16);
          wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
      }

      if (warp_id == tile_owner) {
        wmma::store_matrix_sync(c_tile_warp, c_frag, 16,
                                wmma::mem_row_major);
      }
      __syncthreads();

      if (warp_id == tile_owner) {
        for (int idx = lane; idx < Br * 16; idx += WARP_SIZE) {
          int r = idx / 16, c = idx % 16;
          sO[r][dv_base + c] += c_tile_warp[idx];
        }
      }
      __syncthreads();
    }
  } // end K/V tiles

  // ---- Final normalization: O = O / ℓ ----
  for (int r = 0; r < Br; ++r) {
    float denom = 0.f;
    if (tid == 0)
      denom = sL[r];
    __syncthreads();
    if (tid == 0)
      sTmp[0] = denom;
    __syncthreads();
    const float d = sTmp[0];
    const int q_idx = q_block_start + r;
    if (q_idx < LQ) {
      for (int dv = tid; dv < VALUE_DIM; dv += BLOCKSIZE) {
        O[q_idx * VALUE_DIM + dv] = sO[r][dv] / d;
      }
    }
    __syncthreads();
  }
}
