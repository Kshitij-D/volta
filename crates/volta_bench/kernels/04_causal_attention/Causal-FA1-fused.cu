#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __device__ float __symexpf(float x);

// ======= problem & tiling =======
// Br: query rows per CTA (row blocking)
// Bc: keys per tile and threads per CTA-x (must be power of two)
#define BLOCKSIZE 128
#define Br 16
#define Bc BLOCKSIZE
#define HEAD_DIM 64
#define VALUE_DIM 64
#define LQ 16
#define LK 512

// ----------------------------------------------
// Block-wide reductions over the Bc threads
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
// FlashAttention-style streaming softmax (Br rows per CTA)
// with causal masking via if-branch
// -------------------------------------------------------
__global__ void
sdpa_fa_like(const float *__restrict__ Q, // [LQ, HEAD_DIM]
             const float *__restrict__ K, // [LK, HEAD_DIM]
             const float *__restrict__ V, // [LK, VALUE_DIM]
             float *__restrict__ O,    // [LQ, VALUE_DIM]  (accumulated output)
             float scale,              // 1/sqrt(HEAD_DIM)
             float *__restrict__ ell,  // [LQ]  running denominator per row
             float *__restrict__ mrow) // [LQ]  running max per row
{
  // ---- Shared scratch (one K/V tile, reused across Br rows) ----
  __shared__ float sK[Bc][HEAD_DIM];
  __shared__ float sV[Bc][VALUE_DIM];
  __shared__ float sQ[HEAD_DIM];
  __shared__ float sLogit[Bc];
  __shared__ float sW[Bc];
  __shared__ float sTmp[Bc];

  // Row-scope scalars (reused for each of the Br rows, one at a time)
  __shared__ float s_row_m; // current row's running max after update
  __shared__ float s_row_l; // current row's running denom after update
  __shared__ float s_alpha; // exp(m_old - m_new) for current row

  const unsigned tid = threadIdx.x; // 0..Bc-1
  const int q_block_start = blockIdx.x * Br;

  // ---- Outer loop over K/V tiles (K/V reuse) ----
  for (int k_start = 0; k_start < LK; k_start += Bc) {
    const int k_idx = k_start + tid;

    // Load K/V tile once per CTA (each thread loads one row)
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

    // ---- Inner loop over Br query rows handled by this CTA ----
    for (int r = 0; r < Br; ++r) {
      const int q_idx = q_block_start + r;
      if (q_idx >= LQ)
        continue; // tail

      // Load Q[q_idx] cooperatively
      for (int d = tid; d < HEAD_DIM; d += Bc)
        sQ[d] = Q[q_idx * HEAD_DIM + d];
      __syncthreads();

      // Load this row's running state from HBM (state write-back policy)
      // --- Device-side initialization on the first K/V tile ---
      if (k_start == 0) {
        // start from a clean state: m0=-inf, l0=0, O0=0
        if (tid == 0) {
          s_row_m = -CUDART_INF_F;
          s_row_l = 0.f;
        }
        for (int dv = tid; dv < VALUE_DIM; dv += Bc)
          O[q_idx * VALUE_DIM + dv] = 0.f;
      } else {
        // subsequent tiles: read the running state produced by previous tile
        if (tid == 0) {
          s_row_l = ell[q_idx];
          s_row_m = mrow[q_idx];
        }
      }
      __syncthreads();

      // Compute logits for this tile: S = Q_i * K_j^T
      float logit = -CUDART_INF_F;
      if (k_idx < LK) {
        float dot = 0.f;
        for (int d = 0; d < HEAD_DIM; ++d)
          dot += sQ[d] * sK[tid][d];
        logit = dot * scale;
      }
      // Causal mask (if-branch): mask future keys by setting logit to -inf
      if (k_idx > q_idx)
        logit = -CUDART_INF_F;
      sLogit[tid] = logit;

      // Tile rowmax and online softmax rescale params
      float tmax = tile_max(logit, sTmp);
      if (tid == 0) {
        const float M_old = s_row_m;
        const float M_new = fmaxf(M_old, tmax);
        s_alpha = __symexpf(M_old - M_new); // rescale factor
        s_row_m = M_new;
        // rescale denom (ℓ) here; O is rescaled lazily per dv element below
        s_row_l *= s_alpha;
      }
      __syncthreads();

      // Weights in the new max frame: exp(S - M_new)
      const float alpha = s_alpha;
      float w = 0.f;
      if (k_idx < LK)
        w = __symexpf(sLogit[tid] - s_row_m);
      // Causal mask (if-branch): zero out weight for future keys
      if (k_idx > q_idx)
        w = 0.0f;
      sW[tid] = w;

      // Update denom
      float tsum = tile_sum(w, sTmp);
      if (tid == 0)
        s_row_l += tsum;
      __syncthreads();

      // Update O row: O = O * alpha + (P_tilde V) for this tile
      for (int dv = tid; dv < VALUE_DIM; dv += Bc) {
        float acc = 0.f;
        for (int j = 0; j < Bc; ++j) {
          const int kk = k_start + j;
          if (kk < LK)
            acc += sW[j] * sV[j][dv];
        }
        const int o_off = q_idx * VALUE_DIM + dv;
        const float old_o = O[o_off];
        O[o_off] = old_o * alpha + acc; // write updated slice to HBM
      }
      __syncthreads();

      // Store updated ℓ and m for this row back to HBM
      if (tid == 0) {
        ell[q_idx] = s_row_l;
        mrow[q_idx] = s_row_m;
      }
      __syncthreads();
    } // end for r in Br
  } // end for k_start in tiles

  // Final normalization (convert O from "unnormalized" to softmax*V)
  // Do it once after all K/V tiles are consumed.
  for (int r = 0; r < Br; ++r) {
    const int q_idx = q_block_start + r;
    if (q_idx >= LQ)
      continue;
    // Read ℓ[q] once; divide O[q,:] in place
    float denom = 0.f;
    if (tid == 0)
      denom = ell[q_idx];
    __syncthreads();
    // broadcast denom via shared memory sTmp[0]
    if (tid == 0)
      sTmp[0] = denom;
    __syncthreads();
    const float d = sTmp[0];
    for (int dv = tid; dv < VALUE_DIM; dv += Bc) {
      const int o_off = q_idx * VALUE_DIM + dv;
      O[o_off] = O[o_off] / d;
    }
    __syncthreads();
  }
}
