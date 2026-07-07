// A minimal, correctness-first conv2d kernel matching conv2d.cu shapes
// No WMMA or shared memory; plain loops. Same threads per block and grid tiling.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Keep concrete problem sizes identical to conv2d.cu
#define N_BATCH 100
#define C_IN 3
#define H_IN 224
#define W_IN 224
#define C_OUT 96
#define K_H 11
#define K_W 11
#define STRIDE_H 4
#define STRIDE_W 4
#define PAD_H 2
#define PAD_W 2

#define H_OUT ((H_IN + 2 * PAD_H - K_H) / STRIDE_H + 1)
#define W_OUT ((W_IN + 2 * PAD_W - K_W) / STRIDE_W + 1)

#define M_GEMM (C_OUT)
#define N_GEMM (N_BATCH * H_OUT * W_OUT)
#define K_GEMM (C_IN * K_H * K_W)

// Tile and block config (match conv2d.cu)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define CUDA_WARP_SIZE_CONST 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * CUDA_WARP_SIZE_CONST) // 256
#define BLOCK_M_TILES_WMMA 8
#define BLOCK_N_TILES_WMMA 8
#define TILE_M_PER_BLOCK (BLOCK_M_TILES_WMMA * WMMA_M) // 128
#define TILE_N_PER_BLOCK (BLOCK_N_TILES_WMMA * WMMA_N) // 128

// Tiny helpers to discourage predicate fusion and pointer hoisting.
// Load a single element from input/weight via a volatile pointer.
static __device__ __forceinline__ float ld_input_elem(const float* base,
                                                    int n_batch_idx,
                                                    int ic,
                                                    int h,
                                                    int w) {
  const int idx =
      n_batch_idx * (C_IN * H_IN * W_IN) +
      ic * (H_IN * W_IN) +
      h * W_IN +
      w;
  volatile const float* vp = base + idx; // 32-bit index → PTX mul.wide.s32
  return *vp;
}

static __device__ __forceinline__ float ld_weight_elem(const float* base,
                                                     int oc,
                                                     int ic,
                                                     int kh,
                                                     int kw) {
  const int idx =
      oc * (C_IN * K_H * K_W) +
      ic * (K_H * K_W) +
      kh * K_W +
      kw;
  volatile const float* vp = base + idx; // 32-bit index → PTX mul.wide.s32
  return *vp;
}

__global__ void dumb_conv2d(const float* __restrict__ input_ptr,   // (N, Cin, Hin, Win)
                            const float* __restrict__ weight_ptr,  // (Cout, Cin, Kh, Kw)
                            const float* __restrict__ bias_ptr,    // (Cout)
                            float* __restrict__ output_ptr) {      // (N, Cout, Hout, Wout)
  // Compute this CTA's tile in GEMM space
  const int oc_tile_start = TILE_M_PER_BLOCK * blockIdx.y;  // rows (M)
  const int n_tile_start = TILE_N_PER_BLOCK * blockIdx.x;   // cols (N)

  const int oc_tile_end = min(oc_tile_start + TILE_M_PER_BLOCK, C_OUT);
  const int n_tile_end = min(n_tile_start + TILE_N_PER_BLOCK, N_GEMM);

  const int num_oc = max(0, oc_tile_end - oc_tile_start);
  const int num_n = max(0, n_tile_end - n_tile_start);
  const long long total_elems = 1LL * num_oc * num_n;

  // Strided assignment over the tile
  for (long long lin = threadIdx.x; lin < total_elems; lin += blockDim.x) {
    const int oc_rel = static_cast<int>(lin / num_n);
    const int n_rel = static_cast<int>(lin % num_n);
    const int oc = oc_tile_start + oc_rel;            // 0..C_OUT-1
    const int n_idx = n_tile_start + n_rel;           // 0..N_GEMM-1

    // Decompose N-index → (n_batch_idx, oh_eff, ow_eff)
    const int ow_eff = n_idx % W_OUT;
    const int tmp = n_idx / W_OUT;
    const int oh_eff = tmp % H_OUT;
    const int n_batch_idx = tmp / H_OUT;

    // Compute one output element with padding/stride checks
    float acc = 0.0f;
    // Unconditionally traverse K = C_IN*K_H*K_W with zero-fill for OOB taps.
    // This mirrors the WMMA kernel’s "load-zeros then compute" structure so the
    // symbolic engine sees explicit 0.0 multipliers instead of eliding terms.
    for (int ic = 0; ic < C_IN; ++ic) {
      for (int kh = 0; kh < K_H; ++kh) {
        for (int kw = 0; kw < K_W; ++kw) {
          const int h_in_eff = oh_eff * STRIDE_H - PAD_H + kh;
          const int w_in_eff = ow_eff * STRIDE_W - PAD_W + kw;
          // Zero-fill out-of-bounds input tap (explicit 0.0 term)
          float inp_f = 0.0f;
          if (h_in_eff >= 0 && h_in_eff < H_IN && w_in_eff >= 0 && w_in_eff < W_IN) {
            inp_f = ld_input_elem(input_ptr, n_batch_idx, ic, h_in_eff, w_in_eff);
          }
          const float wgt_f = ld_weight_elem(weight_ptr, oc, ic, kh, kw);
          const __half inp_h = __float2half(inp_f);
          const __half wgt_h = __float2half(wgt_f);
          // Accumulate outer product; keep explicit 0.0 in the expression tree.
          acc += __half2float(__hmul(wgt_h, inp_h));
        }
      }
    }
    acc += bias_ptr[oc];

    // Store to output tensor
    if (oc < C_OUT && n_idx < N_GEMM) {
      output_ptr[1LL * n_batch_idx * C_OUT * H_OUT * W_OUT +
                 1LL * oc * H_OUT * W_OUT +
                 1LL * oh_eff * W_OUT + ow_eff] = acc;
    }
  }
}

// Same simple launcher signature as conv2d.cu
#ifdef CONV_STANDALONE
int main() {
  extern float *d_input, *d_weight, *d_bias, *d_output;

  constexpr int grid_x = (N_GEMM + TILE_N_PER_BLOCK - 1) / TILE_N_PER_BLOCK;
  constexpr int grid_y = (M_GEMM + TILE_M_PER_BLOCK - 1) / TILE_M_PER_BLOCK;

  dim3 grid(grid_x, grid_y);
  dim3 block(THREADS_PER_BLOCK);

  dumb_conv2d<<<grid, block>>>(d_input, d_weight, d_bias, d_output);
  cudaDeviceSynchronize();
  return 0;
}
#endif // CONV_STANDALONE
