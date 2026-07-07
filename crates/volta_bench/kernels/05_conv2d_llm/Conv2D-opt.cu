#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

struct NDecomposed {
  int ow_eff;
  int oh_eff;
  int n_batch_idx;
  bool isValidPixel; // True if this pixel_idx is within N_gemm bounds
  int h_in_base;
  int w_in_base;
};

/* ------------------------------------------------------------------------- */
/* 1. CONCRETE PROBLEM SIZES (edit these if you change the workload)         */
/* ------------------------------------------------------------------------- */
#ifndef N_BATCH
#define N_BATCH 100
#endif
#ifndef C_IN
#define C_IN 3
#endif
#ifndef H_IN
#define H_IN 224
#endif
#ifndef W_IN
#define W_IN 224
#endif
#ifndef C_OUT
#define C_OUT 96
#endif
#ifndef K_H
#define K_H 11
#endif
#ifndef K_W
#define K_W 11
#endif
#ifndef STRIDE_H
#define STRIDE_H 4
#endif
#ifndef STRIDE_W
#define STRIDE_W 4
#endif
#ifndef PAD_H
#define PAD_H 2
#endif
#ifndef PAD_W
#define PAD_W 2
#endif

#define H_OUT ((H_IN + 2 * PAD_H - K_H) / STRIDE_H + 1)
#define W_OUT ((W_IN + 2 * PAD_W - K_W) / STRIDE_W + 1)

#define M_GEMM (C_OUT)
#define N_GEMM (N_BATCH * H_OUT * W_OUT)
#define K_GEMM (C_IN * K_H * K_W)

/* ------------------------------------------------------------------------- */
/* 2. WMMA / TILE CONSTANTS – unchanged                                      */
/* ------------------------------------------------------------------------- */
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define SKEW_HALF 8
#define CUDA_WARP_SIZE_CONST 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * CUDA_WARP_SIZE_CONST)
#define BLOCK_M_TILES_WMMA 8
#define BLOCK_N_TILES_WMMA 8
#define TILE_M_PER_BLOCK (BLOCK_M_TILES_WMMA * WMMA_M) // 128
#define TILE_N_PER_BLOCK (BLOCK_N_TILES_WMMA * WMMA_N) // 128
#define VECTOR_SIZE_H2 2
// Always provide some tail padding slots for n_params_sh so speculative
// prefetch into the next rows is in-bounds. Keep this independent of any
// debug scribbling on shared tiles.
/* ------------------------------------------------------------------------- */
/* 3. *Alias* original parameter names to the concrete macros                */
/*    --> keeps the body of the kernel totally unchanged.                    */
/* ------------------------------------------------------------------------- */
#define N_batch N_BATCH
#define C_in C_IN
#define H_in H_IN
#define W_in W_IN
#define C_out C_OUT
#define K_h K_H
#define K_w K_W
#define stride_h STRIDE_H
#define stride_w STRIDE_W
#define pad_h PAD_H
#define pad_w PAD_W
#define H_out H_OUT
#define W_out W_OUT
#define M_gemm M_GEMM
#define N_gemm N_GEMM
#define K_gemm K_GEMM

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s failed: %s\n", #call,                           \
                   cudaGetErrorString(err_));                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void
conv2d(const float *__restrict__ input_ptr,  // Input: (N, Cin, Hin, Win)
       const float *__restrict__ weight_ptr, // Weights: (Cout, Cin, Kh, Kw)
       const float *__restrict__ bias_ptr,   // Bias: (Cout) or nullptr
       float *__restrict__ output_ptr)       // Output: (N, Cout, Hout, Wout)
{
  // Thread identification
  const int warp_id = threadIdx.x / warpSize; // 0 .. WARPS_PER_BLOCK-1
  const int lane_id = threadIdx.x % warpSize; // 0 .. 31 (or warpSize-1)

  // Top-left corner of the macro-tile this block is responsible for in GEMM
  // terms
  const int block_row_gemm_start = TILE_M_PER_BLOCK * blockIdx.y;
  const int block_col_gemm_start = TILE_N_PER_BLOCK * blockIdx.x;

  // Shared memory for tiles of A (weights) and B (input/im2col) - Double
  // Buffered for K-loop pipelining
  __shared__ half Asub_pipe[2][TILE_M_PER_BLOCK][WMMA_K + SKEW_HALF];
  __shared__ half Bsub_pipe[2][TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF];

  // Shared memory for precomputed N-indices
  __shared__ NDecomposed n_params_sh[TILE_N_PER_BLOCK];

  // Shared memory for output stage (per-warp buffers)
  __shared__ float C_shmem_output_buffers[WARPS_PER_BLOCK][WMMA_M][WMMA_N];

  // Accumulator fragments per warp.
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      acc_frag[BLOCK_N_TILES_WMMA];
  // #pragma unroll
  for (int i = 0; i < BLOCK_N_TILES_WMMA; ++i) {
    wmma::fill_fragment(acc_frag[i], 0.0f);
  }

  // Populate n_params_sh once at the beginning of the kernel
  if (threadIdx.x < TILE_N_PER_BLOCK) {
    int r_b_tile_idx = threadIdx.x;
    int current_pixel_idx = block_col_gemm_start + r_b_tile_idx;

    if (current_pixel_idx < N_gemm) {
      n_params_sh[r_b_tile_idx].ow_eff = current_pixel_idx % W_out;
      int temp_div_wout = current_pixel_idx / W_out;
      n_params_sh[r_b_tile_idx].oh_eff = temp_div_wout % H_out;
      n_params_sh[r_b_tile_idx].n_batch_idx = temp_div_wout / H_out;
      n_params_sh[r_b_tile_idx].isValidPixel = true;

      n_params_sh[r_b_tile_idx].h_in_base =
          n_params_sh[r_b_tile_idx].oh_eff * stride_h - pad_h;
      n_params_sh[r_b_tile_idx].w_in_base =
          n_params_sh[r_b_tile_idx].ow_eff * stride_w - pad_w;
    } else {
      n_params_sh[r_b_tile_idx].isValidPixel = false;
      n_params_sh[r_b_tile_idx].ow_eff = 0;
      n_params_sh[r_b_tile_idx].oh_eff = 0;
      n_params_sh[r_b_tile_idx].n_batch_idx = 0;
      n_params_sh[r_b_tile_idx].h_in_base = 0;
      n_params_sh[r_b_tile_idx].w_in_base = 0;
    }
  }
  __syncthreads();

  // Constants for vectorized shared memory loading
  // Number of half2 elements along K-dim for a shared memory tile row
  const int NUM_H2_ELEMENTS_IN_K_DIM = WMMA_K / VECTOR_SIZE_H2;
  // Number of thread groups, where each group has NUM_H2_ELEMENTS_IN_K_DIM
  // threads. Each group is responsible for loading the K-dimension for one
  // M-row (for A) or N-row (for B) at a time, iterating over M-rows or N-rows
  // with this step size.
  const int NUM_ROW_PROCESSING_GROUPS =
      THREADS_PER_BLOCK / NUM_H2_ELEMENTS_IN_K_DIM;

  // --- K-Loop Pipelining ---
  int num_k_tiles = (K_gemm + WMMA_K - 1) / WMMA_K;

  // --- Prologue: Load first k-tile (k_tile_iter = 0) into pipe_idx = 0 ---
  if (num_k_tiles > 0) {
    int k_tile_start_prologue = 0;
    int current_pipe_idx_prologue = 0;

    // Load Asub_pipe[0] for k_tile_iter = 0
    {
      // This thread is responsible for the 'h2_idx_in_k_dim_A'-th half2 element
      // in the K-dimension of the shared memory tile.
      int h2_idx_in_k_dim_A = threadIdx.x % NUM_H2_ELEMENTS_IN_K_DIM;
      // Starting 'half' index in shared memory for this half2 write.
      int shmem_k_start_for_h2_A = h2_idx_in_k_dim_A * VECTOR_SIZE_H2;

      // Global k-indices for the two half elements.
      int k_global_A_0 = k_tile_start_prologue + shmem_k_start_for_h2_A;
      int k_global_A_1 = k_tile_start_prologue + shmem_k_start_for_h2_A + 1;

      // Decompose k_global_A_0
      int kw_eff_reg_A_0 = 0, kh_eff_reg_A_0 = 0, ic_eff_reg_A_0 = 0;
      bool is_valid_k_A_0 = (k_global_A_0 < K_gemm);
      if (is_valid_k_A_0) {
        kw_eff_reg_A_0 = k_global_A_0 % K_w;
        int temp_div_kw_A_0 = k_global_A_0 / K_w;
        kh_eff_reg_A_0 = temp_div_kw_A_0 % K_h;
        ic_eff_reg_A_0 = temp_div_kw_A_0 / K_h;
      }

      // Decompose k_global_A_1
      int kw_eff_reg_A_1 = 0, kh_eff_reg_A_1 = 0, ic_eff_reg_A_1 = 0;
      bool is_valid_k_A_1 = (k_global_A_1 < K_gemm);
      if (is_valid_k_A_1) {
        kw_eff_reg_A_1 = k_global_A_1 % K_w;
        int temp_div_kw_A_1 = k_global_A_1 / K_w;
        kh_eff_reg_A_1 = temp_div_kw_A_1 % K_h;
        ic_eff_reg_A_1 = temp_div_kw_A_1 / K_h;
      }

      // This thread belongs to 'm_row_group_id_A'-th group of threads.
      // This group iterates over M-rows of the Asub_pipe tile.
      int m_row_group_id_A = threadIdx.x / NUM_H2_ELEMENTS_IN_K_DIM;
      for (int r_a_tile_base = m_row_group_id_A;
           r_a_tile_base < TILE_M_PER_BLOCK;
           r_a_tile_base += NUM_ROW_PROCESSING_GROUPS) {
        int oc_idx = block_row_gemm_start + r_a_tile_base;
        float weight_val_0 = 0.0f;
        if (oc_idx < C_out && is_valid_k_A_0) {
          weight_val_0 = weight_ptr[oc_idx * C_in * K_h * K_w +
                                    ic_eff_reg_A_0 * K_h * K_w +
                                    kh_eff_reg_A_0 * K_w + kw_eff_reg_A_0];
        }
        float weight_val_1 = 0.0f;
        if (oc_idx < C_out && is_valid_k_A_1) {
          weight_val_1 = weight_ptr[oc_idx * C_in * K_h * K_w +
                                    ic_eff_reg_A_1 * K_h * K_w +
                                    kh_eff_reg_A_1 * K_w + kw_eff_reg_A_1];
        }
        half2 *smem_ptr_h2_A = reinterpret_cast<half2 *>(
            &Asub_pipe[current_pipe_idx_prologue][r_a_tile_base]
                      [shmem_k_start_for_h2_A]);
        *smem_ptr_h2_A =
            make_half2(__float2half(weight_val_0), __float2half(weight_val_1));
      }
    }

    // Load Bsub_pipe[0] for k_tile_iter = 0
    {
      int h2_idx_in_k_dim_B = threadIdx.x % NUM_H2_ELEMENTS_IN_K_DIM;
      int shmem_k_start_for_h2_B = h2_idx_in_k_dim_B * VECTOR_SIZE_H2;

      int k_global_B_0 = k_tile_start_prologue + shmem_k_start_for_h2_B;
      int k_global_B_1 = k_tile_start_prologue + shmem_k_start_for_h2_B + 1;

      int kw_eff_reg_B_0 = 0, kh_eff_reg_B_0 = 0, ic_eff_reg_B_0 = 0;
      bool is_valid_k_B_0 = (k_global_B_0 < K_gemm);
      if (is_valid_k_B_0) {
        kw_eff_reg_B_0 = k_global_B_0 % K_w;
        int temp_div_kw_B_0 = k_global_B_0 / K_w;
        kh_eff_reg_B_0 = temp_div_kw_B_0 % K_h;
        ic_eff_reg_B_0 = temp_div_kw_B_0 / K_h;
      }

      int kw_eff_reg_B_1 = 0, kh_eff_reg_B_1 = 0, ic_eff_reg_B_1 = 0;
      bool is_valid_k_B_1 = (k_global_B_1 < K_gemm);
      if (is_valid_k_B_1) {
        kw_eff_reg_B_1 = k_global_B_1 % K_w;
        int temp_div_kw_B_1 = k_global_B_1 / K_w;
        kh_eff_reg_B_1 = temp_div_kw_B_1 % K_h;
        ic_eff_reg_B_1 = temp_div_kw_B_1 / K_h;
      }

      int n_row_group_id_B = threadIdx.x / NUM_H2_ELEMENTS_IN_K_DIM;
      for (int r_b_tile_base = n_row_group_id_B;
           r_b_tile_base < TILE_N_PER_BLOCK;
           r_b_tile_base += NUM_ROW_PROCESSING_GROUPS) {
        float input_val_0 = 0.0f;
        if (n_params_sh[r_b_tile_base].isValidPixel && is_valid_k_B_0) {
          const NDecomposed &current_n_params = n_params_sh[r_b_tile_base];
          int h_in_eff_0 = current_n_params.h_in_base + kh_eff_reg_B_0;
          int w_in_eff_0 = current_n_params.w_in_base + kw_eff_reg_B_0;
          if (h_in_eff_0 >= 0 && h_in_eff_0 < H_in && w_in_eff_0 >= 0 &&
              w_in_eff_0 < W_in) {
            input_val_0 =
                input_ptr[current_n_params.n_batch_idx * C_in * H_in * W_in +
                          ic_eff_reg_B_0 * H_in * W_in + h_in_eff_0 * W_in +
                          w_in_eff_0];
          }
        }
        float input_val_1 = 0.0f;
        if (n_params_sh[r_b_tile_base].isValidPixel && is_valid_k_B_1) {
          const NDecomposed &current_n_params = n_params_sh[r_b_tile_base];
          int h_in_eff_1 = current_n_params.h_in_base + kh_eff_reg_B_1;
          int w_in_eff_1 = current_n_params.w_in_base + kw_eff_reg_B_1;
          if (h_in_eff_1 >= 0 && h_in_eff_1 < H_in && w_in_eff_1 >= 0 &&
              w_in_eff_1 < W_in) {
            input_val_1 =
                input_ptr[current_n_params.n_batch_idx * C_in * H_in * W_in +
                          ic_eff_reg_B_1 * H_in * W_in + h_in_eff_1 * W_in +
                          w_in_eff_1];
          }
        }
        half2 *smem_ptr_h2_B = reinterpret_cast<half2 *>(
            &Bsub_pipe[current_pipe_idx_prologue][r_b_tile_base]
                      [shmem_k_start_for_h2_B]);
        *smem_ptr_h2_B =
            make_half2(__float2half(input_val_0), __float2half(input_val_1));
      }
    }
  }

  // Loop over the K_gemm dimension in tiles of WMMA_K
  for (int k_tile_iter = 0; k_tile_iter < num_k_tiles; ++k_tile_iter) {
    __syncthreads(); // Sync point for pipelining

    int compute_pipe_idx = k_tile_iter % 2;
    int load_pipe_idx = (k_tile_iter + 1) % 2;

    // --- Load Stage for next k-tile (k_tile_iter + 1) into load_pipe_idx ---
    int k_tile_start_for_load = (k_tile_iter + 1) * WMMA_K;
    if (k_tile_start_for_load < K_gemm) {
      // Load Asub_pipe[load_pipe_idx]
      {
        int h2_idx_in_k_dim_A = threadIdx.x % NUM_H2_ELEMENTS_IN_K_DIM;
        int shmem_k_start_for_h2_A = h2_idx_in_k_dim_A * VECTOR_SIZE_H2;

        int k_global_A_0 = k_tile_start_for_load + shmem_k_start_for_h2_A;
        int k_global_A_1 = k_tile_start_for_load + shmem_k_start_for_h2_A + 1;

        int kw_eff_reg_A_0 = 0, kh_eff_reg_A_0 = 0, ic_eff_reg_A_0 = 0;
        bool is_valid_k_A_0 = (k_global_A_0 < K_gemm);
        if (is_valid_k_A_0) {
          kw_eff_reg_A_0 = k_global_A_0 % K_w;
          int temp_div_kw_A_0 = k_global_A_0 / K_w;
          kh_eff_reg_A_0 = temp_div_kw_A_0 % K_h;
          ic_eff_reg_A_0 = temp_div_kw_A_0 / K_h;
        }

        int kw_eff_reg_A_1 = 0, kh_eff_reg_A_1 = 0, ic_eff_reg_A_1 = 0;
        bool is_valid_k_A_1 = (k_global_A_1 < K_gemm);
        if (is_valid_k_A_1) {
          kw_eff_reg_A_1 = k_global_A_1 % K_w;
          int temp_div_kw_A_1 = k_global_A_1 / K_w;
          kh_eff_reg_A_1 = temp_div_kw_A_1 % K_h;
          ic_eff_reg_A_1 = temp_div_kw_A_1 / K_h;
        }

        int m_row_group_id_A = threadIdx.x / NUM_H2_ELEMENTS_IN_K_DIM;
        for (int r_a_tile_base = m_row_group_id_A;
             r_a_tile_base < TILE_M_PER_BLOCK;
             r_a_tile_base += NUM_ROW_PROCESSING_GROUPS) {
          int oc_idx = block_row_gemm_start + r_a_tile_base;
          float weight_val_0 = 0.0f;
          if (oc_idx < C_out && is_valid_k_A_0) {
            weight_val_0 = weight_ptr[oc_idx * C_in * K_h * K_w +
                                      ic_eff_reg_A_0 * K_h * K_w +
                                      kh_eff_reg_A_0 * K_w + kw_eff_reg_A_0];
          }
          float weight_val_1 = 0.0f;
          if (oc_idx < C_out && is_valid_k_A_1) {
            weight_val_1 = weight_ptr[oc_idx * C_in * K_h * K_w +
                                      ic_eff_reg_A_1 * K_h * K_w +
                                      kh_eff_reg_A_1 * K_w + kw_eff_reg_A_1];
          }
          half2 *smem_ptr_h2_A = reinterpret_cast<half2 *>(
              &Asub_pipe[load_pipe_idx][r_a_tile_base][shmem_k_start_for_h2_A]);
          *smem_ptr_h2_A = make_half2(__float2half(weight_val_0),
                                      __float2half(weight_val_1));
        }
      }

      // Load Bsub_pipe[load_pipe_idx]
      {
        int h2_idx_in_k_dim_B = threadIdx.x % NUM_H2_ELEMENTS_IN_K_DIM;
        int shmem_k_start_for_h2_B = h2_idx_in_k_dim_B * VECTOR_SIZE_H2;

        int k_global_B_0 = k_tile_start_for_load + shmem_k_start_for_h2_B;
        int k_global_B_1 = k_tile_start_for_load + shmem_k_start_for_h2_B + 1;

        int kw_eff_reg_B_0 = 0, kh_eff_reg_B_0 = 0, ic_eff_reg_B_0 = 0;
        bool is_valid_k_B_0 = (k_global_B_0 < K_gemm);
        if (is_valid_k_B_0) {
          kw_eff_reg_B_0 = k_global_B_0 % K_w;
          int temp_div_kw_B_0 = k_global_B_0 / K_w;
          kh_eff_reg_B_0 = temp_div_kw_B_0 % K_h;
          ic_eff_reg_B_0 = temp_div_kw_B_0 / K_h;
        }

        int kw_eff_reg_B_1 = 0, kh_eff_reg_B_1 = 0, ic_eff_reg_B_1 = 0;
        bool is_valid_k_B_1 = (k_global_B_1 < K_gemm);
        if (is_valid_k_B_1) {
          kw_eff_reg_B_1 = k_global_B_1 % K_w;
          int temp_div_kw_B_1 = k_global_B_1 / K_w;
          kh_eff_reg_B_1 = temp_div_kw_B_1 % K_h;
          ic_eff_reg_B_1 = temp_div_kw_B_1 / K_h;
        }

        int n_row_group_id_B = threadIdx.x / NUM_H2_ELEMENTS_IN_K_DIM;
        for (int r_b_tile_base = n_row_group_id_B;
             r_b_tile_base < TILE_N_PER_BLOCK;
             r_b_tile_base += NUM_ROW_PROCESSING_GROUPS) {
          float input_val_0 = 0.0f;
          if (n_params_sh[r_b_tile_base].isValidPixel && is_valid_k_B_0) {
            const NDecomposed &current_n_params = n_params_sh[r_b_tile_base];
            int h_in_eff_0 = current_n_params.h_in_base + kh_eff_reg_B_0;
            int w_in_eff_0 = current_n_params.w_in_base + kw_eff_reg_B_0;
            if (h_in_eff_0 >= 0 && h_in_eff_0 < H_in && w_in_eff_0 >= 0 &&
                w_in_eff_0 < W_in) {
              input_val_0 =
                  input_ptr[current_n_params.n_batch_idx * C_in * H_in * W_in +
                            ic_eff_reg_B_0 * H_in * W_in + h_in_eff_0 * W_in +
                            w_in_eff_0];
            }
          }
          float input_val_1 = 0.0f;
          if (n_params_sh[r_b_tile_base].isValidPixel && is_valid_k_B_1) {
            const NDecomposed &current_n_params = n_params_sh[r_b_tile_base];
            int h_in_eff_1 = current_n_params.h_in_base + kh_eff_reg_B_1;
            int w_in_eff_1 = current_n_params.w_in_base + kw_eff_reg_B_1;
            if (h_in_eff_1 >= 0 && h_in_eff_1 < H_in && w_in_eff_1 >= 0 &&
                w_in_eff_1 < W_in) {
              input_val_1 =
                  input_ptr[current_n_params.n_batch_idx * C_in * H_in * W_in +
                            ic_eff_reg_B_1 * H_in * W_in + h_in_eff_1 * W_in +
                            w_in_eff_1];
            }
          }
          half2 *smem_ptr_h2_B = reinterpret_cast<half2 *>(
              &Bsub_pipe[load_pipe_idx][r_b_tile_base][shmem_k_start_for_h2_B]);
          *smem_ptr_h2_B =
              make_half2(__float2half(input_val_0), __float2half(input_val_1));
        }
      }
    }

    // --- Compute Stage for current k-tile (k_tile_iter) using compute_pipe_idx
    // ---
    int a_row_start_in_tile = warp_id * WMMA_M;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        a_frag;
    wmma::load_matrix_sync(a_frag,
                           &Asub_pipe[compute_pipe_idx][a_row_start_in_tile][0],
                           WMMA_K + SKEW_HALF);

    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::col_major>
        b_frag_inner_pipe[2];

    if (BLOCK_N_TILES_WMMA > 0) {
      int b_col_start_in_tile_current = 0 * WMMA_N;
      wmma::load_matrix_sync(
          b_frag_inner_pipe[0],
          &Bsub_pipe[compute_pipe_idx][b_col_start_in_tile_current][0],
          WMMA_K + SKEW_HALF);
    }

    int current_inner_pipe_idx = 0;

    // #pragma unroll
    for (int n_tile = 0; n_tile < BLOCK_N_TILES_WMMA; ++n_tile) {
      int next_inner_pipe_idx = 1 - current_inner_pipe_idx;

      if (n_tile < BLOCK_N_TILES_WMMA - 1) {
        int b_col_start_in_tile_next = (n_tile + 1) * WMMA_N;
        wmma::load_matrix_sync(
            b_frag_inner_pipe[next_inner_pipe_idx],
            &Bsub_pipe[compute_pipe_idx][b_col_start_in_tile_next][0],
            WMMA_K + SKEW_HALF);
      }

      wmma::mma_sync(acc_frag[n_tile], a_frag,
                     b_frag_inner_pipe[current_inner_pipe_idx],
                     acc_frag[n_tile]);

      current_inner_pipe_idx = next_inner_pipe_idx;
    }
  }
  __syncthreads();

  // Store results from accumulator fragments to global memory
  // #pragma unroll
  for (int n_tile = 0; n_tile < BLOCK_N_TILES_WMMA; ++n_tile) {
    wmma::store_matrix_sync(&C_shmem_output_buffers[warp_id][0][0],
                            acc_frag[n_tile], WMMA_N, wmma::mem_row_major);
    // Establish warp-level HB after WMMA store before per-lane reads.

    for (int elem_idx_in_frag = lane_id; elem_idx_in_frag < WMMA_M * WMMA_N;
         elem_idx_in_frag += warpSize) {
      int r_frag = elem_idx_in_frag / WMMA_N;
      int c_frag = elem_idx_in_frag % WMMA_N;

      int oc_idx = block_row_gemm_start + (warp_id * WMMA_M) + r_frag;

      int offset_in_block_N_processing = (n_tile * WMMA_N) + c_frag;

      if (oc_idx < C_out && offset_in_block_N_processing < TILE_N_PER_BLOCK &&
          n_params_sh[offset_in_block_N_processing].isValidPixel) {
        const NDecomposed &current_n_params =
            n_params_sh[offset_in_block_N_processing];
        int ow_eff = current_n_params.ow_eff;
        int oh_eff = current_n_params.oh_eff;
        int n_batch_idx = current_n_params.n_batch_idx;

        float val = C_shmem_output_buffers[warp_id][r_frag][c_frag];
        val += bias_ptr[oc_idx];

        output_ptr[n_batch_idx * C_out * H_out * W_out +
                   oc_idx * H_out * W_out + oh_eff * W_out + ow_eff] = val;
      }
    }
    // Ensure all lanes finish consuming this tile before next store reuses it.
  }
}

#ifdef CONV2D_STANDALONE
int main() {
  const size_t input_bytes =
      static_cast<size_t>(N_batch) * C_in * H_in * W_in * sizeof(float);
  const size_t weight_bytes =
      static_cast<size_t>(C_out) * C_in * K_h * K_w * sizeof(float);
  const size_t bias_bytes = static_cast<size_t>(C_out) * sizeof(float);
  const size_t output_bytes =
      static_cast<size_t>(N_batch) * C_out * H_out * W_out * sizeof(float);

  float *d_input = nullptr;
  float *d_weight = nullptr;
  float *d_bias = nullptr;
  float *d_output = nullptr;

  CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
  CHECK_CUDA(cudaMalloc(&d_weight, weight_bytes));
  CHECK_CUDA(cudaMalloc(&d_bias, bias_bytes));
  CHECK_CUDA(cudaMalloc(&d_output, output_bytes));

  CHECK_CUDA(cudaMemset(d_input, 0, input_bytes));
  CHECK_CUDA(cudaMemset(d_weight, 0, weight_bytes));
  CHECK_CUDA(cudaMemset(d_bias, 0, bias_bytes));
  CHECK_CUDA(cudaMemset(d_output, 0, output_bytes));

  /* Grid configuration (all values are now constexpr) */
  constexpr int grid_x = (N_GEMM + TILE_N_PER_BLOCK - 1) / TILE_N_PER_BLOCK;
  constexpr int grid_y = (M_GEMM + TILE_M_PER_BLOCK - 1) / TILE_M_PER_BLOCK;

  dim3 grid(grid_x, grid_y);
  dim3 block(THREADS_PER_BLOCK);

  /*  <<<  simplified launch  >>>  */
  conv2d<<<grid, block>>>(d_input, d_weight, d_bias, d_output);

  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_bias));
  CHECK_CUDA(cudaFree(d_weight));
  CHECK_CUDA(cudaFree(d_input));
  return 0;
}
#endif // CONV2D_STANDALONE
