//pass
//--gridDim=[32,32] --blockDim=[512,512]
// SOURCE: https://github.com/NVIDIA/Megatron-LM/commit/1ec6b0e941b4e324602d3fd6a955f28cd334a383

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SHFL(var, srcLane) __shfl_sync(__activemask(), var, srcLane)
// Compile-time constants to avoid param-driven control flow during analysis
#define N1_ROWS 1
#define N2_COLS 1
#define HAS_GAMMA 0
#define HAS_BETA 0
// Fix block geometry used in index math for concreteness
#define BLK_X 8
#define BLK_Y 1
#define __requires(x) /* assume x is true */

template<typename U> __device__
void cuWelfordOnlineSum(
  const U curr,
  U& mu,
  U& sigma2,
  U& count)
{
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template<typename U> __device__
void cuChanOnlineSum(
  const U muB,
  const U sigma2B,
  const U countB,
  U& mu,
  U& sigma2,
  U& count)
{
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA*mu + nB*muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template<typename T, typename U> __device__
void cuWelfordMuSigma2(
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const int i1,
  U& mu,
  U& sigma2,
  U* buf) 
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu= U(0);
  sigma2 = U(0);
  if (i1 < N1_ROWS) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = BLK_X * BLK_Y;
    const int thrx = threadIdx.x + threadIdx.y * BLK_X;
    const T* lvals = vals + i1*N2_COLS;
    int l = 4*thrx;
    for (;  l < N2_COLS - 3;  l+=4*numx) {
      for (int k = 0;  k < 4;  ++k) {
        U curr = static_cast<U>(lvals[l+k]);
        cuWelfordOnlineSum<U>(curr,mu,sigma2,count);
      }
    }
    for (;  l < N2_COLS;  ++l) {
      U curr = static_cast<U>(lvals[l]);
      cuWelfordOnlineSum<U>(curr,mu,sigma2,count);
    }
    // intra-warp reductions (safe for blockDim.x <= 32)
    // Use active mask and only combine with valid partner lanes.
    unsigned mask = __activemask();
    for (int l = 0; (1 << l) < blockDim.x && l < 5; ++l) {
      int srcLaneB = threadIdx.x + (1 << l);
      // All lanes participate in the shuffle with the same mask; values read
      // from an inactive source lane are undefined but ignored by the guard.
      U muB = __shfl_sync(mask, mu, srcLaneB);
      U countB = __shfl_sync(mask, count, srcLaneB);
      U sigma2B = __shfl_sync(mask, sigma2, srcLaneB);
      if (srcLaneB < blockDim.x) {
        cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U* ubuf = (U*)buf;
      U* ibuf = (U*)(ubuf + blockDim.y);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2*wrt_y] = mu;
          ubuf[2*wrt_y+1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U muB = ubuf[2*threadIdx.y];
          U sigma2B = ubuf[2*threadIdx.y+1];
          U countB = ibuf[threadIdx.y];
          cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1]/U(N2_COLS);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2/U(N2_COLS), 0);
    }
  }
}


template<typename T, typename U, typename V> __global__
void cuApplyLayerNorm(
  V* __restrict__ output_vals,
  U* __restrict__ mean,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma,
  const V* __restrict__ beta,
  int has_gamma,
  int has_beta
  ) 
{
  __shared__ U* buf;
  // 1D-layout broadcast via shared memory (post: correct sync)
  __shared__ U smu;
  __shared__ U sinv;
  // __requires(N2_COLS == 1024);
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1=blockIdx.y; i1 < N1_ROWS; i1 += gridDim.y) {
    U mu,sigma2;
    cuWelfordMuSigma2(vals,N1_ROWS,N2_COLS,i1,mu,sigma2,buf);
    const T* lvals = vals + i1*N2_COLS;
    V* ovals = output_vals + i1*N2_COLS;
    U c_invvar = rsqrt(sigma2 + epsilon);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      smu = mu;
      sinv = c_invvar;
    }
    __syncthreads();
    const int numx = BLK_X * BLK_Y;
    const int thrx = threadIdx.x + threadIdx.y * BLK_X;
    if (HAS_GAMMA && HAS_BETA) {
      for (int i = thrx;  i < N2_COLS;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = gamma[i] * static_cast<V>(sinv * (curr - smu)) + beta[i];
      }
    } else {
      for (int i = thrx;  i < N2_COLS;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = static_cast<V>(sinv * (curr - smu));
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean[i1] = mu;
      invvar[i1] = c_invvar;
    }
    __syncthreads(); // <- BUG
  }
}

// Explicit template instantiation to force code generation
template __global__ void cuApplyLayerNorm<float, float, float>(
  float* __restrict__ output_vals,
  float* __restrict__ mean,
  float* __restrict__ invvar,
  const float* __restrict__ vals,
  const int n1,
  const int n2,
  const float epsilon,
  const float* __restrict__ gamma,
  const float* __restrict__ beta,
  int has_gamma,
  int has_beta
);
