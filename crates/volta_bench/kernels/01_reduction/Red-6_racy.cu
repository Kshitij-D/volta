#define BLOCKSIZE 8
template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
  if (blockSize >= 64)
    sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32)
    sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16)
    sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8)
    sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4)
    sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2)
    sdata[tid] += sdata[tid + 1];
}

//------------------------------------------------------------------------------
// reduce5 kernel, templated on blockSize
//------------------------------------------------------------------------------
template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata) {
  extern __shared__ int sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int idx = tid;
  int n = BLOCKSIZE;

  // load two elements per thread (if in range)
  int sum = 0;
  if (idx < n)
    sum = g_idata[idx];
  if (idx + blockSize < n)
    sum += g_idata[idx + blockSize];
  sdata[tid] = sum;
  __syncthreads();

  // tree‐based reduction in shared memory
  if (blockSize >= 512) {
    if (tid < 256)
      sdata[tid] += sdata[tid + 256];
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128)
      sdata[tid] += sdata[tid + 128];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64)
      sdata[tid] += sdata[tid + 64];
    __syncthreads();
  }

  // final warp‐level unrolled reduction
  if (tid < 32) {
    warpReduce<blockSize>(sdata, tid);
  }

  // write result for this block
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

template __global__ void reduce6<BLOCKSIZE>(int * /*g_idata*/, int * /*g_odata*/
);
