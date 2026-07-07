// same warp‐wide reduction helper as in Kernel A:
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

// templated 1-block reduction kernel
template <unsigned int blockSize>
__global__ void reduce1block(const int *g_idata, int *g_odata) {
  extern __shared__ int sdata[]; // size == blockSize
  unsigned int tid = threadIdx.x;

  // load into shared mem
  sdata[tid] = g_idata[tid];
  __syncthreads();

  // unroll the tree‐reduction, compile‐time branching on blockSize
  if (blockSize >= 512) {
    if (tid < 512)
      sdata[tid] += sdata[tid + 512];
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 256)
      sdata[tid] += sdata[tid + 256];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 128)
      sdata[tid] += sdata[tid + 128];
    __syncthreads();
  }
  if (blockSize >= 64) {
    if (tid < 64)
      sdata[tid] += sdata[tid + 64];
    __syncthreads();
  }

  // last warp does its own thing without __syncthreads()
  if (tid < 32)
    warpReduce<blockSize>(sdata, tid);

  // write result for this block
  if (tid == 0)
    g_odata[0] = sdata[0];
}

// and here is the concrete instantiation for blockSize = 1024:
template __global__ void reduce1block<8>(const int * /*g_idata*/,
                                         int * /*g_odata*/);
