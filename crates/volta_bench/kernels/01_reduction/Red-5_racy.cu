// red5.cu
#define BLOCKSIZE 8

__device__ __forceinline__ void warpReduce(volatile int *sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void reduce0(int *g_idata, int *g_odata) {
  extern __shared__ int sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = threadIdx.x;
  sdata[tid] = g_idata[i] + g_idata[i + BLOCKSIZE / 2];
  __syncthreads();

  for (unsigned int s = BLOCKSIZE / 4; s > 0; s >>= 1) {
    if (tid < s && tid + s < BLOCKSIZE) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    // cast to volatile so that the compiler doesn’t optimize
    // away the consecutive loads/stores:
    warpReduce((volatile int *)sdata, tid);
  }

  if (tid == 0) {
    // optional: write out the final sum of the block
    g_odata[0] = sdata[0];
  }
}
