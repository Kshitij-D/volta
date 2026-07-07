#define BLOCK_SIZE 128

__global__ void reduce1024_1block(const int *g_idata, int *g_odata) {
  __shared__ int sdata[BLOCK_SIZE];

  unsigned tid = threadIdx.x;
  unsigned i = tid; // blockIdx.x==0, BLOCK_SIZE never used here

  sdata[tid] = g_idata[i];
  __syncthreads();

  // #pragma unroll
  for (unsigned s = 1; s < blockDim.x; s <<= 1) {
    // if ((tid & (2*s - 1)) == 0)
    if (tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];
    }

    __syncthreads();
    if (tid == 0)
      g_odata[0] = sdata[0];
  }
}
