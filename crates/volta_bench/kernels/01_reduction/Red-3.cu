#define BLOCK_SIZE 128
__global__ void reduce3(int *g_idata, int *g_odata) {
  extern __shared__ int sdata[BLOCK_SIZE];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = tid;
  sdata[tid] = g_idata[i];
  __syncthreads();
  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0)
    g_odata[0] = sdata[0];
}
