#define BLOCK_SIZE 128
__global__ void reduce2(int *g_idata, int *g_odata) {
  __shared__ int sdata[BLOCK_SIZE];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();
  // do reduction in shared mem
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index + s < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0)
    g_odata[0] = sdata[0];
}
