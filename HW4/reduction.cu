#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <iostream>


void reduction(double* sum_ptr, const double* a, long N){
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) sum += a[i];
  *sum_ptr = sum;
}

// Note: matrices are stored in row major order
void MMult0(long N, double *A, double *x, double *c) {
  #pragma omp parallel for
  for (long i = 0; i < N; i++) {
    for (long p = 0; p < N; p++) {
        double A_ip = A[i*N+p];
        double B_p = x[p];
        c[i]+= A_ip * B_p;
      }
    }
}


void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 1024

__global__ void reduction_kernel2(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void dot1(double* sum, const double* a, const double *b, long N){
  __shared__ double smem[BLOCK_SIZE];
  int tnum = threadIdx.x;
  if (tnum < N){ 
	  smem[threadIdx.x] = a[(blockIdx.x)*N+tnum]*b[tnum];
	  tnum+=BLOCK_SIZE;
	  while (tnum < N){ // add the rest of the row
		smem[threadIdx.x] += a[(blockIdx.x)*N+tnum]*b[tnum];
		tnum+=BLOCK_SIZE;
	  }	
  } else smem[threadIdx.x] = 0.0;
  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}


int main() {
  long N;
  std::cout << "N? : " << std::endl;
  std::cin >> N;
  double *x, *A, *c, *gpprod;
  cudaMallocHost((void**)&x, N * sizeof(double));
  cudaMallocHost((void**)&A, N*N*sizeof(double));
  cudaMallocHost((void**)&c, N * sizeof(double));
  cudaMallocHost((void**)&gpprod, N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++){
	 x[i] = drand48();
	 c[i] = 0;
  }
  for (long i = 0; i < N*N; i++){
	 A[i] = drand48();
  }

  double tt = omp_get_wtime();
  MMult0(N, A, x, c);
  printf("CPU Bandwidth = %f GB/s\n", (N*N+3.0*N)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *x_d, *A_d, *sum_d;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&A_d, N*N*sizeof(double));
  cudaMalloc(&sum_d, N*sizeof(double));

  cudaMemcpyAsync(A_d, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  tt = omp_get_wtime();


  long Nb = N; // each block will compute 1 entry
  dot1<<<Nb,BLOCK_SIZE>>>(sum_d, A_d, x_d, N);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMemcpyAsync(gpprod, sum_d, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", (N*N+3.0*N)*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  double error=0;
  for (int i=0; i<N; i++){
	error+=(gpprod[i]-c[i])*(gpprod[i]-c[i]);
  }
  printf("Total squared error = %f\n", error);
  cudaFree(x_d);
  cudaFree(A_d);
  cudaFree(sum_d);
  cudaFreeHost(x);
  cudaFreeHost(A);
  cudaFreeHost(c);
  cudaFreeHost(gpprod);
  return 0;
}
