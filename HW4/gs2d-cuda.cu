// 2D Poisson solver using Gauss-Seidel methods, CUDA version
// Ondrej Maxian
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>


double computeResidual(long N, double *u, double *f){
 double hsq = 1.0/((N+1.0)*(N+1.0));
 double res=0;
 //Timer t1;
 //t1.tic();
 #pragma omp parallel for default(none) \
 shared(u,f,hsq,N) \
 reduction(+:res)
 for (long j = 1; j <= N; j++) {
  for (long i = 1; i <= N; i++) {
	double resi=(-u[(N+2)*j+i-1]-u[(N+2)*(j-1)+i]
		-u[(N+2)*j+i+1]-u[(N+2)*(j+1)+i]+4*u[(N+2)*j+i])/hsq-f[(N+2)*j+i];
	res+=resi*resi;
  }
 }
 //std::cout << "Time to calc residual: " << t1.toc() << std::endl;
 res=sqrt(res);
 return res;
}
  

void GaussSeidel( long N, double *u, double *f, int maxiters) {
  double *ukp1=(double*) malloc((N+2)*(N+2) * sizeof(double));
  double hsq = 1.0/((N+1.0)*(N+1.0));
  double res=0;
  double relres=1e6;
  int iter=0;
  // Compute the initial residual
  double res0=computeResidual(N,u,f);
  std::cout << "Initial residual: " << res0 << std::endl;
  while (relres > 1e-6 && iter < maxiters){
   // Do the iteration
   //Timer t1;
   //t1.tic();
   # pragma omp parallel for shared(u,hsq,N,f) 
   // Red points first
   for (long j = 1; j <= N; j++) {
    long iStart=j%2;
    if (iStart==0){iStart=2;}
    for (long i = iStart; i <= N; i+=2) {
	 u[(N+2)*j+i]=0.25*(hsq*f[(N+2)*j+i]+u[(N+2)*j+i-1]+u[(N+2)*(j-1)+i]
		+u[(N+2)*j+i+1]+u[(N+2)*(j+1)+i]);
         //std::cout << "Red point " << (N+2)*j+i << " " << u[(N+2)*j+i] << std::endl;
    }
   }
   # pragma omp parallel for shared(u,hsq,N,f)
   // Black points
   for (long j = 1; j <= N; j++) {
    for (long i = (j%2)+1; i <= N; i+=2) {
	 u[(N+2)*j+i]=0.25*(hsq*f[(N+2)*j+i]+u[(N+2)*j+i-1]+u[(N+2)*(j-1)+i]
		+u[(N+2)*j+i+1]+u[(N+2)*(j+1)+i]);
         //std::cout << "Black point " << (N+2)*j+i << " " << u[(N+2)*j+i] <<  std::endl;
    }
   }
   //std::cout << "Time to do iteration and copy: " << t1.toc() << std::endl;
   res=computeResidual(N,u,f);
   iter++;
   relres=res/res0;
   if(iter%100==0) printf("%10d %10f \n", iter, relres);
   }
  free(ukp1);
}

#define BLOCK_SIZE 1024

// rb is 0 for red points, 1 for black points
__global__ void gpGS(double* u, const double *f, long N, int rb){
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
  int j = idx/(N+2);
  int i = idx % (N+2);
  double hsq = 1.0/((N+1.0)*(N+1.0));
  if (i >= 1 && i <= N && j >= 1 && j <= N && ((i+j)%2)==rb){
  	u[(N+2)*j+i]=0.25*(hsq*f[(N+2)*j+i]+u[(N+2)*j+i-1]+u[(N+2)*(j-1)+i]
		+u[(N+2)*j+i+1]+u[(N+2)*(j+1)+i]);
  }
}

__global__ void resvec(double* resv, const double* u, const double *f, long N){
  __shared__ double smem[BLOCK_SIZE];
  double hsq = 1.0/((N+1.0)*(N+1.0));
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
  int j = idx/(N+2);
  int i = idx % (N+2);
  if (i >= 1 && i <= N && j >= 1 && j <= N){
	  double resi=(-u[(N+2)*j+i-1]-u[(N+2)*(j-1)+i]
		-u[(N+2)*j+i+1]-u[(N+2)*(j+1)+i]+4*u[(N+2)*j+i])/hsq-f[(N+2)*j+i];
	  smem[threadIdx.x]=resi*resi;
  } else{
	 smem[threadIdx.x]=0;
  }

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
    if (threadIdx.x == 0) resv[blockIdx.x] = smem[0] + smem[1];
  }
}

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


int main(int argc, char** argv) {
    int N;
    std::cout << "N? : " << std::endl;
    std::cin >> N;
    double *u, *f, *ugp;
    cudaMallocHost((void**)&u, (N+2)*(N+2) * sizeof(double));
    cudaMallocHost((void**)&f, (N+2)*(N+2) * sizeof(double));
    cudaMallocHost((void**)&ugp, (N+2)*(N+2) * sizeof(double));
    // Initialize u and f 
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (N+2)*(N+2); i++){ 
	u[i]=0.0;
	f[i]=1.0;
        ugp[i]=0.0;
    }
    int maxiters=1000;
    double tt = omp_get_wtime();
    // CPU Version
    GaussSeidel(N,u,f,maxiters);
    printf("CPU time = %f s\n", (omp_get_wtime()-tt));
    // GPU Version
    double *u_d, *up1_d, *f_d, *sum_d;
    cudaMalloc(&u_d, (N+2)*(N+2)*sizeof(double));
    cudaMalloc(&up1_d, (N+2)*(N+2)*sizeof(double));
    cudaMalloc(&f_d, (N+2)*(N+2)*sizeof(double));
    cudaMalloc(&sum_d, (N+2)*(N+2)*sizeof(double));
    cudaMemcpyAsync(u_d, ugp, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(up1_d, ugp, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(f_d, f, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(sum_d, ugp, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
    //printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();

    tt = omp_get_wtime();
    long Nb = ((N+2)*(N+2)+BLOCK_SIZE-1)/(BLOCK_SIZE);; // each block will compute 1 entry
    int gpiter=0;
    double gpres=1.0;
    double sum;
    while (gpres > 1e-6 && gpiter < maxiters){
	gpiter++;
    	gpGS<<<Nb,BLOCK_SIZE>>>(u_d, f_d, N, 0); // 0 means red points
    	cudaDeviceSynchronize();
	gpGS<<<Nb,BLOCK_SIZE>>>(u_d, f_d, N, 1); // 1 means black points
    	cudaDeviceSynchronize();
    	long Nbr = Nb;
        //Residual calculation
        resvec<<<Nbr,BLOCK_SIZE>>>(sum_d, u_d, f_d, N);
        while (Nbr > 1) {
	   long N1 = Nbr;
	   Nbr = (Nbr+BLOCK_SIZE-1)/(BLOCK_SIZE);
	   reduction_kernel2<<<Nbr,BLOCK_SIZE>>>(sum_d + Nbr, sum_d, N1);
	   sum_d += Nbr;
        }
        cudaMemcpyAsync(&sum, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
	gpres=sqrt(sum)/((double) N);
	if(gpiter%100==0) printf("GPU %10d %10f \n", gpiter, gpres);
    }    
    printf("GPU time = %f s \n", (omp_get_wtime()-tt));
    cudaMemcpyAsync(ugp, u_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost);
    double error=0.0;
    //printf("First entry GPU, CPU: %f %f \n", ugp[0], u[0]);
    for (int i=0; i<(N+2)*(N+2); i++){
	error+=(ugp[i]-u[i])*(ugp[i]-u[i]);
	if (i==0) printf("First entry error: %f \n", (ugp[i]-u[i]));
    }
    printf("Total squared error = %f\n", error);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaFree(u_d);
    cudaFree(up1_d);
    cudaFree(f_d);
    cudaFree(sum_d);
    cudaFreeHost(u);
    cudaFreeHost(ugp);
    cudaFreeHost(f);
    return 0;
}
