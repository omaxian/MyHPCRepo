// 2D Poisson solver using Jacobi methods
// Ondrej Maxian
#include <stdio.h>
#include "utils.h"
#include <math.h>
#include <omp.h>


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
  

void Jacobi( long N, double *u, double *f, int maxiters) {
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
   # pragma omp parallel shared(ukp1,u,hsq,N,f) 
   { // start parallel region
   # pragma omp for
   for (long j = 1; j <= N; j++) {
    for (long i = 1; i <= N; i++) {
	 ukp1[(N+2)*j+i]=0.25*(hsq*f[(N+2)*j+i]+u[(N+2)*j+i-1]+u[(N+2)*(j-1)+i]
		+u[(N+2)*j+i+1]+u[(N+2)*(j+1)+i]);
    }
   }
   # pragma omp for
   for (long j = 1; j <= N; j++) {
    for (long i = 1; i <= N; i++) {
	 u[(N+2)*j+i]=ukp1[(N+2)*j+i];
    }
   }
   } // end parallel region
   //std::cout << "Time to do iteration and copy: " << t1.toc() << std::endl;
   res=computeResidual(N,u,f);
   iter++;
   relres=res/res0;
   printf("%10d %10f \n", iter, relres);
   }
  free(ukp1);
}

int main(int argc, char** argv) {
    int N;
    std::cout << "N? : " << std::endl;
    std::cin >> N;
    double* u = (double*) malloc((N+2)*(N+2) * sizeof(double)); // m x k
    double* f = (double*) malloc((N+2)*(N+2) * sizeof(double)); // k x n
    for (int i = 0; i < (N+2)*(N+2); i++){ 
	u[i]=0.0;
	f[i]=1.0;
    }
    // Initialize f
    int maxiters=1000;
    Timer t;
    t.tic();
    // Here switch between which method
    Jacobi(N,u,f,maxiters);
    double time = t.toc();
    std::cout << "Time : " << time << std::endl;	
    free(u);
    free(f);

  return 0;
}
