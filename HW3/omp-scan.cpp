#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  
  if (n == 0) return;
  prefix_sum[0] = 0;
  int nthreads;
  # pragma omp parallel 
  {
    nthreads = omp_get_num_threads();
  }
  std:: cout << nthreads << std::endl;
  int increment = n/nthreads;
  # pragma omp parallel
  {
	  int tid = omp_get_thread_num();
	  int startindex=increment*tid+1;
	  int endindex=std::min((tid+1)*increment+1,(int)n);
	  //std::cout << "Thread " << tid << " starting at " << startindex << " and ending at " << endindex << std::endl;
	  for (long i = startindex; i < endindex; i++) {
		prefix_sum[i] = prefix_sum[i-1] + A[i-1];
	  }
  }
  // Loop over the threads in serial
  for (int tid=1; tid < nthreads; tid++){
		int start=increment*tid+1;
		int end=std::min((tid+1)*increment+1,(int)n);
		for (long i = start; i < end; i++){
			prefix_sum[i]+=prefix_sum[start-1];
		}
	}
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
