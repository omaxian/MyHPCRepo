/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
//int nthreads, i tid;
//float total;
// Problem was that variables that should be private were
// declared outside of the parallel region. Now all of the
// variables are declared inside the parallel region. 

/*** Spawn parallel region ***/
#pragma omp parallel
  {
  /* Obtain thread number */
  int tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    int nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  float total = 0.0;
  //#pragma omp for schedule(dynamic,100)
  // This loop is already being parallelized (done on each thread). 
  // No need for the extra parallel for statement. 
  for (int i=0; i<1000000; i++) //i is private now. 
     total = total + i*1.0;
  // Now you will get the same result for every thread no matter how many threads
  // you run. 

  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}
