/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

// This code did not compile initially despite the description. 
// I got the compile error: "sum" is private in outer context. 
// To get the code to compile, I got rid of the routine dotprod
// and wrote it into the main function using one parallel for 
// loop with a + reduction. This gives the same (correct) answer
// every time. 

float a[VECLEN], b[VECLEN];

/*float dotprod ()
{
int i,tid;
float sum;

tid = omp_get_thread_num();
#pragma omp for reduction(+:sum)
  for (i=0; i < VECLEN; i++)
    {
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
    }
}*/


int main (int argc, char *argv[]) {
int i;
float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

//#pragma omp parallel
  //dotprod();
  //int tid = omp_get_thread_num();
#pragma omp parallel for default(none) \
                         shared(a,b) \
                         reduction(+:sum)
  for (i=0; i < VECLEN; i++)
    {
    sum = sum + (a[i]*b[i]);
   // printf("  tid= %d i=%d\n",tid,i);
    }

printf("Sum = %f\n",sum);

}

