#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double time_pingpong(long Nrepeat, long Nsize, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size ); 

  int* msg = (int*) malloc(Nsize*sizeof(int));
  for (long i = 0; i < Nsize; i++) msg[i] = 0;
  if (!rank) printf("Size of array: %f MB\n", Nsize*sizeof(int)/1e6);

  MPI_Barrier(comm);
  double tt = MPI_Wtime();
  int proc0, proc1;
  for (long repeat  = 0; repeat < Nrepeat; repeat++) {
    proc0=repeat % size;
    proc1=(repeat+1) % size;
    MPI_Status status;

      if (rank == proc0){
	msg[Nsize-1]+=proc0;
        MPI_Send(msg, Nsize*sizeof(int), MPI_CHAR, proc1, repeat, comm);
      } else if (rank == proc1){
        MPI_Recv(msg, Nsize*sizeof(int), MPI_CHAR, proc0, repeat, comm, &status);
	//printf("Process %d recving\n", proc1);
      }
  }
  MPI_Barrier(comm);
  tt = MPI_Wtime() - tt;
  double expected=Nrepeat*(((double)size-1.0)*0.5);
  if (rank==0){
  	printf("Final value of number %d (should be %d)\n", msg[Nsize-1], (int)expected);
  }
  free(msg);
  return tt;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size ); 

  long Nrepeat = 1000*size;
  // This one does just 1 integer
  double tt = time_pingpong(Nrepeat, 1, comm);
  if (!rank) printf("Memory latency: %e ms\n", tt/Nrepeat * 1000);

  // Large array
  Nrepeat = 1000*size;
  long Nsize = 500000;
  tt = time_pingpong(Nrepeat, Nsize, comm);
  if (!rank) printf("Memory bandwidth: %e GB/s\n", (Nsize*Nrepeat*sizeof(int))/tt/1e9);

  MPI_Finalize();
}

