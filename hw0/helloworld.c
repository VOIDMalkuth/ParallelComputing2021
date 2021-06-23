#include <stdio.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int myId, numProcs;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    printf("Hello world from process %d / %d\n", myId, numProcs);

    MPI_Finalize();
    return 0;
}