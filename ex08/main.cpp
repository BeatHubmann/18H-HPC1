#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <mpi.h>

inline long exact(const long N){
    // TODO b): Implement the analytical solution.
    return (N * N + N) / 2;
}

void reduce_mpi(const int rank, long& sum){
    // TODO e): Perform the reduction using blocking collectives.
    // This is the naive implementation   
    int tag{0}, size{0};
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
    {
        long recv_sum;
        for (int i= 1; i < size; i++)
        {
            MPI_Recv(&recv_sum, 1, MPI_LONG, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += recv_sum;
        }
    }
    else
    {
        MPI_Send(&sum, 1, MPI_LONG, 0, tag, MPI_COMM_WORLD);
    }
}

// PRE: size is a power of 2 for simplicity
void reduce_manual(int rank, int size, long& sum){
    // TODO f): Implement a tree based reduction using blocking point-to-point communication.
    // This is the tree-based implementation
    int tag{0}, tree_lvls{(int)std::log2(size)};
    long recv_sum{0};

    for (int i= tree_lvls; i > 0; i--)
        for (int j= 0; j < 1 << (i - 1); j++)
        {
            const int recvr{j};
            const int sendr{j + (1 << (i - 1))};
            if (rank == recvr)
            {
                MPI_Recv(&recv_sum, 1, MPI_LONG, sendr, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                sum += recv_sum;
            }
            else if (rank == sendr)
            {
                MPI_Send(&sum, 1, MPI_LONG, recvr, tag, MPI_COMM_WORLD);    
            }
        }    

}


int main(int argc, char** argv){
    const long N = 10000000;
    
    // Initialize MPI
    int rank, size;
    // TODO c): Initialize MPI and obtain the rank and the number of processes (size)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // -------------------------
    // Perform the local sum:
    // -------------------------
    long sum = 0;
    
    // Determine work load per rank
    long N_per_rank = N / size;
    
    // TODO d): Determine the range of the subsum that should be calculated by this rank.
    long N_start{N_per_rank * rank + 1};
    long N_end{N_start + N_per_rank - 1};
    if (N - N_end < N_per_rank)
        N_end= N;

    // N_start + (N_start+1) + ... + (N_start+N_per_rank-1)
    for(long i = N_start; i <= N_end; ++i){
        sum += i;
    }
    
    // -------------------------
    // Reduction
    // -------------------------
    // reduce_mpi(rank, sum);
    reduce_manual(rank, size, sum);
    
    // -------------------------
    // Print the result
    // -------------------------
    if(rank == 0){
        std::cout << std::left << std::setw(25) << "Final result (exact): " << exact(N) << std::endl;
        std::cout << std::left << std::setw(25) << "Final result (MPI): " << sum << std::endl;
    }
    // Finalize MPI
    // TODO c): Finalize MPI
    MPI_Finalize();
    return 0;
}
