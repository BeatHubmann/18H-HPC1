#include <cassert>
#include <map>
#include <mpi.h>
#include <iostream>

#include "def.h"
#include "index.h"
#include "op.h"
#include "io.h"

// Adds vectors.
Vect Add(const Vect& a, const Vect& b) {
  auto r = a;
  for (Size i = 0; i < a.v.size(); ++i) {
    r.v[i] += b.v[i];
  }
  return r;
}

// Multiplies vector and scalar.
Vect Mul(const Vect& a, Real k) {
  auto r = a;
  for (auto& e : r.v) {
    e *= k;
  }
  return r;
}

// Multiplies matrix and vector.
Vect Mul(const Matr& A, const Vect& u, MPI_Comm comm)
{
  // TODO 2a
  int rank, n_procs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_procs);

  std::vector<VI> idx_wish_list;
  std::vector<VI> idx_send_list;
  idx_wish_list.resize(n_procs);
  idx_send_list.resize(n_procs);

  int* count_messages= new int[n_procs]{0};
  int* total_messages= new int[n_procs]{0};

  Vect Au;
  Au.v.resize(L);

  for (Size i= 0; i < L; i++) // traverse rows
  {
    for (Size k= A.ki[i]; k < A.ki[i + 1]; k++)
    {
      const Size global_col_of_k{A.gjk[k]};
      const int rank_of_k{(int)GlbToRank(global_col_of_k)};
      if (rank_of_k != rank)
      {
        idx_wish_list[rank_of_k].push_back(global_col_of_k);
        std::cout << rank << ". rank: wish list add " << global_col_of_k << " from rank " << rank_of_k << std::endl;
        count_messages[rank_of_k]= 1; // we have to write to rank_of_k
      }
      else
      {
        Au.v[i] += A.a[k] * u.v[GlbToLoc(global_col_of_k)];
      }
    }
  }

  int outgoing{0};
  for (int i= 0; i < n_procs; i++)
    if (count_messages[i] == 1) outgoing++;

  MPI_Allreduce(&count_messages[0], &total_messages[0], n_procs,
                MPI_INT, MPI_SUM, comm);

  const int incoming{total_messages[rank]};  
  
  MPI_Request out_request[outgoing];
  MPI_Status out_status[outgoing];


  const int tag{rank};
  int idx{0};

  for (int i= 0; i < n_procs; i++) // send requests
  {
    if (count_messages[i] == 1)
    {
      Size send_buffer[idx_wish_list[i].size()];
      int count{0};
      for (auto c : idx_wish_list[i])
        send_buffer[count++]= c;
      MPI_Isend(&send_buffer[0], count, MS, i, tag, comm, &out_request[idx++]);
    }
  }

  for (int i= 0; i < incoming; i++) // receive requests
  {
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
    int source= status.MPI_SOURCE;
    int count{0};
    MPI_Get_count(&status, MS, &count);
    Size recv_buffer[count];
    MPI_Recv(&recv_buffer[0], count, MS, source,
              MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
    for (int c= 0; c < count; c++)
      std::cout << rank << ". rank: request[" << c << "] for  " <<recv_buffer[c] << " from rank "<< source << std::endl;
    idx_send_list[source]= VI(recv_buffer, recv_buffer + count);
  }
  MPI_Waitall(outgoing, out_request, out_status);


  MPI_Request ans_request[incoming];
  MPI_Status ans_status[incoming];
  idx= 0;

  for (int i= 0; i < n_procs; i++) // send replies
  {
    if (idx_send_list[i].size() > 0)
    {
      Size send_buffer[idx_send_list[i].size()];
      int count{0};
      for (auto c : idx_send_list[i])
      {
        std::cout << rank << ". rank has to send " << c << " == (local) " << GlbToLoc(c) << " = " << u.v[GlbToLoc(c)] << " to " << i << std::endl;
        send_buffer[count++]= u.v[GlbToLoc(c)];
      }
      MPI_Isend(&send_buffer[0], count, MR, i, tag, comm, &ans_request[idx++]);
    }
  }

  VR u_recv(L * L); // to save the parts of u obtained from other ranks

  for (int i= 0; i < n_procs; i++) // receive replies
  {
    if (count_messages[i] == 1)
    {
      std::cout << rank << ". rank is waiting for reply from " << i << std::endl;
      const int count{(int)idx_wish_list[i].size()};
      Size recv_buffer[count];
      MPI_Recv(&recv_buffer, count, MR, i, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
      for (int c= 0; c < count; c++)
      {
         u_recv[idx_wish_list[i][c]]= recv_buffer[c];
         std::cout << rank << ". rank has received from " << i << " : " << idx_wish_list[i][c] << " = " << recv_buffer[c] << std::endl;
      }
    }
  }

  MPI_Waitall(incoming, ans_request, ans_status);

  for (Size i= 0; i < L; i++) // traverse rows again and use u_recv 
  {
    for (Size k= A.ki[i]; k < A.ki[i + 1]; k++)
    {
      const Size global_col_of_k{A.gjk[k]};
      const int rank_of_k{(int)GlbToRank(global_col_of_k)};
      if (rank_of_k != rank)
        Au.v[i] += A.a[k] * u_recv[global_col_of_k];
    }
  }

  delete[] count_messages;
  delete[] total_messages;
  return Au;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, n_procs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_procs);

  // Laplacian
  Matr A = GetLapl(rank);

  // Initial vector
  Vect u;
  for (Size i = 0; i < L; ++i) {
    Size gi = LocToGlb(i, rank);
    auto xy = GlbToCoord(gi);
    Real x = Real(xy[0]) / NX;
    Real y = Real(xy[1]) / NY;
    Real dx = x - 0.5;
    Real dy = y - 0.5;
    Real r = 0.2;
    u.v.push_back(dx*dx + dy*dy < r*r ? 1. : 0.);
  }

  Write(u, comm, "u0");

  const Size nt = 10; // number of time steps
  for (Size t = 0; t < nt; ++t) {
    Vect du = Mul(A, u, comm);
    Real k = 0.25; // scaling, k <= 0.25 required for stability.
    du = Mul(du, k);
    u = Add(u, du);
  }

  Write(u, comm, "u1");
  WriteMpi(u, comm, "u1_MPI");

  MPI_Finalize();
}
