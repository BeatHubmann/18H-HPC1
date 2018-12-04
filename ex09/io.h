#pragma once

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Prints std::vector as space-separated list.
template <class T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& v) {
  std::string p = "";
  for (auto a : v) {
    o << p << a;
    p = " ";
  }
  return o;
}

// Prints matrix to stream.
void Print0(const Matr& a, std::ostream& o) {
  for (Size i = 0; i < L; ++i) {
    o << i << ":";
    for (Size k = a.ki[i]; k < a.ki[i + 1]; ++k) {
      o << " " << a.a[k] << "[" << a.gjk[k] << "]";
    }
    o << "\n";
  }
  o.flush();
}

// Prints matrix to stream, with synchronization.
void Print(const Matr& a, std::ostream& o, MPI_Comm comm) {
  int r, nr;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &nr);

  for (int i = 0; i < nr; ++i) {
    MPI_Barrier(comm);
    if (i == r) {
      o << "rank=" << r << std::endl;
      Print0(a, o);
    }
  }
}

// Prints vector to stream.
void Print0(const Vect& u, int r, std::ostream& o) {
  for (Size i = 0; i < L; ++i) {
    auto gi = LocToGlb(i, r);
    auto xy = GlbToCoord(gi);
    o << Real(xy[0] + 0.5) / NX << " " << Real(xy[1] + 0.5) / NX 
        << " " << u.v[i] << "\n";
  }
  o.flush();
}

// Prints vector to stream, with synchronization.
void Print(const Vect& u, MPI_Comm comm, std::ostream& o) {
  int r, nr;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &nr);

  for (int i = 0; i < nr; ++i) {
    MPI_Barrier(comm);
    if (i == r) {
      Print0(u, r, o);
    }
  }
}

// Writes vector to file.
// fn: filename
void Write(const Vect& u, MPI_Comm comm, std::string fn) {
  int r, nr;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &nr);

  if (r == 0) {
    std::ofstream f(fn, std::ofstream::trunc);
  }
  for (int i = 0; i < nr; ++i) {
    std::ofstream f(fn, std::ofstream::app);
    MPI_Barrier(comm);
    if (i == r) {
      Print0(u, r, f);
    }
  }
}

// Writes vector to file.
// fn: filename
// This exactly does what's asked: Writes u to file using MPI I/O
// It's binary format without prettification though, but task said nothing about that
void WriteMpi(const Vect& u, MPI_Comm comm, std::string fn)
{
  // TODO 2b
  int rank;
  MPI_Comm_rank(comm, &rank);

  MPI_File file;
  MPI_Offset offset= sizeof(double) * L * rank;
  
  double* buffer= new double[L];
  for (Size i= 0; i < L; i++)
    buffer[i]= (double)u.v[i];

  MPI_File_open(comm, fn.c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY,
                MPI_INFO_NULL, &file);
  
  MPI_File_seek(file, offset, MPI_SEEK_SET);  
  MPI_File_write(file, buffer, L, MR, MPI_STATUS_IGNORE);
  
  MPI_File_close(&file);
}


