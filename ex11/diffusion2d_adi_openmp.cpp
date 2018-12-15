#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include "timer.hpp"
#include <iomanip>
#include <omp.h>



struct Diagnostics
{
    double time;
    double heat;

    Diagnostics(double time, double heat) : time(time), heat(heat) {}
};

class Diffusion2D
{
public:
    Diffusion2D(const double D,
                const double L,
                const int N,
                const double dt,
                const int rank)
            : D_(D), L_(L), N_(N), dt_(dt), rank_(rank)
    {
        // Real space grid spacing.
        dr_ = L_ / (N_ - 1);

        // Actual dimension of a row (+2 for the ghost cells).
        real_N_ = N + 2;

        // Total number of cells.
        Ntot_ = (N_ + 2) * (N_ + 2);

        rho_.resize(Ntot_, 0.);
        rho_tmp_.resize(Ntot_, 0.);

        // Initialize field on grid
        initialize_rho();

        R_ = 2.*dr_*dr_ / dt_;
        // TODO:
        // Initialize diagonals of the coefficient
        // matrix A, where Ax=b is the corresponding
        // system to be solved
        // ...
        alpha= D_ * dt_ / 2 / dr_ / dr_;
        a_= std::vector<double>(Ntot_ - 1,       - alpha);
        b_= std::vector<double>(Ntot_    , 1 + 2 * alpha);
        c_= std::vector<double>(Ntot_ - 1,       - alpha);
    }


    void ThomasAlgorithm(const std::vector<double>& a,
                         const std::vector<double>& b,
                         const std::vector<double>& c,
                         const std::vector<double>& d,
                         std::vector<double>& x)
    {   
        std::vector<double> c_star(Ntot_, 0.0);
        std::vector<double> d_star(Ntot_, 0.0);
        // first row
        c_star[0]= c[0] / b[0];
        d_star[0]= d[0] / b[0];

        // rest of rows: forward sweep
        for (int i= 1; i < Ntot_; i++)
        {
            const double div_fact= 1.0 / (b[i] - a[i] * c_star[i]);
            c_star[i]= c[i] * div_fact;
            d_star[i]= (d[i] - a[i] * d_star[i-1]) * div_fact;
        }

        // reverse sweep
        for (int i= Ntot_ - 1; i > 0; i--)
            x[i]= d_star[i] - c_star[i] * d[i + 1];
    }



    void advance()
    {
        // TODO:
        // Implement the ADI scheme for diffusion
        // and parallelize with OpenMP
        // ...
#pragma omp parallel for collapse(2)
        for (int j= 1; j <= N_; j++)
            for (int i= 1; i <= N_; i++)
                rho_tmp_[j * (N_ + 2) + i]= alpha           * rho_[(j - 1) * (N_ + 2) + i] 
                                          + (1 - 2 * alpha) * rho_[ j      * (N_ + 2) + i]
                                          + alpha           * rho_[(j + 1) * (N_ + 2) + i];
        

        // ADI Step 1: Update rows at half timestep
        // Solve implicit system with Thomas algorithm
        ThomasAlgorithm(a_, b_, c_, rho_tmp_, rho_);

        // ADI: Step 2: Update columns at full timestep
        // Solve implicit system with Thomas algorithm
#pragma omp parallel for collapse(2)
        for (int i= 1; i <= N_; i++)
            for (int j= 1; j <= N_ ; j++)
                rho_tmp_[j * (N_ + 2) + i]= alpha           * rho_[j * (N_ + 2) + i - 1] 
                                          + (1 - 2 * alpha) * rho_[j * (N_ + 2) + i    ]
                                          + alpha           * rho_[j * (N_ + 2) + i + 1];
        
        ThomasAlgorithm(a_, b_, c_, rho_tmp_, rho_);
    }

    void write_rho_to_file(const std::string& filename)
    {
        std::ofstream file(filename.c_str());
        // Set highest possible precision, this way we are sure we are
        file << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

        // Loop over vector and write output to file
        for(int i = 0; i < (int)rho_.size(); ++i)
            file << rho_[i] << " ";

        file << std::endl;
        // File closes automatically at end of scope!
    }


    void compute_diagnostics(const double t)
    {
        double heat = 0.0;
        for (int i = 1; i <= N_; ++i)
            for (int j = 1; j <= N_; ++j)
                heat += dr_ * dr_ * rho_[i * real_N_ + j];

#if DEBUG
        std::cout << "t = " << t << " heat = " << heat << '\n';
#endif
        diag_.push_back(Diagnostics(t, heat));
    }


    void write_diagnostics(const std::string &filename) const
    {
        std::ofstream out_file(filename, std::ios::out);
        for (const Diagnostics &d : diag_)
            out_file << d.time << '\t' << d.heat << '\n';
        out_file.close();
    }


private:

    void initialize_rho()
    {
        /* Initialize rho(x, y, t=0) */

        const double bound{0.5}; // for initial conditions

        // TODO:
        // Initialize field rho based on the
        // prescribed initial conditions
        // and parallelize with OpenMP

        auto InBox= [bound, dr= dr_, L= L_](int i, int j){return (std::abs((i - 1) * dr - L / 2) < bound)
                                                              && (std::abs((j - 1) * dr - L / 2) < bound);};

#pragma omp parallel for collapse(2)
        for (int j= 1; j <= N_; j++) 
            for (int i= 1; i <= N_; i++)
            {
                if (InBox(i, j))
                    rho_[j * (N_ + 2) + i]= 1.0;
                else
                    rho_[j * (N_ + 2) + i]= 0.0;
            }
    }

    double alpha;
    double D_, L_;
    int N_, Ntot_, real_N_;
    double dr_, dt_;
    double R_;
    int rank_;
    std::vector<double> rho_, rho_tmp_;
    std::vector<Diagnostics> diag_;
    std::vector<double> a_, b_, c_;
};



int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " D L N dt\n";
        return 1;
    }

#pragma omp parallel
    {
#pragma omp master
        std::cout << "Running with " << omp_get_num_threads() << " threads\n";
    }

    const double D = std::stod(argv[1]);  //diffusion constant
    const double L = std::stod(argv[2]);  //domain side size
    const int N = std::stoul(argv[3]);    //number of grid points per dimension
    const double dt = std::stod(argv[4]); //timestep

    Diffusion2D system(D, L, N, dt, 0);

    timer t;
    t.start();
    for (int step = 0; step < 10000; ++step)
    {
        system.advance();
#ifndef _PERF_
        system.compute_diagnostics(dt * step);
        if (step % 1000 == 0)
            system.write_rho_to_file("density" + std::to_string(step) + ".dat");

#endif
    }
    t.stop();

    std::cout << "Timing: " << N << ' ' << t.get_timing() << '\n';

#ifndef _PERF_
    system.write_diagnostics("diagnostics_openmp.dat");
#endif
    return 0;
}
