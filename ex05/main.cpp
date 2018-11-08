#include <iostream>
#include <random>
#include <vector>
#include <array>
#include <cblas.h>
#include <omp.h>

void OwnDGEMV(const int m, const int n, const double alpha, const double* A,
              const double* x, const double beta, double* y)
{
    for (int i= 0; i < m; i++)
    {
        double Aix{0};
        for (int j= 0; j < n; j++)
        {
            Aix += A[i * n + j] * x[j]; 
        }
        y[i]= alpha * Aix + beta * y[i]; 
    }
}

// returns the Rayleigh quotient for real matrix A of size m x n and vector x of size n x 1
double RayleighQuotient(const int m, const int n, const double* A, const double* x)
{
        double* Ax= new double[m];
        double* xTAx= new double[1];

        OwnDGEMV(m, n, 1, A, x, 0, Ax);
        OwnDGEMV(1, n, 1, x, Ax, 0, xTAx);
        const double result{xTAx[0]}; 

        delete[] Ax;
        delete[] xTAx;
        return result;
}

// calculate and return L2 norm of matrix A of size m x n
double L2Norm(const int m, const int n, const double* const A)
{
    double sum{0.0};
    for (int i= 0; i < m; i++)
    {
        for (int j= 0; j < n; j++)
        {
            sum += A[i * m + j] * A[i * m + j];
        }
    }
    return std::sqrt(sum);
}

// init square matrix A of size m x m according to task description
void InitMatrix(const int m, double* A, const double alpha, const int seed)
{
    std::default_random_engine rd(seed);
    std::uniform_real_distribution<double> unif(0, 1);
    for (int i= 0; i < m; i++)
    {
        for (int j= 0; j < m; j++)
        {
            A[i * m + j]= unif(rd);
        }
    }
    for (int i= 0; i < m; i++)
    {   
        A[i * m + i]= (i + 1) * alpha;
        for (int j= i + 1; j < m; j++)
        {
            A[j * m + i]= A[i * m + j];    
        }
    }
}

// for debug only: Use only w/ small m, n
// prints A to screen
void PrintMatrix(const int m, const int n, const double* A)
{
    for (int i= 0; i < m; i++)
    {
        for (int j= 0; j < m; j++)
        {
            std::cout << A[i * m + j] << "\t\t";
        }
        std::cout << std::endl;
    }
}

int main()
{
    static const int m{500};
    const std::vector<double> alphas{1./8., 1./4., 1./2., 1., 3./2., 2., 4., 8., 16.};
    std::vector<double> lambdas{};
    const double eps{std::pow(10, -12)};

    for (auto alpha : alphas)
    {
        double* const A= new double[m * m];
        InitMatrix(m, A, alpha, 1);
        double* const q= new double[m]();
        q[0]= 1;

        double lambda_0{0};
        double lambda_1{1};
        double* const Aq= new double[m];
        double L2_norm{0};

        while (std::abs(lambda_1 - lambda_0) > eps)
        {
            lambda_0= lambda_1;
            OwnDGEMV(m, m, 1, A, q, 0, Aq);
            L2_norm= L2Norm(m, 1, Aq);
            for (int i= 0; i < m; i++)
            {
                q[i]= Aq[i] / L2_norm;
            }
            lambda_1= RayleighQuotient(m, m, A, q);
        }        
        lambdas.push_back(lambda_1);
        
        delete[] A; 
        delete[] q;
        delete[] Aq;
    }

    for (auto value : lambdas)
    {
        std::cout << value << std::endl;
    }

    return 0;
}