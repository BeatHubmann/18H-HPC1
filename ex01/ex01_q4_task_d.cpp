/* Filename: ex01_q4_task_d.cpp
   Course: HPCSE I, HS2018, ETHZ
   Exercise No: 01
   Author: bhubmann@student.ethz.ch
   INPUT: None
   OUTPUT: None
   DESCRIPTION: Runs 1D diffusion FDM scheme as given by exercise,
	            1000 iterations for naive FLOP/s performance calculation
*/

#include <iostream>
#include <cmath>
#include <cstring>
#include <omp.h>

void Diffusion1D(const double* const u_old, double* const u_new, const int N, const double factor)
{	
	u_new[0]=     u_old[0]   + factor * (u_old[N-1] - 2 * u_old[0]   + u_old[1]);	
	for (int i= 0; i < N - 1; i++)
	{
	    u_new[i]= u_old[i]   + factor * (u_old[i-1] - 2 * u_old[i]   + u_old[i+1]);
	}
	u_new[N-1]=   u_old[N-1] + factor * (u_old[N-2] - 2 * u_old[N-1] + u_old[0]);	
}

int main()
{
	const int kNumIterations= 1000;
	const int kBufferSize = 128 * 1024 * 1024; // 128 M
	const int kGridPointsN= 1024 * 1024; // 1 M
	const double kDomainLengthL= 1000.0;
	const double kAlpha= 1.0e-4;
	const double kCFL= 0.5;
	const double kDeltaX= kDomainLengthL / (kGridPointsN - 1);
	const double kDeltaT= kCFL * kDeltaX * kDeltaX / kAlpha;
	const double kFactor= kDeltaT * kAlpha / (kDeltaX * kDeltaX);

	char* filler= new char[kBufferSize]; // 16 MB
	
	double* u_0= new double[kGridPointsN]; // 8 MB
	double* u_1= new double[kGridPointsN]; // 8 MB

	auto lInitData = [kDomainLengthL](const double x_value)
	 {return std::sin(2. * M_PI / kDomainLengthL * x_value);};
	
	for (int i= 0; i < kGridPointsN; i++)
	{
		u_0[i]= lInitData(i * kDeltaX);
		u_1[i]= u_0[i];
	}

	double total_time= 0.0;

	for (int i= 0; i < kNumIterations; i++)
	{
		volatile int j= i * i;
		memset(filler, j, kBufferSize);
		const double kStartTime= omp_get_wtime();
		Diffusion1D(u_0, u_1, kGridPointsN, kFactor);
		const double kEndTime= omp_get_wtime();
		total_time += (kEndTime - kStartTime);
		double* u_temp= u_0;
		u_0= u_1;
		u_1= u_temp;
	}
	const long int total_flops= 5 * (long int)kNumIterations * (long int)kGridPointsN;
	const double performance= total_flops / (total_time * 1e9);
	std::cout << kNumIterations << " iterations, total time [s]: " << total_time << std::endl
	          << "total naive FLOP operations: " << total_flops << std::endl
	          << "measured performance [GFLOP/s]: " << performance << std::endl;
	delete[] filler;
	delete[] u_0;
	delete[] u_1;
	return 0;
}
