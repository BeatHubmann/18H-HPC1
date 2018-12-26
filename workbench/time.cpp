#include <cstdio>
#include <chrono>

std::chrono::duration<double> tsum{0};
int counter{0};

int main()
{
		for (int i= 0; i < 2 << 24; i++)
		{
			const auto t0= std::chrono::steady_clock::now();
			counter+= i;
			const auto t1= std::chrono::steady_clock::now();
			tsum+= t1 - t0;
		}
		std::printf("Total time: %f", tsum.count());
		return 0;
}
