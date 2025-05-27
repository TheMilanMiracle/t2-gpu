#include <chrono>
#include <iostream>
#include <unistd.h>

#include "conway.h"


int aliveNeighbours(bool* &w, size_t x, size_t y)
{
	auto idx = [&](const size_t i, const size_t j){return w[((i + N) % N) + (((j + M) % M) * N)];};

	return  idx(x-1, y-1) + idx(x, y-1) + idx(x+1, y-1) +
		idx(x-1, y)  + idx(x+1, y) +
		idx(x-1, y+1) + idx(x, y+1) + idx(x+1, y+1);
}


int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		std::cout << "usage: ./seq {N} {M} {steps}" << std::endl;
		exit(1);
	}

	N = atoi(argv[1]);
	M = atoi(argv[2]);

	bool* w = new bool[N * M];
	init(w);

	if (N <= 50 and M <= 100)
		std::cout << w << std::endl;

	const int maxStep = atoi(argv[3]);

	int step = 0;
	long totalTimeNS = 0.;
	long totalTimeMS = 0.;
	while (step++ < maxStep)
	{
		bool* nextStep = new bool[N * M];

		auto start = std::chrono::high_resolution_clock::now();
		{
			for (int i = 0; i < N; i++)
				for(int j = 0; j < M; j++)
				{
					int neighbours = aliveNeighbours(w, i, j);
					nextStep[i + j * N] = neighbours == 3 or (neighbours == 2 and w[i + j * N]);
				}
		}
		auto end = std::chrono::high_resolution_clock::now();

		w = nextStep;
		
		std::cout	<< "step "
					<< step
					<< " computed in "
					<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
					<< "ms ("
					<< std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
					<< ") ns"
					<< std::endl;

		totalTimeNS += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		totalTimeMS += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		if (N <= 50 and M <= 100)
		{
			std::cout << "\033[2J\033[H" << w << std::endl;
			usleep(500000);
		}
	}

	std::cout	<< "avg computing time: "
				<< totalTimeMS / static_cast<float>(maxStep)
				<< " ms ("
				<< totalTimeNS / static_cast<float>(maxStep)
				<< " ns)";

	delete[] w;

	return 0;
}
