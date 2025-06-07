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

	auto t0 = std::chrono::high_resolution_clock::now();
	bool* w = new bool[N * M];
	init(w);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::cout	<< "Initial data set in "
				<< std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
				<< " microseconds"
				<< std::endl;


	const int maxStep = atoi(argv[3]);

	int step = 0;
	auto start = std::chrono::high_resolution_clock::now();
	while (step++ < maxStep)
	{
		bool* nextStep = new bool[N * M];

		t0 = std::chrono::high_resolution_clock::now();
		{
			for (int i = 0; i < N; i++)
				for(int j = 0; j < M; j++)
				{
					int neighbours = aliveNeighbours(w, i, j);
					nextStep[i + j * N] = neighbours == 3 or (neighbours == 2 and w[i + j * N]);
				}
		}
		t1 = std::chrono::high_resolution_clock::now();

		w = nextStep;
		
		std::cout	<< "step "
					<< step
					<< " computed in "
					<< std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
					<< " microseconds"
					<< std::endl;

	}

	auto end = std::chrono::high_resolution_clock::now();

	std::cout	<< "Total time: "
				<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
				<< " microseconds"
				<< std::endl;

	std::cout	<< "avg computing time: "
			<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / static_cast<float>(maxStep)
			<< " microseconds"
			<< std:: endl;

	delete[] w;

	return 0;
}
