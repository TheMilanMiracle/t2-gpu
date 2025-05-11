#include "conway-gol.hpp"

#include <iostream>
#include <unistd.h>
#include <chrono>
#include <vector>



int aliveNeighbours(const World &w, int x, int y)
{
	return  w(x-1, y-1) + w(x, y-1) + w(x+1, y-1) +
		w(x-1, y)  + w(x+1, y) +
		w(x-1, y+1) + w(x, y+1) + w(x+1, y+1);
}



int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		std::cout << "usage: ./seq {N} {M} {steps}" << std::endl;
		exit(1);
	}

	int N = atoi(argv[1]), M = atoi(argv[2]);

	World w(N, M);
	w.init();

	if (N <= 50 and M <= 100)
		std::cout << w << std::endl;

	const int maxStep = atoi(argv[3]);
	int step = 0;
	long totalTime = 0.;
	while (step++ <= maxStep)
	{
		World nextStep(N, M);

		auto start = std::chrono::high_resolution_clock::now();
		{
			for (int i = 0; i < N; i++)
				for(int j = 0; j < M; j++)
				{
					int neighbours = aliveNeighbours(w, i, j);
					nextStep(i, j) = neighbours == 3 or (neighbours == 2 and w(i, j));

				}
		}
		auto end = std::chrono::high_resolution_clock::now();

		w = nextStep;
		
		std::cout << "step " << step-1 << " computed in " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "ns" << std::endl;
		totalTime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		if (N <= 50 and M <= 100)
		{
			std::cout << "\033[2J\033[H" << w << std::endl;
			usleep(500000);
		}
	}

	std::cout << "avg computing time: " << totalTime / static_cast<float>(maxStep) << "ns" << std::endl;

	return 0;

}
