#include <iostream>
#include <unistd.h>
#include <chrono>

#include "conway.h"


__device__
int aliveNeighbours(const bool* &w, size_t x, size_t y, const size_t N, const size_t M)
{
	auto idx = [&](const size_t i, const size_t j){return w[((i + N) % N) + (((j + M) % M) * N)];};

	return  idx(x-1, y-1) + idx(x, y-1) + idx(x+1, y-1) +
		idx(x-1, y)  + idx(x+1, y) +
		idx(x-1, y+1) + idx(x, y+1) + idx(x+1, y+1);
}

__global__
void calcStep(const bool* currentStep, bool* nextStep, const size_t N, const size_t M)
{
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t i = idx / M;
	const size_t j = idx % M;

	auto _idx = [&](const size_t i, const size_t j){return ((i + N) % N) + (((j + M) % M) * N);};

	int neighbours = aliveNeighbours(currentStep, i, j, N, M);
	nextStep[_idx(i, j)] = neighbours == 3 or (neighbours == 2 and currentStep[_idx(i, j)]);
}

int runParallelConway()
{
	bool* hWorld = new bool[N*M];

	init(hWorld);

	if (N <= 50 and M <= 100)
		std::cout << hWorld << std::endl;

	bool *dCurrent, *dNext;
	cudaMalloc(&dCurrent, N * M * sizeof(bool));
	cudaMalloc(&dNext, N * M * sizeof(bool));

	cudaMemcpy(dCurrent, hWorld, N * M * sizeof(bool), cudaMemcpyHostToDevice);

	size_t numBlocks = (N*M + blockSize - 1) / blockSize;

	dim3 blockDim(blockSize, blockSize);
	dim3 gridDim(numBlocks, numBlocks);

	const int maxStep = simSteps;
	int step = 0;
	long totalTimeNS = 0.;
	long totalTimeMS = 0.;
	while (step++ < maxStep)
	{
		auto start = std::chrono::high_resolution_clock::now();
		calcStep<<<numBlocks, blockSize>>>(dCurrent, dNext, N, M);
		cudaDeviceSynchronize();
		auto end = std::chrono::high_resolution_clock::now();

		if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
			fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
			exit(1);
		}

		cudaMemcpy(hWorld, dNext, N * M, cudaMemcpyDeviceToHost);
		cudaMemcpy(dCurrent, dNext, N * M, cudaMemcpyDeviceToDevice);
		cudaMemset(dNext, false, N * M * sizeof(bool));

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
			std::cout << "\033[2J\033[H" << hWorld << std::endl;
			usleep(500000);
		}

	}

	std::cout	<< "avg computing time: "
				<< totalTimeMS / static_cast<float>(maxStep)
				<< " ms ("
				<< totalTimeNS / static_cast<float>(maxStep)
				<< " ns)";

	cudaFree(dCurrent);
	cudaFree(dNext);
	delete[] hWorld;

	return 0;
}
