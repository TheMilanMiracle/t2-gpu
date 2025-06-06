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
	auto t0 = std::chrono::high_resolution_clock::now();
	bool* hWorld = new bool[N*M];

	init(hWorld);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::cout	<< "Initial data set in "
				<< std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
				<< " microseconds"
				<< std::endl;

	t0 = std::chrono::high_resolution_clock::now();
	bool *dCurrent, *dNext;
	cudaMalloc(&dCurrent, N * M * sizeof(bool));
	cudaMalloc(&dNext, N * M * sizeof(bool));

	cudaMemcpy(dCurrent, hWorld, N * M * sizeof(bool), cudaMemcpyHostToDevice);
	t1 = std::chrono::high_resolution_clock::now();

	std::cout	<< "Data copied to device in "
				<< std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
				<< " microseconds"
				<< std::endl;

	size_t numBlocks = (N*M + blockSize - 1) / blockSize;

	dim3 blockDim(blockSize, blockSize);
	dim3 gridDim(numBlocks, numBlocks);

	const int maxStep = simSteps;
	int step = 0;
	auto cycleStart = std::chrono::high_resolution_clock::now();
	while (step++ < maxStep)
	{
		t0 = std::chrono::high_resolution_clock::now();
		calcStep<<<numBlocks, blockSize>>>(dCurrent, dNext, N, M);
		cudaDeviceSynchronize();
		t1 = std::chrono::high_resolution_clock::now();

		if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
			fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
			exit(1);
		}

		std::cout	<< "step "
					<< step
					<< " computed in "
					<< std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
					<< " microseconds"
					<< std::endl;

		t0 = std::chrono::high_resolution_clock::now();
		cudaMemcpy(hWorld, dNext, N * M, cudaMemcpyDeviceToHost);
		cudaMemcpy(dCurrent, dNext, N * M, cudaMemcpyDeviceToDevice);
		cudaMemset(dNext, false, N * M * sizeof(bool));
		t1 = std::chrono::high_resolution_clock::now();

		std::cout	<< "Updated buffers in "
					<< std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
					<< " microseconds"
					<< std::endl;

	}

	auto cycleEnd = std::chrono::high_resolution_clock::now();

	std::cout	<< "Total time: "
				<< std::chrono::duration_cast<std::chrono::microseconds>(cycleEnd - cycleStart).count()
				<< " microseconds"
				<< std::endl;

	std::cout	<< "avg computing time: "
				<< std::chrono::duration_cast<std::chrono::microseconds>(cycleEnd - cycleStart).count() / static_cast<float>(maxStep)
				<< " microseconds"
				<< std:: endl;

	cudaFree(dCurrent);
	cudaFree(dNext);
	delete[] hWorld;

	return 0;
}
