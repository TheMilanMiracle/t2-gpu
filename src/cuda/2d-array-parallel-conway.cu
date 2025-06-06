#include <iostream>
#include <unistd.h>
#include <chrono>

#include "conway.h"


__device__
int aliveNeighbours(bool** &w, size_t x, size_t y, const size_t N, const size_t M) {
	auto idx = [&](size_t i, size_t j){return w[(i+N)%N][(j+M)%M];};

	return  idx(x-1,y-1)	+ idx(x, y-1) + idx(x+1, y-1) +
			idx(x-1, y)		+ idx(x+1, y) +
			idx(x-1, y+1)	+ idx(x, y+1) + idx(x+1, y+1);
}

__global__
void calcStep1(bool** currentStep, bool** nextStep, const size_t N, const size_t M) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N * M) return;

	size_t i = idx / M;
	size_t j = idx % M;

	int neighbours = aliveNeighbours(currentStep, i, j, N, M);
	nextStep[i][j] = neighbours == 3 or (neighbours == 2 and currentStep[i][j]);
}

int run2DArrayParallelConway() {
	int maxStep = simSteps;

	auto t0 = std::chrono::high_resolution_clock::now();
	bool **hWorld = new bool*[N];
	for (int i = 0; i < N; i++) {
		hWorld[i] = new bool[M];
	}

	init(hWorld);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::cout	<< "Initial data set in "
				<< std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
				<< " microseconds"
				<< std::endl;

	t0 = std::chrono::high_resolution_clock::now();
	bool *dCurrentData, *dNextData;
	cudaMalloc(&dCurrentData, N * M * sizeof(bool));
	cudaMalloc(&dNextData, N * M * sizeof(bool));

	bool **dCurrent, **dNext;
	cudaMalloc(&dCurrent, N * sizeof(bool*));
	cudaMalloc(&dNext, N * sizeof(bool*));

	bool **hCurrent = new bool*[N];
	bool **hNext = new bool*[N];
	for (int i = 0; i < N; i++) {
		hCurrent[i] = dCurrentData + i * M;
		hNext[i] = dNextData + i * M;
	}

	cudaMemcpy(dCurrent, hCurrent, N * sizeof(bool*), cudaMemcpyHostToDevice);
	cudaMemcpy(dNext, hNext, N * sizeof(bool*), cudaMemcpyHostToDevice);

	for (int i = 0; i < N; i++)
		cudaMemcpy(dCurrentData + i * M, hWorld[i], M * sizeof(bool), cudaMemcpyHostToDevice);

	t1 = std::chrono::high_resolution_clock::now();

	std::cout	<< "Data copied to device in "
				<< std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
				<< " microseconds"
				<< std::endl;

	int numBlocks = (N*M + blockSize - 1) / blockSize;

	dim3 blockDim(blockSize, blockSize);
	dim3 gridDim(numBlocks, numBlocks);

    int step = 0;
	auto cycleStart = std::chrono::high_resolution_clock::now();
    while (step++ < maxStep)
    {

    	t0 = std::chrono::high_resolution_clock::now();
        calcStep1<<<numBlocks, blockSize>>>(dCurrent, dNext, N, M);
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
    	for (int i = 0; i < N; i++)
    		cudaMemcpy(hWorld[i], dNextData + i * M, M * sizeof(bool), cudaMemcpyDeviceToHost);

    	std::swap(dCurrent, dNext);
    	std::swap(dCurrentData, dNextData);

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

	for (int i = 0; i < N; i++) delete[] hWorld[i];
	delete[] hWorld;
	delete[] hCurrent;
	delete[] hNext;

	cudaFree(dCurrentData);
	cudaFree(dNextData);
	cudaFree(dCurrent);
	cudaFree(dNext);

    return 0;
}
