#include <iostream>
#include <unistd.h>
#include <chrono>

#include "conway.h"


__device__
int aliveNeighbours(bool** &w, int x, int y, const int N, const int M) {
	auto idx = [&](int i, int j){return w[(i+N)%N][(j+M)%M];};

	return  idx(x-1,y-1)	+ idx(x, y-1) + idx(x+1, y-1) +
			idx(x-1, y)		+ idx(x+1, y) +
			idx(x-1, y+1)	+ idx(x, y+1) + idx(x+1, y+1);
}

__global__
void calcStep(bool** currentStep, bool** nextStep, const int N, const int M) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N * M) return;

	int i = idx / M;
	int j = idx % M;

	int neighbours = aliveNeighbours(currentStep, i, j, N, M);
	nextStep[i][j] = neighbours == 3 or (neighbours == 2 and currentStep[i][j]);
}

int run2DArrayParallelConway() {
	int maxStep = simSteps;

	bool **hWorld = new bool*[N];
	for (int i = 0; i < N; i++) {
		hWorld[i] = new bool[M];
	}

	init(hWorld);

	if (N <= 50 && M <= 100)
		std::cout << hWorld << std::endl;

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

	int blockSize = 32;
	int numBlocks = (N*M + blockSize - 1) / blockSize;

	dim3 blockDim(blockSize, blockSize);
	dim3 gridDim(numBlocks, numBlocks);

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

    	for (int i = 0; i < N; i++)
    		cudaMemcpy(hWorld[i], dNextData + i * M, M * sizeof(bool), cudaMemcpyDeviceToHost);

    	std::swap(dCurrent, dNext);
    	std::swap(dCurrentData, dNextData);

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

        if (N <= 50 && M <= 100) {
	        std::cout << "\033[2J\033[H" << hWorld << std::endl;
        	usleep(500000);
        }
    }

	std::cout	<< "avg computing time: "
			<< totalTimeMS / static_cast<float>(maxStep)
			<< " ms ("
			<< totalTimeNS / static_cast<float>(maxStep)
			<< " ns)";

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
