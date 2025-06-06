#include <iostream>
#include <unistd.h>
#include <chrono>

#include "conway.h"

__device__
int aliveNeighbours(bool* &w, size_t x, size_t y, const size_t N, const size_t M)
{
	auto idx = [&](const size_t i, const size_t j){return w[((i + N) % N) + (((j + M) % M) * N)];};

	return  idx(x-1, y-1) + idx(x, y-1) + idx(x+1, y-1) +
		idx(x-1, y)  + idx(x+1, y) +
		idx(x-1, y+1) + idx(x, y+1) + idx(x+1, y+1);
}

__global__
void calcStep2(const bool* currentStep, bool* nextStep, const size_t N, const size_t M)
{
	const size_t lx = threadIdx.x + 1;
	const size_t ly = threadIdx.y + 1;
	const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t gy = blockIdx.y * blockDim.y + threadIdx.y;

	auto idx = [&](const size_t i, const size_t j){return ((i + N) % N) + (((j + M) % M) * N);};

	__shared__ bool localStep[blockSize + 2][blockSize +2];

	localStep[lx][ly] = currentStep[idx(gx, gy)];

	if (threadIdx.x == 0 or
		threadIdx.x == blockDim.x - 1 or
		threadIdx.y == 0 or
		threadIdx.y == blockDim.y - 1
		) {
		int dh = (threadIdx.x == 0) ? -1 : (threadIdx.x == blockDim.x - 1 ? 1 : 0);
		int dv = (threadIdx.y == 0) ? -1 : (threadIdx.y == blockDim.y - 1 ? 1 : 0);

		int cells = (dv and dh) ? 3 : 1;

		int d[3][2] = {
			{dh, dv},
			{dh, 0},
			{0, dv}
		};

		for (int c = 0; c < cells; c++) {
			int dx = d[c][0];
			int dy = d[c][1];
			localStep[lx + dx][ly + dy] = currentStep[idx(gx + dx, gy + dy)];
		}
	}

	__syncthreads();

	int neighbours = localStep[lx-1][ly-1] + localStep[lx][ly-1] + localStep[lx+1][ly-1] +
					 localStep[lx-1][ly]   + localStep[lx+1][ly] +
					 localStep[lx-1][ly+1] + localStep[lx][ly+1] + localStep[lx+1][ly+1];
	nextStep[idx(gx, gy)] = neighbours == 3 or (neighbours == 2 and localStep[lx][ly]);
}

int runLocalMemParallelConway()
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

	size_t gridDimX = (N + blockSize - 1) / blockSize;
	size_t gridDimY = (M + blockSize - 1) / blockSize;

	dim3 blockDim(blockSize, blockSize);
	dim3 gridDim(gridDimX, gridDimY);

	const int maxStep = simSteps;
	int step = 0;
	auto cycleStart = std::chrono::high_resolution_clock::now();
	while (step++ < maxStep)
	{
		t0 = std::chrono::high_resolution_clock::now();
		calcStep2<<<gridDim, blockDim>>>(dCurrent, dNext, N, M);
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
		cudaMemcpy(hWorld, dNext, N * M * sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(dCurrent, dNext, N * M * sizeof(bool), cudaMemcpyDeviceToDevice);
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
