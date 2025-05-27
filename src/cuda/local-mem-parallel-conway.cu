#include <iostream>
#include <unistd.h>
#include <chrono>

#include "conway.h"

__device__
int aliveNeighbours(bool* &w, int x, int y, const int N, const int M)
{
	auto idx = [&](const size_t i, const size_t j){return w[((i + N) % N) + (((j + M) % M) * N)];};

	return  idx(x-1, y-1) + idx(x, y-1) + idx(x+1, y-1) +
		idx(x-1, y)  + idx(x+1, y) +
		idx(x-1, y+1) + idx(x, y+1) + idx(x+1, y+1);
}

__global__
void calcStep(const bool* currentStep, bool* nextStep, const int N, const int M)
{
	const int lx = threadIdx.x + 1;
	const int ly = threadIdx.y + 1;
	const int gx = blockIdx.x * blockDim.x + threadIdx.x;
	const int gy = blockIdx.y * blockDim.y + threadIdx.y;

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
	bool* hWorld = new bool[N*M];

	init(hWorld);

	if (N <= 50 and M <= 100)
		std::cout << hWorld << std::endl;

	bool *dCurrent, *dNext;
	cudaMalloc(&dCurrent, N * M * sizeof(bool));
	cudaMalloc(&dNext, N * M * sizeof(bool));

	cudaMemcpy(dCurrent, hWorld, N * M * sizeof(bool), cudaMemcpyHostToDevice);

	size_t gridDimX = (N + blockSize - 1) / blockSize;
	size_t gridDimY = (M + blockSize - 1) / blockSize;

	dim3 blockDim(blockSize, blockSize);
	dim3 gridDim(gridDimX, gridDimY);

	const int maxStep = simSteps;
	int step = 0;
	long totalTimeNS = 0.;
	long totalTimeMS = 0.;
	while (step++ < maxStep)
	{
		auto start = std::chrono::high_resolution_clock::now();
		calcStep<<<gridDim, blockDim>>>(dCurrent, dNext, N, M);
		cudaDeviceSynchronize();
		auto end = std::chrono::high_resolution_clock::now();

		if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
			fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
			exit(1);
		}

		cudaMemcpy(hWorld, dNext, N * M * sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(dCurrent, dNext, N * M * sizeof(bool), cudaMemcpyDeviceToDevice);
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
