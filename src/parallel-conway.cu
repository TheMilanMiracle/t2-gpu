#include <iostream>
#include <unistd.h>
#include <chrono>
#include <vector>

int N, M;

__host__
std::ostream &operator<<(std::ostream &os, bool* w)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
			os << (w[i + j * N] ? "o" : ".") << " ";
		os << std::endl;
	}
	return os;
}

__host__
void init(bool* world) {
	auto idx = [&](const int i, const int j){return i + j * N;};

	const std::vector<std::pair<int,int>> gosper_glider_gun = {
		{24,0},{22,1},{24,1},
		{12,2},{13,2},{20,2},{21,2},{34,2},{35,2},
		{11,3},{15,3},{20,3},{21,3},{34,3},{35,3},
		{0,4},{1,4},{10,4},{11,4},{15,4},{16,4},{22,4},{24,4},
		{0,5},{1,5},{10,5},{12,5},{16,5},{17,5},{22,5},{24,5},
		{10,6},{11,6},{12,6},{13,6},{14,6},{15,6},{16,6},
		{0,7},{1,7},{2,7},{3,7},{4,7},{5,7},{6,7},{7,7},{8,7},{9,7},
		{10,7},{11,7},{12,7},{13,7},{14,7},{15,7},{16,7}
	};

	for (auto [i, j] : gosper_glider_gun)
		world[idx(i, j)] = true;
}

__device__
int worldIdx(int x, int y, const int N, const int M) {
	x = (x + N) % N;
	y = (y + M) % M;
	return x + y * N;
}

__device__
int aliveNeighbours(const bool* &w, int x, int y, const int N, const int M)
{
	return  w[worldIdx(x-1, y-1, N, M)] + w[worldIdx(x, y-1, N, M)] + w[worldIdx(x+1, y-1, N, M)] +
		w[worldIdx(x-1, y, N, M)]  + w[worldIdx(x+1, y, N, M)] +
		w[worldIdx(x-1, y+1, N, M)] + w[worldIdx(x, y+1, N, M)] + w[worldIdx(x+1, y+1, N, M)];
}

__global__
void calcStep(const bool* currentStep, bool* nextStep, const int N, const int M)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (int k = 0; k < N * M; k += blockDim.x * gridDim.x)
	{
		int i = idx / M;
		int j = idx % M;

		int neighbours = aliveNeighbours(currentStep, i, j, N, M);
		nextStep[worldIdx(i, j, N, M)] = neighbours == 3 or (neighbours == 2 and currentStep[worldIdx(i, j, N, M)]);
	}
}

int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		std::cout << "usage: ./cuda {N} {M} {steps}" << std::endl;
		exit(1);
	}

	N = atoi(argv[1]);
	M = atoi(argv[2]);

	bool hWorld[N*M];
	memset(hWorld, false, N*M*sizeof(bool));

	init(hWorld);

	if (N <= 50 and M <= 100)
		std::cout << hWorld << std::endl;

	bool *dCurrent, *dNext;
	cudaMalloc(&dCurrent, N * M * sizeof(bool));
	cudaMalloc(&dNext, N * M * sizeof(bool));

	cudaMemcpy(dCurrent, hWorld, N * M * sizeof(bool), cudaMemcpyHostToDevice);

	int blockSize = 16;
	int numBlocks = (N*M + blockSize - 1) / blockSize;

	dim3 blockDim(blockSize, blockSize);
	dim3 gridDim(numBlocks, numBlocks);

	const int maxStep = atoi(argv[3]);
	int step = 0;
	long totalTime = 0.;
	while (step++ <= maxStep)
	{
		auto start = std::chrono::high_resolution_clock::now();
		calcStep<<<numBlocks, blockSize>>>(dCurrent, dNext, N, M);
		cudaDeviceSynchronize();
		auto end = std::chrono::high_resolution_clock::now();

		cudaMemcpy(hWorld, dNext, N * M, cudaMemcpyDeviceToHost);
		cudaMemcpy(dCurrent, dNext, N * M, cudaMemcpyDeviceToDevice);
		cudaMemset(dNext, false, N * M * sizeof(bool));


		std::cout << "step " << step-1 << " computed in " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "ns" << std::endl;
		totalTime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		if (N <= 50 and M <= 100)
		{
			std::cout << "\033[2J\033[H" << hWorld << std::endl;
			usleep(500000);
		}

	}

	std::cout << "avg computing time: " << totalTime / static_cast<float>(maxStep) << "ns" << std::endl;

	cudaFree(dCurrent);
	cudaFree(dNext);

	return 0;
}
