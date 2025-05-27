#ifndef PARALLELCUDA_H
#define PARALLELCUDA_H
#include <ostream>

constexpr int blockSize = 32;
inline size_t N;
inline size_t M;
inline int simSteps = 0;

std::ostream &operator<<(std::ostream &os, bool* arr);
std::ostream &operator<<(std::ostream &os, bool** arr);

void init(bool* &arr);
void init(bool** &arr);

int runParallelConway();
int run2DArrayParallelConway();
int runLocalMemParallelConway();

#endif