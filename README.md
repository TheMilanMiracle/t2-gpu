# Parallel Implementation of Conway's Game of Life
This mini-project was developed as an assignment for the course on GPU Computing CC7515.  It consists of 2 implementations of the same problem, one in CUDA and the other in OpenCL. 

Other than achiving a working solution, there were implemented 3 different configurations of the same problem:
- Using 1 dimension to represent the problem and global memory for thread access.
- Using 2 dimensions and global memory.
- Using 1 dimensión and local memory per group of threads.
- Using 2 dimension and local memory (_only in OpenCL_)

## Building the project
```
mkdir build
cd build
cmake ..
make
```

This will generate 3 binaries: `cuda`, `opencl` and `seq`. Which, as expected, run the implementation in cuda, in opencl and sequential, respectevely. 

## Running the simulation
The sequential implementation can be runned as followed:
```
./bin/seq N M steps
```
Where N and M are the grid sizes, and steps represents how many steps have to be calculated.

Similarly to the sequential, the cuda implementation runs with the following command:
```
./cuda N M steps mode
```
`mode` corresponds to which of the configurations will be runned. The options are:
- `simple`: 1 dimension and global memory
- `array2d`: 2 dimensions and global memory
- `localmemory`: 1 dimension and local memory

Lastly, the opencl implementation:
```
./openclConway N M steps mode
```
Where mode corresponds to one of the following:
- `0`: 1 dimension and global memory
- `1`: 2 dimensions and global memory
- `2`: 1 dimension and local memory
- `3`: 2 dimensions and local memory