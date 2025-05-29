#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdarg>
#include <map>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>


// #define THREADS 40
// // blocks of 4 by 4
// #define BLOCK_SIZE 8

using std::chrono::microseconds;


// Queue abstraction del aux
class Queue
{
private:
    cl::Platform _platform;
    cl::Device _device;
    cl::Context _context;
    cl::CommandQueue _queue;
    std::vector<cl::Buffer> _buffers;
    cl::Kernel _kernel;
    cl::Program _program;
    void setKernelArgs(int idx) {} // Base case for recursion

    template <typename Last>
    void setKernelArgs(int idx, Last last)
    {
        _kernel.setArg(idx, last);
    };

    template <typename First, typename... Rest>
    void setKernelArgs(int idx, First first, Rest... rest)
    {
        _kernel.setArg(idx, first);
        setKernelArgs(idx + 1, rest...);
    };

public:
    Queue()
    {
        // Query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        _platform = platforms.front();
        std::cout << "Platform: " << _platform.getInfo<CL_PLATFORM_NAME>()
                  << std::endl;

        // Get a list of devices on this platform
        std::vector<cl::Device> devices;
        // Select the platform.
        _platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        _device = devices.front();
        std::cout << "Device: " << _device.getInfo<CL_DEVICE_NAME>()
                  << std::endl;

        std::vector<::size_t> maxWorkItems = _device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
        std::cout << "Max sizes: " << maxWorkItems[0] << " " << maxWorkItems[1] << " " << maxWorkItems[2] << std::endl;
        std::cout << "Max group size: " << _device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
        // Create a context
        _context = cl::Context(devices);

        // Create a command queue
        // Select the device.
        _queue = cl::CommandQueue(_context, _device);
    }

    // Manda a la cola una escritura de buffer
    template <typename T>
    int addBuffer(std::vector<T> &data, cl_mem_flags flags = CL_MEM_READ_WRITE)
    {
        cl::Buffer buffer(_context, flags, data.size() * sizeof(T));
        _queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, data.size() * sizeof(T),
                                  data.data());
        _buffers.push_back(buffer);
        return _buffers.size() - 1;
    }

    template <typename T>
    void updateBuffer(std::vector<T> &data, int index){
        _queue.enqueueWriteBuffer(_buffers[index], CL_TRUE, 0, data.size() * sizeof(T),
                                  data.data());
    }

    // Lee el kernel de un archivo
    void setKernel(const std::string &file, const std::string &kernelName)
    {
        std::ifstream sourceFile(file);
        std::stringstream sourceCode;
        sourceCode << sourceFile.rdbuf();

        // Make and build program from the source code
        _program = cl::Program(_context, sourceCode.str(), true);

        // Make kernel
        _kernel = cl::Kernel(_program, kernelName.c_str());
    }

    // Lee a data el buffer #index
    template <typename T>
    void readBuffer(std::vector<T> &data, int index = 0)
    {
        _queue.enqueueReadBuffer(_buffers[index], CL_TRUE, 0,
                                 data.size() * sizeof(T), data.data());
    }


    template <typename... Args>
    cl::Event operator()(cl::NDRange globalSize, cl::NDRange localSize, Args... args)
    {
        // Set the kernel arguments
        for (size_t i = 0; i < _buffers.size(); ++i)
        {
            _kernel.setArg(i, _buffers[i]);
        }
        //add local memory arg
        //_kernel.setArg(_buffers.size(), cl::LocalSpaceArg);
        // add a +1 to index for local array
        setKernelArgs(_buffers.size(), args...);

        cl::Event event;
        _queue.enqueueNDRangeKernel(_kernel, cl::NullRange, globalSize, localSize,
                                    nullptr, &event);
        event.wait();
        return event;
    }
};

void initWorld(std::vector<int> &world, const int M, const int N){
    const std::vector<std::pair<int,int>> glider = {
		{4, 4},
        {5, 5},
        {6, 3}, {6, 4}, {6, 5}
	};
    int offsets[5] = {10, 15, 20, 25, 30};
    for(auto k : offsets){
        for (auto [i, j] : glider){
		    world[(j + k)%N + i * M] = 1;
            world[j + (i + k)%M * M] = 1;
            world[(j + k)%N + (i + k)%M * M] = 1;
        }
    }
    
}

void printWorld(std::vector < int > &world, int N, int M){
    for(int i = 0; i < N*M; i++){
        if(world[i]) std::cout << "■";
        else std::cout << "□";
        if((i+1) % M == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printWorldPlain(std::vector < int > &world, int N, int M){
    for(int i = 0; i < N*M; i++){
        std::cout << world[i];
        if((i+1) % M == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
}

void report(std::string rep){
    std::cout << rep << std::endl;
}


int main(int argc, char const *argv[])
{
    if (argc != 5)
	{
        report("usage: ./openclConway {N} {M} {steps} {0 | 1 | 2}");
		exit(1);
	}

	int N = atoi(argv[1]);
	int M = atoi(argv[2]);

    int type = atoi(argv[4]);
    if(type != 0 && type != 1 && type != 2){
        report("usage: ./openclConway {N} {M} {steps} {type: 0 | 1 | 2}");
		exit(1);
    }


    try
    {

        Queue q;
        report("Queue created");

        auto t_start = std::chrono::high_resolution_clock::now();

        // empty world and next state
        std::vector<int> dCurrent(N*M, 0), dNext(N*M);
        //fill world
        initWorld(dCurrent, M, N);
        auto t_end = std::chrono::high_resolution_clock::now();
        auto t_create_data =
            std::chrono::duration_cast<microseconds>(t_end - t_start).count();

        report("World created");

        t_start = std::chrono::high_resolution_clock::now();
        q.addBuffer(dCurrent, CL_MEM_READ_ONLY);
        q.addBuffer(dNext, CL_MEM_WRITE_ONLY);
        t_end = std::chrono::high_resolution_clock::now();
        auto t_copy_to_device =
            std::chrono::duration_cast<microseconds>(t_end - t_start).count();
        report("Values copied to device");

        // Read the program source
        std::string source = type == 0 ? "bin/CalcStep.cl" : (type == 1 ? "bin/CalcStep2D.cl" : "bin/CalcStepGroups.cl");
        q.setKernel(source, "calcStep");
        report("Kernel " + source + " sent to device");

        int block_size = 64;
        int block_size2d = 8;

        // Execute the function on the device 
        cl::NDRange globalSize; 
        cl::NDRange localSize; 
        if(type == 0){
            globalSize = cl::NDRange((N * M) / block_size);
            localSize = cl::NDRange(block_size);
        }
        else{
            globalSize = cl::NDRange(N / block_size2d, M / block_size2d);
            localSize = cl::NDRange(block_size2d, block_size2d);
        }
        
        
        report("Starting cycle");
        
        const int maxStep = atoi(argv[3]);
        int step = 0;
        std::vector< std::string > kernel_per_step (maxStep+1);
        std::vector< std::string > memory_per_step (maxStep+1);

        auto t_start_cycle = std::chrono::high_resolution_clock::now();
        while (step++ <= maxStep){
            
            t_start = std::chrono::high_resolution_clock::now();
            // calls kernel, the buffers are passed as first arguments in the order they were added
            // N and M are passed as the rest of the kernel arguments
            cl::Event event = q(globalSize, localSize, N, M);
            event.wait();
            t_end = std::chrono::high_resolution_clock::now();
            auto t_kernel_step = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
            
            t_start = std::chrono::high_resolution_clock::now();
            // Copy the output variable from device to host
            // Copies it to current, for next step
            q.readBuffer(dNext, 1);

            // write data in dNext to buffer index 0 (dCurrent)
            q.updateBuffer(dNext, 0);
            t_end = std::chrono::high_resolution_clock::now();
            auto t_kernel_memory = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
            
            kernel_per_step[step-1] = std::to_string(t_kernel_step);
            memory_per_step[step-1] = std::to_string(t_kernel_memory);
        }

        report("Finished cycle");

        t_end = std::chrono::high_resolution_clock::now();
        auto t_kernel =
            std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start_cycle)
                .count();

        report("Time to create data: " + std::to_string(t_create_data) + " microseconds");
        report("Time to copy data to device: " + std::to_string(t_copy_to_device) + " microseconds");
        report("Time to execute kernel: " + std::to_string(t_kernel) + " microseconds");
        
        report("For analysis:");
        report(std::to_string(t_create_data) + " " + std::to_string(t_copy_to_device) + " " + std::to_string(t_kernel));
        
        for(int i = 0; i < maxStep; i++){
            report(kernel_per_step[i] + " " + memory_per_step[i]);
        }
    }
    catch (cl::Error err)
    {
        std::cerr << "Error (" << err.err() << "): " << err.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
    return 0;
}