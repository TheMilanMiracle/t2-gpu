#include <iostream>
#include <set>

#include "conway.h"

int main(int argc, char* argv[]) {
    if (argc != 5)
    {
        std::cout << "usage: ./seq {N} {M} {steps} {mode}" << std::endl;
        std::cout << "\tmodes: simple | array2d | localmemory" << std::endl;
        exit(1);
    }

    N = atoi(argv[1]);
    M = atoi(argv[2]);
    simSteps = atoi(argv[3]);

    std::string mode = argv[4];
    const std::set<std::string> validModes = {"simple", "array2d", "localmemory"};

    if (validModes.find(mode) == validModes.end()) {
        std::cerr << "unknown mode \"" << mode << "\".\n";
        std::cerr << "\tValid modes: simple | array2d | localmemory\n";
        return 1;
    }

    if (mode == "simple")
        runParallelConway();
    else if (mode == "array2d")
        run2DArrayParallelConway();
    else // mode == "localmemory"
        runLocalMemParallelConway();

    return 0;
}