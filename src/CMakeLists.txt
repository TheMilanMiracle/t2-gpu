add_library(
        conway STATIC
        cuda/2d-array-parallel-conway.cu
        cuda/local-mem-parallel-conway.cu
        cuda/simple-parallel-conway.cu
        conway.cpp
)
target_include_directories(conway PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(seq sequential-conway.cpp)
target_link_libraries(seq conway)

add_executable(cuda cuda/cuda-main.cu)
target_link_libraries(cuda conway)


#copy kernels to bin
file(COPY opencl/CalcStep.cl DESTINATION ${PROJECT_SOURCE_DIR}/bin/kernel/)
file(COPY opencl/CalcStep2D.cl DESTINATION ${PROJECT_SOURCE_DIR}/bin/kernel/)
file(COPY opencl/CalcStepGroups.cl DESTINATION ${PROJECT_SOURCE_DIR}/bin/kernel/)

add_executable(opencl opencl/opencl-main.cpp)
target_include_directories(opencl PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(opencl OpenCL)

