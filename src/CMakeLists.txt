add_library(conway-gol STATIC conway-gol.cpp)
target_include_directories(conway-gol PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(seq sequential-conway.cpp)
target_link_libraries(seq conway-gol)

find_package(CUDA)
if(CUDA_FOUND)
add_executable(cuda-parallel parallel-conway.cu)
endif()

#copy kernels to bin
file(COPY CalcStep.cl DESTINATION ${PROJECT_SOURCE_DIR}/bin/)
file(COPY CalcStepGroups.cl DESTINATION ${PROJECT_SOURCE_DIR}/bin/)

add_executable(openclConway parallel-conway.cpp)
target_include_directories(openclConway PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(openclConway OpenCL)

