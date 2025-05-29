cmake_minimum_required(VERSION 3.22)

# project(tarea2 LANGUAGES CXX CUDA) ejecutar sin cuda
project(tarea2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/)

add_subdirectory(src)
