add_library(conway-gol STATIC conway-gol.cpp)
target_include_directories(conway-gol PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(seq sequential-conway.cpp)
target_link_libraries(seq conway-gol)

add_executable(cuda-parallel parallel-conway.cu)
