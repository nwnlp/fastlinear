cmake_minimum_required(VERSION 3.12)
project(fastlinear)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++")

include_directories(src/math)
include_directories(src/solver)

add_executable(test_eigen
        src/solver/lbfgs.cpp
        src/test/test_eigen.cpp)

add_executable(test_lbfgs
        src/solver/lbfgs.cpp
        src/test/test_lbfgs.cpp)