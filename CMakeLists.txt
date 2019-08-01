cmake_minimum_required(VERSION 3.12)
project(fastlinear)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(src/math)
include_directories(src/solver)
include_directories(src/utils)
include_directories(src/io)
include_directories(src/objective)
include_directories(src/linear)
include_directories(src/application)

#add_executable(test_eigen
#        src/solver/lbfgs.cpp
#        src/test/test_eigen.cpp)

#add_executable(test_lbfgs
#        src/solver/lbfgs.cpp
#        src/test/test_lbfgs.cpp)

add_executable(fastlinear
        src/io/dataset.cpp
        src/io/file_io.cpp
        src/io/parser.cpp
        src/linear/linear.cpp
        src/solver/lbfgs.cpp
        src/main.cpp)
