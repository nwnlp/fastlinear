cmake_minimum_required(VERSION 3.12)
project(fastlinear)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions( -DWEIGHT_T_USE_DOUBLE)
add_definitions( -DLABEL_T_USE_DOUBLE)


include_directories(src/math)
include_directories(src/solver)
include_directories(src/utils)
include_directories(src/io)
include_directories(src/objective)
include_directories(src/linear)
include_directories(src/application)
include_directories(src/math)
include_directories(src/metrics)


#add_executable(test_alglib
#        src/math/alglib/optimization.cpp
#        src/math/alglib/linalg.cpp
#        src/math/alglib/alglibinternal.cpp
#        src/math/alglib/alglibmisc.cpp
#        src/math/alglib/solvers.cpp
#        src/math/alglib/ap.cpp
#        src/test/test_alglib.cpp)

#add_executable(test_eigen
#        src/solver/lbfgs.cpp
#        src/test/test_eigen.cpp)
#
#add_executable(test_lbfgs
#        src/solver/lbfgs.cpp
#        src/test/test_lbfgs.cpp)

add_executable(fastlinear
        src/io/dataset.cpp
        src/io/file_io.cpp
        src/io/parser.cpp
        src/linear/linear.cpp
        src/solver/lbfgs.cpp
        src/math/alglib/optimization.cpp
        src/math/alglib/linalg.cpp
        src/math/alglib/alglibinternal.cpp
        src/math/alglib/alglibmisc.cpp
        src/math/alglib/solvers.cpp
        src/math/alglib/ap.cpp
        src/main.cpp)
