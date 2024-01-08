# To configure the correct python env in CLion, add to CMake options:
# -DPython3_ROOT_DIR=<path to python env bin dir>

cmake_minimum_required(VERSION 3.2)
cmake_policy(SET CMP0057 NEW)

project(hnsw_lib)

include_directories("${PROJECT_BINARY_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/hnswlib")


set(SOURCE_EXE main.cpp)
set(SOURCE_LIB sift_1b.cpp)

add_library(sift_test STATIC ${SOURCE_LIB})

add_executable(main ${SOURCE_EXE})
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  SET(CMAKE_CXX_FLAGS "-Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -ftree-vectorize")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  SET(CMAKE_CXX_FLAGS "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  SET(CMAKE_CXX_FLAGS "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize" )
endif()

add_executable(test_updates examples/updates_test.cpp)
add_executable(searchKnnCloserFirst_test examples/searchKnnCloserFirst_test.cpp)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")  # needed for pybind11
execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE pybind11_DIR
)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(hnswlib python_bindings/bindings.cpp)

target_link_libraries(main sift_test)
