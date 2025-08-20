cmake_minimum_required(VERSION 3.10)

project(transformbench LANGUAGES CXX)

set(CMAKE_CUDA_ARCHITECTURES 80)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(TTG_ENABLE_CUDA ON)
else(CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "CUDA compiler not found")
endif(CMAKE_CUDA_COMPILER)
set(HAVE_CUDA ${CMAKE_CUDA_COMPILER} CACHE BOOL "True if we can compile .cu files")

option(USE_SUGGEST_LAYOUT "Use suggested layout instead of get_layout" ON)
option(DEBUG_TENSOR_TYPE "Compile-time print cute tensor types (breaks build)" OFF)
set(USE_CUBLASDX_VERSION "25.06" CACHE STRING "Version of cublasDx to use")


# fetch cublasDx
include(FetchContent)
FetchContent_Declare(
  cublasdx
  URL https://developer.download.nvidia.com/compute/cublasdx/redist/cublasdx/nvidia-mathdx-${USE_CUBLASDX_VERSION}.0.tar.gz
)
FetchContent_MakeAvailable(cublasdx)
FetchContent_GetProperties(cublasdx
	SOURCE_DIR CUBLASDX_SOURCE_DIR
	BINARY_DIR CUBLASDX_BINARY_DIR
)

# look for cublasDx
find_package(mathdx REQUIRED COMPONENTS cublasdx HINTS ${CUBLASDX_SOURCE_DIR}/nvidia/mathdx/25.06/)
if (TARGET mathdx::cublasdx)
  message(STATUS "Found cublasDx at ${mathdx_CUBLASDX_DIR}")
else()
  message(FATAL_ERROR "cublasDx not found")
endif()

# Simple interface that holds cublasDx and CUDA settings
add_library(libmra INTERFACE)
# Link against cublasDx and CUDA
target_link_libraries(libmra INTERFACE mathdx::cublasdx)
# Set the CUDA architecture
target_compile_definitions(libmra INTERFACE MRA_CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES})
# Enable support for constexpr and extended lambdas
target_compile_options(libmra INTERFACE --expt-relaxed-constexpr --extended-lambda)

# Add the transformbench executable
add_executable(transformbench transformbench.cu)

# Link against the MRA interface
target_link_libraries(transformbench PUBLIC libmra)

if (USE_SUGGEST_LAYOUT)
    # Enable using suggested layout instead of get_layout
    target_compile_definitions(transformbench PUBLIC USE_SUGGEST_LAYOUT)
endif (USE_SUGGEST_LAYOUT)

if (DEBUG_TENSOR_TYPE)
    # Enable compile-time printing of cute tensor types (breaks build)
    target_compile_definitions(transformbench PUBLIC DEBUG_TENSOR_TYPE)
endif (DEBUG_TENSOR_TYPE)
