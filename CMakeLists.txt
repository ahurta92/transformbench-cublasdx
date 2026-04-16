cmake_minimum_required(VERSION 3.10)

project(transformbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(HIP)
if(CMAKE_HIP_COMPILER)
  enable_language(HIP)
else()
  message(FATAL_ERROR "HIP compiler not found")
endif()

# Simple interface that holds HIP settings
add_library(libmra INTERFACE)
target_compile_definitions(libmra INTERFACE MRA_HAVE_HIP=1)

# Add the transformbench executable
add_executable(transformbench_hip transformbench.hip)
target_link_libraries(transformbench_hip PUBLIC libmra)

# Correctness test: GPU L1 vs CPU reference (mirrors transform3d.cc)
add_executable(validate validate.hip)
target_link_libraries(validate PUBLIC libmra)

# Multi-level correctness test (-l selects level, -K and -N override defaults)
add_executable(validate_levels validate_levels.hip)
target_link_libraries(validate_levels PUBLIC libmra)
