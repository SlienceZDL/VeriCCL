cmake_minimum_required(VERSION 3.20)
project(vericcl_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# GMP: headers from a system libgmp-dev or an extracted copy
# (VERICCL_GMP_ROOT, e.g. .deps/gmp-dev/root/usr); the runtime library is the
# system libgmp.so.10 that gmpy2 also uses.
set(VERICCL_GMP_ROOT "" CACHE PATH "Prefix holding include/gmp.h")
find_path(GMP_INCLUDE_DIR gmp.h
  HINTS ${VERICCL_GMP_ROOT}/include ${VERICCL_GMP_ROOT}/include/x86_64-linux-gnu)
# Shared library only (names are tried in order across all directories): a
# static libgmp.a is not position independent and cannot go into the core.
find_library(GMP_LIBRARY NAMES libgmp.so.10 libgmp.so
  HINTS ${VERICCL_GMP_ROOT}/lib ${VERICCL_GMP_ROOT}/lib/x86_64-linux-gnu
  PATHS /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib)
if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
  message(FATAL_ERROR "GMP not found (gmp.h: ${GMP_INCLUDE_DIR}, library: ${GMP_LIBRARY}); set VERICCL_GMP_ROOT")
endif()

add_library(vericcl_core SHARED src/simulate.cpp)
target_include_directories(vericcl_core PRIVATE include ${GMP_INCLUDE_DIR})
target_link_libraries(vericcl_core PRIVATE ${GMP_LIBRARY})
target_compile_options(vericcl_core PRIVATE -O2 -Wall -Wextra -fvisibility=hidden)
set_target_properties(vericcl_core PROPERTIES CXX_VISIBILITY_PRESET hidden)
# Exported C entry points carry default visibility via the explicit attribute.
target_compile_definitions(vericcl_core PRIVATE VERICCL_CORE_BUILD)
