cmake_minimum_required (VERSION 2.8)

project (GRIT)
include (CTest)

#----------------------------------------------------------------------
enable_language(Fortran)

#----------------------------------------------------------------------
set(PROJECT_TPL_DIR ${CMAKE_SOURCE_DIR}/../TPLs)
set(PROJECT_EXAMPLE_DIR ${CMAKE_SOURCE_DIR}/../examples)

#----------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_TPL_DIR}/cmake/Modules)
find_package(kokkos REQUIRED)
include_directories(${KOKKOS_INCLUDE_DIRS})
find_package(BLAS)
find_package(LAPACK)
find_package(MPI REQUIRED)

#----------------------------------------------------------------------
if(GRIT_USE_CUDA)
  set(CMAKE_CXX_COMPILER ${KOKKOS_DIR}/bin/nvcc_wrapper)
endif()

include_directories(${MPI_CXX_INCLUDE_PATH})
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS})

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fopenmp")
endif()

#----------------------------------------------------------------------
set(HOSTNAME $ENV{HOSTNAME})
if(NOT HOSTNAME)
  SITE_NAME(HOSTNAME)
endif()

#----------------------------------------------------------------------
set(LIBRARIESLIST ${LAPACK_LIBRARIES} ${KOKKOS_LIBRARIES} ${MPI_CXX_LIBRARIES} ${CMAKE_DL_LIBS})
include_directories(${CMAKE_SOURCE_DIR})

#----------------------------------------------------------------------
message(STATUS "HOSTNAME is " ${HOSTNAME})
message(STATUS "CMAKE_BUILD_TYPE is " ${CMAKE_BUILD_TYPE})

#----------------------------------------------------------------------
add_executable(LegranditeTest LegranditeTest.C)
target_link_libraries(LegranditeTest ${DNSLIBS} ${LIBRARIESLIST})
add_test(LegranditeTest LegranditeTest)
