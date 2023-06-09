cmake_minimum_required (VERSION 3.16)

project (GRIT)
include (CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#----------------------------------------------------------------------
set(PROJECT_TPL_DIR ${CMAKE_SOURCE_DIR}/../TPLs)
set(PROJECT_EXAMPLE_DIR ${CMAKE_SOURCE_DIR}/../examples)

#----------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_TPL_DIR}/cmake/Modules)
find_package(Kokkos REQUIRED)
link_libraries(Kokkos::kokkos)
find_package(Boost COMPONENTS system filesystem mpi serialization program_options REQUIRED)
link_libraries(Boost::system Boost::filesystem Boost::mpi Boost::serialization Boost::program_options)
find_package(silo REQUIRED)
include_directories(${SILO_INCLUDE_DIRS})
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
find_package(MPI REQUIRED)

add_definitions(-DBOOST_NO_AUTO_PTR)

#----------------------------------------------------------------------
if(GRIT_USE_CUDA)
	#  set(CMAKE_CXX_COMPILER ${KOKKOS_DIR}/bin/nvcc_wrapper)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda")
endif()

include_directories(${MPI_CXX_INCLUDE_PATH})
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS})

#----------------------------------------------------------------------
set(HOSTNAME $ENV{HOSTNAME})
if(NOT HOSTNAME)
  SITE_NAME(HOSTNAME)
endif()

if(${HOSTNAME} MATCHES "^summitdev")
  set(SERIALEXEC "jsrun")
  set(SERIALEXEC_PREFLAGS "-n1 -g1")
  set(MPIEXEC "jsrun -n1 -g1 -c ALL_CPUS")
  set(MPIEXEC_NUMPROC_FLAG "-a")
elseif(${HOSTNAME} MATCHES "^summit")
  set(SERIALEXEC "jsrun")
  set(SERIALEXEC_PREFLAGS -n1 -g1 --smpiargs none)
  set(MPIEXEC "jsrun")
  set(MPIEXEC_PREFLAGS -n1 -g1 -c ALL_CPUS)
  set(MPIEXEC_NUMPROC_FLAG "-a")
elseif(${HOSTNAME} MATCHES "^percival")
  set(SERIALEXEC "aprun")
  set(MPIEXEC "aprun")
  set(MPIEXEC_NUMPROC_FLAG "-n")
  set(LAPACK_LIBRARIES "") #Compiler brings this in
elseif(${HOSTNAME} MATCHES "^titan")
  set(SERIALEXEC "aprun")
  set(MPIEXEC "aprun")
  set(MPIEXEC_NUMPROC_FLAG "-n")
  set(LAPACK_LIBRARIES "") #Compiler brings this in
endif()

#----------------------------------------------------------------------
set(LIBRARIESLIST ${LAPACK_LIBRARIES} ${Boost_LIBRARIES} ${KOKKOS_LIBRARIES} ${MPI_CXX_LIBRARIES}
        ${SILO_LIBRARIES} ${HDF5_LIBRARIES} ${CMAKE_DL_LIBS})
if(${HOSTNAME} MATCHES "^percival")
  set(LIBRARIESLIST ${LIBRARIESLIST} "-lmvec")
endif()
include_directories(${CMAKE_SOURCE_DIR})

#----------------------------------------------------------------------
message(STATUS "HOSTNAME is " ${HOSTNAME})
message(STATUS "CMAKE_BUILD_TYPE is " ${CMAKE_BUILD_TYPE})

#----------------------------------------------------------------------
add_subdirectory(init)
add_subdirectory(FDM)
add_subdirectory(IO)

#----------------------------------------------------------------------
add_library(Limestone RunParams.C)
set(DNSLIBS Limestone CollectiveSingleFile)

add_executable(LegranditeTest LegranditeTest.C)
target_link_libraries(LegranditeTest ${DNSLIBS} ${LIBRARIESLIST})
add_test(LegranditeTest ${SERIALEXEC} ${SERIALEXEC_PREFLAGS} LegranditeTest)

add_executable(SiloWriterTest SiloWriterTest.C)
target_link_libraries(SiloWriterTest ${DNSLIBS} ${LIBRARIESLIST})
add_test(SiloWriterTest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} SiloWriterTest)

add_executable(RestartTest RestartTest.C)
target_link_libraries(RestartTest ${DNSLIBS} ${LIBRARIESLIST})
add_test(RestartTest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} RestartTest)

add_executable(LintCompactTest LintCompactTest.C)
target_link_libraries(LintCompactTest ${DNSLIBS} ${LIBRARIESLIST})
add_test(LintCompactTest ${SERIALEXEC} ${SERIALEXEC_PREFLAGS} LintCompactTest)

add_executable(LintExchangeTest LintExchangeTest.C)
target_link_libraries(LintExchangeTest ${DNSLIBS} ${LIBRARIESLIST})
add_test(LintExchangeTest ${SERIALEXEC} ${SERIALEXEC_PREFLAGS} LintExchangeTest)

add_executable(ConstantVelAdvectTest ConstantVelAdvectTest.C)
target_link_libraries(ConstantVelAdvectTest ${DNSLIBS} ${LIBRARIESLIST})
add_test(ConstantVelAdvectTest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 27 ${MPIEXEC_PREFLAGS} ConstantVelAdvectTest)
set_tests_properties(ConstantVelAdvectTest PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)

add_executable(GoethiteTest GoethiteTest.C)
target_link_libraries(GoethiteTest ${DNSLIBS} ${LIBRARIESLIST})
add_test(GoethiteTest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 27 ${MPIEXEC_PREFLAGS} GoethiteTest)
set_tests_properties(GoethiteTest PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)

add_executable(SpigotControlTest SpigotControlTest.C)
target_link_libraries(SpigotControlTest ${DNSLIBS} ${LIBRARIESLIST})
add_test(SpigotControlTest ${SERIALEXEC} ${SERIALEXEC_PREFLAGS} SpigotControlTest)

