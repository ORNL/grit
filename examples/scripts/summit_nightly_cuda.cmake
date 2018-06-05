get_filename_component(GRIT_TEST_ROOT_DIRECTORY "$ENV{HOME}/grit/workspace/nightly-summit-cuda" REALPATH)
set(CTEST_SOURCE_DIRECTORY "${GRIT_TEST_ROOT_DIRECTORY}/grit/src")
set(CTEST_BINARY_DIRECTORY "${GRIT_TEST_ROOT_DIRECTORY}/build")

set(CTEST_SITE $ENV{CTEST_SITE})
set(CTEST_BUILD_NAME "summit-cuda")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_BUILD_FLAGS "-j4")

SET(ENV{FC}  "gfortran")
SET(ENV{CC}  "gcc")
SET(ENV{CXX} "g++")

find_program(CTEST_GIT_COMMAND NAMES git)
set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
set(CTEST_CONFIGURE_COMMAND "${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE:STRING=Release -DGRIT_USE_CUDA:BOOL=ON -DBOOST_ROOT=$ENV{BOOST_DIR} -DKOKKOS_DIR=$ENV{KOKKOS_DIR} -DBoost_USE_STATIC_LIBS:BOOL=ON -DHDF5_ROOT=$ENV{HDF5_DIR} -DSILO_DIR=$ENV{SILO_DIR} -DCMAKE_CXX_FLAGS=-expt-extended-lambda ${CTEST_SOURCE_DIRECTORY}")

ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})
ctest_start("Nightly")
ctest_update()
ctest_configure()
ctest_build()
ctest_test()
ctest_submit()
