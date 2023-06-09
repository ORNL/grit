set(GRIT_TEST_ROOT_DIRECTORY "$ENV{HOME}/grit/workspace/nightly-gnu-openmp")
set(CTEST_SOURCE_DIRECTORY "${GRIT_TEST_ROOT_DIRECTORY}/grit/src")
set(CTEST_BINARY_DIRECTORY "${GRIT_TEST_ROOT_DIRECTORY}/build")

SITE_NAME(HOSTNAME)
set(CTEST_SITE ${HOSTNAME})
set(CTEST_BUILD_NAME "linux-gnu-openmp")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_BUILD_FLAGS "-j4")

find_program(CTEST_GIT_COMMAND NAMES git)
set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
set(CTEST_CONFIGURE_COMMAND "${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE:STRING=Debug -DKOKKOS_DIR=/opt/sw/Kokkos/BuildOpenMP_d3a9419 -DSILO_DIR=/opt/sw/silo/4.10.2 ${CTEST_SOURCE_DIRECTORY}")

ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})
ctest_start("Nightly")
ctest_update()
ctest_configure()
ctest_build()
ctest_test()
ctest_submit()
