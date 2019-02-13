add_executable(TuckTest TuckTest.C)
target_link_libraries(TuckTest CollectiveSingleFile ${LIBRARIESLIST})
add_test(TuckTest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 27 ${MPIEXEC_PREFLAGS} TuckTest)
set_tests_properties(TuckTest PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)

add_executable(GossamerTest GossamerTest.C)
target_link_libraries(GossamerTest Init ${LIBRARIESLIST})
add_test(GossamerTest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} GossamerTest)

