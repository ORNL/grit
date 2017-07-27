#include "Yarn.h"
#include "Dust.h"

const int NX=120, NY=140, NZ=96; // Per MPI rank problem size
const int NH=2; 
float mx=1, my=1, mz=1; //No. of full waves across NX, NY, NZ


int main(int argc, char *argv[]){
  Kokkos::initialize();

  const size_t NG=(NX+2*NH)*(NY+2*NH)*(NZ+2*NH);
  Yarn::ScalarFieldType F("F", NG);

  Dust tracers;

  Kokkos::finalize();
  return(0);
}
