#include "boost/mpi.hpp"
#include "GlobalVariables.h"

const int NX=32, NY=32, NZ=32;
const int px= 2, py=2 , pz=2 ;

boost::mpi::communicator globalcomm;
Corduroy cartcomm;
Greige grid;

int main(int argc, char *argv[]){
  boost::mpi::environment env(argc, argv);

  return(0);
}
