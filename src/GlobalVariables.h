#ifndef GLOBALVARIABLES_H
#define GLOBALVARIABLES_H

#include <boost/mpi.hpp>
#include "RunParams.h"
#include "FDM/Corduroy.h"
#include "FDM/Greige.h"

extern double simtime;
extern double timestepsize;
extern double last_checkpoint_simtime;
extern RunParams param;
extern boost::mpi::communicator globalcomm;
extern Corduroy cartcomm;
extern Greige grid, localgrid;

#endif
