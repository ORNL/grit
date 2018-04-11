#ifndef GLOBALVARIABLES_H
#define GLOBALVARIABLES_H

#include <boost/mpi.hpp>
#include "Corduroy.h"
#include "Greige.h"

extern double simtime;
extern double timestepsize;
extern double last_checkpoint_simtime;
extern boost::mpi::communicator globalcomm;
extern Corduroy cartcomm;
extern Greige globalgrid, localgrid;
extern bool periodic_x, periodic_y, periodic_z;

#endif
