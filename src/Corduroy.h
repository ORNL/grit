#ifndef CORDUROY_H
#define CORDUROY_H

#include <boost/mpi.hpp>

class Corduroy {
  public:
    boost::mpi::communicator x , y , z ; // lines
    boost::mpi::communicator xy, yz, zx; //planes

  public:
    /* ---------------------------------------------------------------------- */
    Corduroy() : 
        x (MPI_COMM_NULL, boost::mpi::comm_duplicate),
        y (MPI_COMM_NULL, boost::mpi::comm_duplicate),
        z (MPI_COMM_NULL, boost::mpi::comm_duplicate),
        xy(MPI_COMM_NULL, boost::mpi::comm_duplicate),
        yz(MPI_COMM_NULL, boost::mpi::comm_duplicate),
        zx(MPI_COMM_NULL, boost::mpi::comm_duplicate)
        {}; 
    /* ---------------------------------------------------------------------- */
    Corduroy(boost::mpi::communicator globalcomm, int px, int py, int pz) {
      if(globalcomm.size()!=px*py*pz) 
          throw std::domain_error("Cartesian decomposition does not match processor count"); 

      // Get the xyz coordinates
      int gid = globalcomm.rank();
      int zid = gid/(px*py);
      int yid = (gid-zid*px*py)/px;
      int xid = (gid-zid*px*py)%px;

      // Split into planes
      xy=globalcomm.split(zid); 
      yz=globalcomm.split(xid); 
      zx=globalcomm.split(yid); 

      //Extract lines from planes
      x=xy.split(yid, xid); 
      y=yz.split(zid, yid); 
      z=zx.split(xid, zid); 
    }; 
}; 

#endif
