#include "FDM/Yarn.h"
#include "FDM/Inego.h"
#include "FDM/Tuck.h"
#include "GlobalVariables.h"
#include "IO/CollectiveSingleFile.h"

const int NX=115, NY= 84, NZ= 93; // Per MPI rank problem size
const int NH=1;
const float mx=2.5, my=1.7, mz=2.0; //No. of full waves across NX, NY, NZ
const int   px=  3, py=  3, pz=  3;

RunParams param;
boost::mpi::communicator globalcomm;
Corduroy cartcomm;
Greige grid;

int main(int argc, char *argv[]){
  boost::mpi::environment env(argc, argv);
  Kokkos::ScopeGuard KokkosScopeGuard;
  if(globalcomm.rank()==0) Kokkos::DefaultExecutionSpace::print_configuration(std::cout);

  param.set("px", std::to_string(px));
  param.set("py", std::to_string(py));
  param.set("pz", std::to_string(pz));
  param.set("periodic_x", "false");
  param.set("periodic_y", "false");
  param.set("periodic_z", "true" );
  cartcomm=Corduroy(globalcomm, px, py, pz);

       grid=Greige(-3.0,-4.5,-2.8, 0.1, 0.1, 0.1);

  double pi=4.0*atan(1.0);
  const size_t NP=NX*NY*NZ;
  const size_t NG=(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH);
  Yarn::ScalarFieldType F("F", NG);
  Yarn::ScalarFieldType G("G", NP);

  double kx=2.0*pi*mx/(px*NX);
  double ky=2.0*pi*my/(py*NY);
  double kz=2.0*pi*mz/(pz*NZ);
  int    rx=cartcomm.x.rank();
  int    ry=cartcomm.y.rank();
  int    rz=cartcomm.z.rank();

  Kokkos::parallel_for(NP, KOKKOS_LAMBDA (const size_t& n) {
      int      i, j, k;
      TUP::tie(i, j, k) =   Inego<NX       , NY       , NZ       ,   0,   0,   0>(n) ;
      double C = cos((double(rx*NX+i)+0.5)*kx)
               * cos((double(ry*NY+j)+0.5)*ky)
               * cos((double(rz*NZ+k)+0.5)*kz);
      double filtersum=0.0;
      for(int kk=-NH; kk<2+NH; kk++) {
        for(int jj=-NH; jj<2+NH; jj++) {
          for(int ii=-NH; ii<2+NH; ii++) {
        double delx=ii-0.5;
        double dely=jj-0.5;
        double delz=kk-0.5;
               filtersum +=sqrt(delx*delx+dely*dely+delz*delz);
      } } }
      for(int kk=-NH; kk<2+NH; kk++) {
        for(int jj=-NH; jj<2+NH; jj++) {
          for(int ii=-NH; ii<2+NH; ii++) {
        double delx=ii-0.5;
        double dely=jj-0.5;
        double delz=kk-0.5;
        double filtercoeff=sqrt(delx*delx+dely*dely+delz*delz)/filtersum;
        size_t nn   =   Inego<NX+1+2*NH, NY+1+2*NH, NZ+1+2*NH, -NH, -NH, -NH>(i+ii,j+jj,k+kk) ;
        Kokkos::atomic_add(&F(nn), filtercoeff*C);
      } } }
  } );

  Tuck<NX,NY,NZ,NH>::unfill_ghost(F, G);

  CollectiveWriteScalartoSingleFile ("TuckG.dat", NX, NY, NZ, G);
  if(globalcomm.rank()==0)
      WriteBOVforScalarinSingleFile ("TuckG.bov", NX, NY, NZ, "TuckG.dat", "G");

  return(0);
}
