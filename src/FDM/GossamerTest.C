#include <stdio.h>
#include "FDM/Gossamer.h"
#include "init/Sinewave.h"
#include "GlobalVariables.h"
#include "IO/CollectiveSingleFile.h"

RunParams param;
boost::mpi::communicator globalcomm;
Corduroy cartcomm;
Greige grid;

const int NX=30, NY=35, NZ=24;
const int px=2 , py=2 , pz= 2;
const int GXL=1, GXR=6, GYL=5, GYR=2, GZL=3, GZR=4;

int main(int argc, char *argv[]){
  boost::mpi::environment env(argc, argv);
  Kokkos::ScopeGuard KokkosScopeGuard;
  param.set("px", std::to_string(px));
  param.set("py", std::to_string(py));
  param.set("pz", std::to_string(pz));

  cartcomm=Corduroy(globalcomm, px, py, pz);
  grid=Greige(-0.5*(NX*px-1), -0.5*(NY*py-1), -0.5*(NZ*pz-1), 1.0, 1.0, 1.0);

  if(globalcomm.rank()==0){
    std::cout << "Initialized MPI with " << globalcomm.size() << " ranks\n";
    std::cout << "Partitioned to cartesian topology  (" << px << "," << py << "," << pz << ")\n";
    std::cout << "Per MPI problem size ("               << NX << "," << NY << "," << NZ << ")\n";
  };

  const size_t NP=NX*NY*NZ;
  Yarn::ScalarFieldType F("F", NP);
  initSinewave(NP, NX, NY, NZ, F);

  const int NXG=NX+GXL+GXR;
  const int NYG=NY+GYL+GYR;
  const int NZG=NZ+GZL+GZR;
  const size_t NPG=NXG*NYG*NZG;
  Yarn::ScalarFieldType FG("FG", NPG);
  Gossamer<double, NX, NY, NZ, GXL, GXR, GYL, GYR, GZL, GZR>::fill_ghost(F, FG);

  double *Fa=new double[NPG];
  Yarn::ScalarFieldToFortranArray (NPG, Fa, FG);

  Pique<double> Fq(Fa, NXG, NYG, NZG, 1, -GXL, -GYL, -GZL, 0);
   double xl=grid.x(cartcomm.x.size()*NX)-grid.x(0);
   double yl=grid.y(cartcomm.y.size()*NY)-grid.y(0);
   double zl=grid.z(cartcomm.z.size()*NZ)-grid.z(0);
   double pi=4.0*atan(1.0);
  for(int k=-GZL; k<NZ+GZR; k++){
    for(int j=-GYL; j<NY+GYR; j++){
      for(int i=-GXL; i<NX+GXR; i++){
        double x =grid.x(cartcomm.x.rank()*NX+i);
        double y =grid.y(cartcomm.y.rank()*NY+j);
        double z =grid.z(cartcomm.z.rank()*NZ+k);
        double fx=sin(2.0*pi*x/xl);
        double fy=sin(2.0*pi*y/yl);
        double fz=sin(2.0*pi*z/zl);
        double f2=fx*fy*fz;
        if(std::abs(Fq(i,j,k)-f2)>1e-12) {
          printf("rsa err in ellipse %d %d %d %d %12.5e %12.5e\n", globalcomm.rank(), i, j, k, Fq(i,j,k), f2);
          return(1);
        }
      }
    }
  }
  return(0);
}
