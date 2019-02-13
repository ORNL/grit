#include "Sinewave.h"
#include "FDM/Pique.h"
#include "GlobalVariables.h"

void initSinewave(const size_t NP, const int NX, const int NY, const int NZ, Yarn::ScalarFieldType F){
  assert(NP==NX*NY*NZ);
  double pi=4.0*atan(1.0);
  double *Fa=new double[NP];
  Pique<double> Fq(Fa, NX, NY, NZ);

  double xl=grid.x(cartcomm.x.size()*NX)-grid.x(0);
  double yl=grid.y(cartcomm.y.size()*NY)-grid.y(0);
  double zl=grid.z(cartcomm.z.size()*NZ)-grid.z(0);

  for(int k=0; k<NZ; k++){
    for(int j=0; j<NY; j++){
      for(int i=0; i<NX; i++){
        double x =grid.x(cartcomm.x.rank()*NX+i);
        double y =grid.y(cartcomm.y.rank()*NY+j);
        double z =grid.z(cartcomm.z.rank()*NZ+k);
        double fx=sin(2.0*pi*x/xl);
        double fy=sin(2.0*pi*y/yl);
        double fz=sin(2.0*pi*z/zl);
        Fq(i,j,k)=fx*fy*fz;
      }
    }
  }
  Yarn::FortranArrayToScalarField(NX*NY*NZ, Fa, F);
  delete[] Fa;
}
