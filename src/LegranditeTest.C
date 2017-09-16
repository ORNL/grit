#include "Yarn.h"
#include "Dust.h"
#include "Inego.h"
#include "Legrandite.h"

#include <Kokkos_Random.hpp>

const int NX=115, NY= 84, NZ= 93; // Per MPI rank problem size
const int NH=1;
float mx=1.4, my=0.8, mz=1.1; //No. of full waves across NX, NY, NZ

class DustTest : public Dust {
  public:
    DustTest() {
      init_position();
    };
  private:
    void init_position() {
      Kokkos::Random_XorShift64_Pool<> rand_pool(23791);
      Kokkos::View<double [NDUST]> randloc("randloc");
      Kokkos::fill_random(randloc, rand_pool, NX);
      Kokkos::parallel_for(NDUST, KOKKOS_LAMBDA(const size_t n) { loc(n,0)=randloc(n); } );
      Kokkos::fill_random(randloc, rand_pool, NY);
      Kokkos::parallel_for(NDUST, KOKKOS_LAMBDA(const size_t n) { loc(n,1)=randloc(n); } );
      Kokkos::fill_random(randloc, rand_pool, NZ);
      Kokkos::parallel_for(NDUST, KOKKOS_LAMBDA(const size_t n) { loc(n,2)=randloc(n); } );
    };

};

int main(int argc, char *argv[]){
  Kokkos::initialize();

  double pi=4.0*atan(1.0);
  const size_t NG=(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH);
  Yarn::ScalarFieldType F("F", NG);

  Kokkos::parallel_for(NG, KOKKOS_LAMBDA (const size_t& n) {
      int      i, j, k;
      TUP::tie(i, j, k) =   Inego<NX+1+2*NH, NY+1+2*NH, NZ+1+2*NH, -NH, -NH, -NH>(n) ;
      double kx=2.0*pi*mx/NX;
      double ky=2.0*pi*my/NY;
      double kz=2.0*pi*mz/NZ;
      F (n)= cos(i*kx)
           * cos(j*ky)
           * cos(k*kz);
  } );

  DustTest tracers;
  Dust::ScalarPointType P("P");
  Dust::ScalarPointType Q("Q");

  Kokkos::parallel_for(DustTest::NDUST, KOKKOS_LAMBDA(const size_t& n) {
      double kx=2.0*pi*mx/NX;
      double ky=2.0*pi*my/NY;
      double kz=2.0*pi*mz/NZ;
      Q (n)= cos(tracers.loc(n,0)*kx)
           * cos(tracers.loc(n,1)*ky)
           * cos(tracers.loc(n,2)*kz);
  } );

  Legrandite<NH>::interpolate(NX, NY, NZ, F, tracers.loc, P);

  double err;
  size_t nd=DustTest::NDUST;
  Kokkos::parallel_reduce(nd, KOKKOS_LAMBDA(const size_t& n, double& lsum) {
      lsum+=(P(n)-Q(n))*(P(n)-Q(n));
  }, err);
  err/=double(nd);
  printf("NH NY NX NZ are %1d %4d %4d %4d\n", NH, NX, NY, NZ);
  printf("L2 norm error is %12.5e\n", err);

  Kokkos::finalize();
  return(0);
}
