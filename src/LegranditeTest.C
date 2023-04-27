#include "FDM/Yarn.h"
#include "Dust.h"
#include "FDM/Inego.h"
#include "Legrandite.h"

#include <Kokkos_Random.hpp>

const int NX=115, NY= 84, NZ= 93; // Per MPI rank problem size
const int NH=1;
const float mx=1.4, my=0.8, mz=1.1; //No. of full waves across NX, NY, NZ

class DustTest : public Dust {
  private:
    typedef Kokkos::Random_XorShift64_Pool<> GeneratorPool;
  public:
    DustTest() {
      GeneratorPool pool(23791);
      Kokkos::parallel_for(NDUST, init_position(pool, state, loc));
    };
    /* ---------------------------------------------------------------------- */
    struct init_position {
      GeneratorPool pool;
      HealthPointType state;
      LocationVecType loc;
      //constructor
      init_position(GeneratorPool pool_, HealthPointType state_, LocationVecType loc_)
        : pool(pool_), state(state_), loc(loc_) { } ;
      KOKKOS_INLINE_FUNCTION
      void operator()(const size_t n) const {
        GeneratorPool::generator_type gen = pool.get_state();
        state(n) = HEALTHY;
        loc(n,0)=gen.drand(NX);
        loc(n,1)=gen.drand(NY);
        loc(n,2)=gen.drand(NZ);
        pool.free_state(gen);
      }
    };
    /* ---------------------------------------------------------------------- */
};

int main(int argc, char *argv[]){
  Kokkos::ScopeGuard KokkosScopeGuard;
  Kokkos::print_configuration(std::cout);

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

  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::duration<float> fsec;
  Kokkos::fence();
  auto start_clock=Time::now();

  int numsteps=10;
  for(int i=0; i<numsteps; i++) {
  Legrandite<NH>::interpolate(NX, NY, NZ, F, tracers.loc, tracers.state, P);
  }

  Kokkos::fence();
  auto finish_clock=Time::now();
  fsec fs=finish_clock-start_clock;
  printf("NDUST is %zu\n", tracers.NDUST);
  printf("Time taken          is %6.3f (msecs) \n", fs.count()*1e3                                     );
  printf("Time taken per step is %6.3f (nsecs) \n", fs.count()*1e9/float(numsteps)/float(tracers.NDUST));

  double err;
  size_t nd=DustTest::NDUST;
  Kokkos::parallel_reduce(nd, KOKKOS_LAMBDA(const size_t& n, double& lsum) {
      if(tracers.state(n)==Dust::HEALTHY) lsum+=(P(n)-Q(n))*(P(n)-Q(n));
  }, err);
  err=sqrt(err/double(nd));
  printf("NH NY NX NZ are %1d %4d %4d %4d\n", NH, NX, NY, NZ);
  printf("L2 norm error is %12.5e\n", err);

  return(0);
}
