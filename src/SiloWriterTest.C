#include "Yarn.h"
#include "Dust.h"
#include "Inego.h"
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
      GeneratorPool pool( 4911);
      Kokkos::parallel_for(NDUST, init_position(pool, loc));
      Dust::ScalarPointType P("P");
      Dust::ScalarPointType Q("Q");
      ScalarPointVariables.insert(std::make_pair("P", P));
      ScalarPointVariables.insert(std::make_pair("Q", Q));
    };
    /* ---------------------------------------------------------------------- */
    struct init_position {
      GeneratorPool pool;
      LocationVecType loc;
      //constructor
      init_position(GeneratorPool pool_, LocationVecType loc_)
        : pool(pool_), loc(loc_) { } ;
      KOKKOS_INLINE_FUNCTION
      void operator()(const size_t n) const {
        GeneratorPool::generator_type gen = pool.get_state();
        loc(n,0)=gen.drand(NX);
        loc(n,1)=gen.drand(NY);
        loc(n,2)=gen.drand(NZ);
        pool.free_state(gen);
      }
    };
    /* ---------------------------------------------------------------------- */
};

int main(int argc, char *argv[]){
  Kokkos::initialize();
  printf ("%s on Kokkos execution space %s\n", argv[0], typeid (Kokkos::DefaultExecutionSpace).name());
  Kokkos::DefaultExecutionSpace::print_configuration(std::cout);

  double pi=4.0*atan(1.0);
  const size_t NG=(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH);
  DustTest tracers;

  Kokkos::parallel_for(DustTest::NDUST, KOKKOS_LAMBDA(const size_t& n) {
      double kx=2.0*pi*mx/NX;
      double ky=2.0*pi*my/NY;
      double kz=2.0*pi*mz/NZ;
      DustTest::ScalarPointType P, Q;
      P=tracers.ScalarPointVariables.find("P")->second;
      Q=tracers.ScalarPointVariables.find("Q")->second;
      P (n)= cos(tracers.loc(n,0)*kx)
           * cos(tracers.loc(n,1)*ky)
           * cos(tracers.loc(n,2)*kz);
      Q(n)=  1.0 - P(n);
  } );

  tracers.write_silo();

  Kokkos::finalize();
  return(0);
}
