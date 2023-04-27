#include <random>
#include <Kokkos_Random.hpp>
#include "FDM/Yarn.h"
#include "Dust.h"
#include "Lint.h"
#include "GlobalVariables.h"

const int NX=115, NY= 84, NZ= 93; // Per MPI rank problem size
const float mx=1.4, my=0.8, mz=1.1; //No. of full waves across NX, NY, NZ

boost::mpi::communicator globalcomm;
Greige grid, localgrid;
Corduroy cartcomm;

class DustTest : public Dust {
  private:
    typedef Kokkos::Random_XorShift64_Pool<> GeneratorPool;
  public:
    DustTest(uint32_t seed=4911) {
      GeneratorPool pool(seed);
      Kokkos::parallel_for(NDUST, init_position(pool, state, loc));
      Dust::ScalarPointType P("P");
      Dust::ScalarPointType Q("Q");
      ScalarPointVariables.insert(std::make_pair("P", P));
      ScalarPointVariables.insert(std::make_pair("Q", Q));
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
  boost::mpi::environment env(argc, argv);
  Kokkos::ScopeGuard KokkosScopeGuard;
  if(globalcomm.rank()==0) Kokkos::print_configuration(std::cout);
  cartcomm=Corduroy(globalcomm, 2, 2, 2);

       grid=Greige(-3.0,-4.5,-2.8, 0.1, 0.15, 0.12);
  localgrid=Greige(grid.x(cartcomm.x.rank()*NX),
                   grid.y(cartcomm.y.rank()*NY),
                   grid.z(cartcomm.z.rank()*NZ), 0.1, 0.15, 0.12);

  double pi=4.0*atan(1.0);

  Lint<DustTest> Parcels;
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int> dist(3, 9);
  int numparcels=dist(mt);

  for(int n=0; n<numparcels; n++) {
    DustTest tracers(345+100*globalcomm.rank()+n);
    DustTest::ScalarPointType P, Q;
    P=tracers.ScalarPointVariables.find("P")->second;
    Q=tracers.ScalarPointVariables.find("Q")->second;
    Kokkos::parallel_for(DustTest::NDUST, KOKKOS_LAMBDA(const size_t& n) {
        double kx=2.0*pi*mx/NX;
        double ky=2.0*pi*my/NY;
        double kz=2.0*pi*mz/NZ;
        P (n)= cos(tracers.loc(n,0)*kx)
             * cos(tracers.loc(n,1)*ky)
             * cos(tracers.loc(n,2)*kz);
        Q(n)=  1.0 - P(n);
    } );
    Parcels.push_back(tracers);
  }

  Parcels.write_silo("SiloWriterTest0");

  for(auto tracers : Parcels) {
    Kokkos::parallel_for(DustTest::NDUST, KOKKOS_LAMBDA(const size_t& n) {
        tracers.loc(n,0)+=1.0;
        tracers.loc(n,1)+=1.20;
        tracers.loc(n,2)+=1.40;
    } );
  }

  Parcels.write_silo("SiloWriterTest1");
  for(auto tracers : Parcels) {
    Kokkos::parallel_for(DustTest::NDUST, KOKKOS_LAMBDA(const size_t& n) {
        tracers.loc(n,0)+=1.0;
        tracers.loc(n,1)+=1.20;
        tracers.loc(n,2)+=1.40;
    } );
  }

  Parcels.write_silo("SiloWriterTest2");
  for(auto tracers : Parcels) {
    Kokkos::parallel_for(DustTest::NDUST, KOKKOS_LAMBDA(const size_t& n) {
        tracers.loc(n,0)+=1.0;
        tracers.loc(n,1)+=1.20;
        tracers.loc(n,2)+=1.40;
    } );
  }

  Parcels.write_silo("SiloWriterTest3");

return(0);
}
