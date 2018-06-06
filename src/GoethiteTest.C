#include "FDM/Yarn.h"
#include "Dust.h"
#include "FDM/Inego.h"
#include "Goethite.h"
#include "FDM/Tuck.h"
#include "RunParams.h"
#include "FDM/Greige.h"
#include <Kokkos_Random.hpp>

const int NX= 47, NY= 73, NZ=107; // Per MPI rank problem size
const int NH=1;
const float mx=1.0, my=2.0, mz=3.0; //No. of full waves across NX, NY, NZ
const int   px=  3, py=  3, pz=  3;

Corduroy cartcomm;
RunParams param;
boost::mpi::communicator globalcomm;
Greige grid, localgrid;

class DustTest: public Dust {
   private:
  typedef Kokkos::Random_XorShift64_Pool<> GeneratorPool;
   public:
  DustTest() {
          GeneratorPool pool(23792+236*globalcomm.rank());
          Kokkos::parallel_for(NDUST, init_position(pool, loc, state));
  };
    /* ---------------------------------------------------------------------- */
    //functor to initialize the particles
    struct init_position {
      GeneratorPool pool;
      LocationVecType loc;
      PointHealthType state;
      //constructor
      init_position(GeneratorPool pool_, LocationVecType loc_, PointHealthType state_)
        : pool(pool_), loc(loc_), state(state_){ } ;
      KOKKOS_INLINE_FUNCTION
      void operator()(const size_t n) const {
        GeneratorPool::generator_type gen = pool.get_state();
        loc(n,0)=gen.drand(NX);
        loc(n,1)=gen.drand(NY);
        loc(n,2)=gen.drand(NZ);
        state(n)=HEALTHY;
        pool.free_state(gen);
      }
    };
};


/* ---------------------------------------------------------------------- */
int main(int argc, char *argv[]){
  boost::mpi::environment env(argc,argv);
  Kokkos::ScopeGuard KokkosScopeGuard;
  if(globalcomm.rank()==0) Kokkos::DefaultExecutionSpace::print_configuration(std::cout);

  param.set("px", std::to_string(px));
  param.set("py", std::to_string(py));
  param.set("pz", std::to_string(pz));
  param.set("periodic_x", "true");
  param.set("periodic_y", "true");
  param.set("periodic_z", "true" );
  cartcomm=Corduroy(globalcomm, px, py, pz);

       grid=Greige(-3.0,-4.5,-2.8, 0.1, 0.1, 0.1);
  localgrid=Greige(grid.x(cartcomm.x.rank()*NX),
                   grid.y(cartcomm.y.rank()*NY),
                   grid.z(cartcomm.z.rank()*NZ), 0.1, 0.1, 0.1);

  double pi=4.0*atan(1.0);
  size_t NDUST=DustTest::NDUST;
  DustTest tracers;
  Dust::ScalarPointType P("P");
  Kokkos::parallel_for(NDUST, KOKKOS_LAMBDA(const size_t& n) {
    double kx=2.0*pi*mx/(px*NX);
    double ky=2.0*pi*my/(px*NY);
    double kz=2.0*pi*mz/(px*NZ);
    P (n)= fabs(cos(tracers.loc(n,0)*kx)
              * cos(tracers.loc(n,1)*ky)
              * cos(tracers.loc(n,2)*kz));
  } );

  const size_t NP=NX*NY*NZ;
  const size_t NG=(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH);
  Yarn::ScalarFieldType F("F", NG);
  Yarn::ScalarFieldType G("G", NP);

  Goethite<NX,NY,NZ,NH>::deposit(F, tracers.loc, tracers.state, P);
  Tuck    <NX,NY,NZ,NH>::unfill_ghost(F,G);

  double total_particle;
  Kokkos::parallel_reduce(NDUST, KOKKOS_LAMBDA(const size_t& n, double& lsum) { lsum+=P(n); },total_particle);
  total_particle=boost::mpi::all_reduce(globalcomm, total_particle, std::plus<double>());

  double total_deposit;
  Kokkos::parallel_reduce(NG, KOKKOS_LAMBDA(const size_t& n, double& lsum) { lsum+=F(n); },total_deposit);
  total_deposit=boost::mpi::all_reduce(globalcomm, total_deposit, std::plus<double>());

  double total_tuck;
  Kokkos::parallel_reduce(NP, KOKKOS_LAMBDA(const size_t& n, double& lsum) { lsum+=G(n); },total_tuck);
  total_tuck=boost::mpi::all_reduce(globalcomm, total_tuck, std::plus<double>());

  if(globalcomm.rank()==0) {
    printf("Total sources: %12.5e %12.5e %12.5e\n", total_particle, total_deposit, total_tuck);
  }

  if(fabs(total_tuck-total_particle)>1e-8*total_particle) return(1);
  return(0);
}
