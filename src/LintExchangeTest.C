#include <random>
#include <Kokkos_Random.hpp>
#include "Dust.h"
#include "Lint.h"
#include "GlobalVariables.h"

const int NX=115, NY= 84, NZ= 93; // Per MPI rank problem size
const float mx=1.4, my=0.8, mz=1.1; //No. of full waves across NX, NY, NZ

boost::mpi::communicator globalcomm;
Greige grid, localgrid;

class DustTest : public Dust {
  public:
    DustTest(uint64_t ssn_start=0) : Dust (ssn_start) {
      Dust::ScalarPointType P("P");
      Dust::ScalarPointType Q("Q");
      ScalarPointVariables.insert(std::make_pair("P", P));
      ScalarPointVariables.insert(std::make_pair("Q", Q));
    };
};

int main(int argc, char *argv[]){
  boost::mpi::environment env(argc, argv);
  Kokkos::ScopeGuard KokkosScopeGuard;
  printf ("%s on Kokkos execution space %s\n", argv[0], typeid (Kokkos::DefaultExecutionSpace).name());
  Kokkos::print_configuration(std::cout);

       grid=Greige(-3.0,-4.5,-2.8, 0.1, 0.15, 0.12);
  localgrid=Greige(-3.0,-4.5,-2.8, 0.1, 0.15, 0.12);

  double pi=4.0*atan(1.0);

  int numparcels=2;
  Lint<DustTest> Parcels;
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.2, 0.3);
  typedef Kokkos::Random_XorShift64_Pool<> GeneratorPool;
  GeneratorPool pool(345);

  for(int m=0; m<numparcels; m++) {
    DustTest tracers(0+DustTest::NDUST*m);
    DustTest::ScalarPointType P, Q;
    P=tracers.ScalarPointVariables.find("P")->second;
    Q=tracers.ScalarPointVariables.find("Q")->second;

    float killfraction=dist(mt);
    Kokkos::parallel_for(DustTest::NDUST, KOKKOS_LAMBDA(const size_t& n) {
        GeneratorPool::generator_type gen = pool.get_state();
        float p=gen.frand(1.0);
        if(p>killfraction) tracers.state(n) = DustTest::HEALTHY;
        tracers.loc(n,0)=gen.drand(NX);
        tracers.loc(n,1)=gen.drand(NY);
        tracers.loc(n,2)=gen.drand(NZ);
        double kx=2.0*pi*mx/NX;
        double ky=2.0*pi*my/NY;
        double kz=2.0*pi*mz/NZ;
        P (n)= cos(tracers.loc(n,0)*kx)
             * cos(tracers.loc(n,1)*ky)
             * cos(tracers.loc(n,2)*kz);
        Q(n)=  1.0 - P(n);
        pool.free_state(gen);
    } );
    Parcels.push_back(tracers);
  }

  printf("Lint Exchange Test %4zu parcels with fill percentages = ", Parcels.size());
  for(DustTest P: Parcels) printf("%9.6f ", 1.0-float(P.getcount(DustTest::UNOCCUPIED))/DustTest::NDUST);
  printf("\n");

  Parcels.write_silo("LintExchangeSent");

  Lint<DustTest> Parcelr;

  boost::mpi::request reqs[2];
  reqs[0]=globalcomm.isend(0, 351, Parcels);
  reqs[1]=globalcomm.irecv(0, 351, Parcelr);
  boost::mpi::wait_all(reqs, reqs+2);

  Parcelr.write_silo("LintExchangeRecv");

  return(0);
}
