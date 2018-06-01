#include <random>
#include <Kokkos_Random.hpp>
#include "Dust.h"
#include "Lint.h"
#include "GlobalVariables.h"

const int NX= 45, NY= 34, NZ= 23;
const int px=  3, py=  3, pz=  3;

boost::mpi::communicator globalcomm;
Corduroy cartcomm;
Greige globalgrid, localgrid;
bool periodic_x=true, periodic_y=true, periodic_z=true;

class DustTest : public Dust {
  public:
    DustTest(uint64_t ssn_start=0) : Dust (ssn_start) {
      Dust::ScalarPointType D("D");
      Dust::Vectr3PointType V("V");
      ScalarPointVariables.insert(std::make_pair("D", D));
      Vectr3PointVariables.insert(std::make_pair("V", V));
    };
};

/* ---------------------------------------------------------------------- */
int main(int argc, char *argv[]){
  boost::mpi::environment env(argc, argv);
  Kokkos::initialize();
  if(globalcomm.rank()==0) Kokkos::DefaultExecutionSpace::print_configuration(std::cout);
  cartcomm=Corduroy(globalcomm, px, py, pz);

  globalgrid=Greige(-3.0,-4.5,-2.8, 0.1, 0.15, 0.12);
   localgrid=Greige(globalgrid.x(cartcomm.x.rank()*NX),
                    globalgrid.y(cartcomm.y.rank()*NY),
                    globalgrid.z(cartcomm.z.rank()*NZ), 0.1, 0.15, 0.12);

  int numparcels=2;
  Lint<DustTest> Parcels;
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.0, 0.0);
  typedef Kokkos::Random_XorShift64_Pool<> GeneratorPool;
  GeneratorPool pool(34*globalcomm.rank()+729);

  for(int i=0; i<numparcels; i++) {
    DustTest tracers(100000*globalcomm.rank()+DustTest::NDUST*i);
    DustTest::ScalarPointType D=tracers.ScalarPointVariables.find("D")->second;
    DustTest::Vectr3PointType V=tracers.Vectr3PointVariables.find("V")->second;

    float killfraction=dist(mt);
    Kokkos::parallel_for(DustTest::NDUST, KOKKOS_LAMBDA(const size_t& n) {
        GeneratorPool::generator_type gen = pool.get_state();
        float p=gen.frand(1.0);
        if(p>killfraction) tracers.state(n) = DustTest::HEALTHY;
        tracers.loc(n,0)=gen.drand(NX);
        tracers.loc(n,1)=gen.drand(NY);
        tracers.loc(n,2)=gen.drand(NZ);
        //D(n)=1.0+gen.drand(7.0);
        V(n,0)=-5.0+gen.drand(10.0);
        V(n,1)=-5.0+gen.drand(10.0);
        V(n,2)=-5.0+gen.drand(10.0);
        D(n)=sqrt(V(n,0)*V(n,0)+V(n,1)*V(n,1)+V(n,2)*V(n,2));
        pool.free_state(gen);
    } );
    Parcels.push_back(tracers);
  }

  Parcels.write_silo("AdvectTest00");
  for(int i=1; i<= 5; i++) {
    for(auto p: Parcels) {
      DustTest::Vectr3PointType V=p.Vectr3PointVariables.find("V")->second;
      Kokkos::parallel_for(DustTest::NDUST, KOKKOS_LAMBDA(const size_t& n) {
        if(p.state(n)==DustTest::HEALTHY) 
          for(int m=0; m<3; m++) p.loc(n,m)+=V(n,m);
      } );
    }
    Parcels.exchange(NX, NY, NZ);

    char filename[100];
    sprintf(filename, "AdvectTest%02d", i);
    Parcels.write_silo(filename);
  }

  Kokkos::finalize();

  return(0);
}
