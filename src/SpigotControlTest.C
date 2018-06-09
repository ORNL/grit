#include <random>
#include <Kokkos_Random.hpp>
#include "FDM/Yarn.h"
#include "Dust.h"
#include "Lint.h"
#include "GlobalVariables.h"
#include "Goethite.h"
#include "FDM/Tuck.h"
#include "IO/CollectiveSingleFile.h"
#include "Spigot.h"

const int NX= 47, NY= 32, NZ= 29; // Per MPI rank problem size
const int NH=1;
const float mx=2.0, my=1.0, mz=1.0; //No. of full waves across NX, NY, NZ

boost::mpi::communicator globalcomm;
RunParams param;
Corduroy cartcomm;
Greige grid, localgrid;

/* ---------------------------------------------------------------------- */
class DustTest : public Dust {
  public:
    DustTest(uint64_t ssn_start=0) : Dust (ssn_start) {
      Dust::ScalarPointType P("P");
      Dust::ScalarPointType W("W");
      ScalarPointVariables.insert(std::make_pair("P", P));
      ScalarPointVariables.insert(std::make_pair("W", W));
    };
};
/* ---------------------------------------------------------------------- */
struct calcmax {
  Yarn::StridedScalarFieldType S;
  typedef double value_type;
  //constructor
  calcmax(Yarn::StridedScalarFieldType S_) : S(S_) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t n, double &maxS) const {
    if(maxS<S(n)) maxS=S(n);
  }
  KOKKOS_INLINE_FUNCTION
  void init(double & maxS) const {
    maxS=-1e50;
  }
  KOKKOS_INLINE_FUNCTION
  void join(volatile double &maxS, volatile const double &maxS2) const {
    if(maxS<maxS2) maxS=maxS2;
  }
};
/* ---------------------------------------------------------------------- */
struct calcmin {
  Yarn::StridedScalarFieldType S;
  typedef double value_type;
  //constructor
  calcmin(Yarn::StridedScalarFieldType S_) : S(S_) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t n, double &minS) const {
    if(minS>S(n)) minS=S(n);
  }
  KOKKOS_INLINE_FUNCTION
  void init(double & minS) const {
    minS=1e50;
  }
  KOKKOS_INLINE_FUNCTION
  void join(volatile double &minS, volatile const double &minS2) const {
    if(minS>minS2) minS=minS2;
  }
};

/* ---------------------------------------------------------------------- */
int main(int argc, char *argv[]){
  boost::mpi::environment env(argc, argv);
  Kokkos::ScopeGuard KokkosScopeGuard;
  Kokkos::DefaultExecutionSpace::print_configuration(std::cout);

  param.set("periodic_x", "true");
  param.set("periodic_y", "true");
  param.set("periodic_z", "true" );
  param.set("px", "1");
  param.set("py", "1");
  param.set("pz", "1");
  cartcomm=Corduroy(globalcomm, 1, 1, 1);

       grid=Greige(-3.0,-4.5,-2.8, 0.1, 0.15, 0.12);
  localgrid=Greige(-3.0,-4.5,-2.8, 0.1, 0.15, 0.12);

  double pi=4.0*atan(1.0);

  int numparcels=42;
  Lint<DustTest> Parcels;
  typedef Kokkos::Random_XorShift64_Pool<> GeneratorPool;
  GeneratorPool pool(511);
  float killfraction=0.13;

  for(int m=0; m<numparcels; m++) {
    DustTest tracers(0+DustTest::NDUST*m);
    DustTest::ScalarPointType P, W;
    P=tracers.ScalarPointVariables.find("P")->second;
    W=tracers.ScalarPointVariables.find("W")->second;

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
        W(n)=  1.0;//Weight
        pool.free_state(gen);
    } );
    Parcels.push_back(tracers);
  }

  printf("Spigot Control Test originally %4zu parcels with fill percentages = ", Parcels.size());
  for(DustTest P: Parcels) printf("%9.6f ", float(P.getcount(DustTest::HEALTHY))/DustTest::NDUST);
  printf("\n");
  Parcels.write_silo("BeforeSpigot");

  /* ---------------------------------------------------------------------- */
  const size_t NP=NX*NY*NZ;
  const size_t NG=(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH);
  double expG;
  {
  Yarn::ScalarFieldType F("F", NG);
  Yarn::ScalarFieldType G("G", NP);
  for(DustTest P: Parcels)
      Goethite<NX,NY,NZ,NH>::deposit(F, P.loc, P.state, P.ScalarPointVariables.find("W")->second);
  Tuck<NX,NY,NZ,NH>::unfill_ghost(F,G);

  double minG, maxG, sumG;
  expG=0.0;
  for(DustTest P: Parcels) expG+=P.getcount(DustTest::HEALTHY);
  Kokkos::parallel_reduce(NP, calcmin(G), minG);
  Kokkos::parallel_reduce(NP, calcmax(G), maxG);
  Kokkos::parallel_reduce(NP,  KOKKOS_LAMBDA(const size_t& n, double& lsum) { lsum+=G(n); }, sumG);

  printf("Expected deposit: %12.5e Actual sum %12.5e \n", expG, sumG);
  if(fabs(sumG-expG)>1e-8*expG) return(1);
  expG/=double(NX*NY*NZ);
  printf("Expected per point: %12.5e Actual min %12.5e max %12.5e\n", expG, minG, maxG);
  }

  /* ---------------------------------------------------------------------- */
  Lint<DustTest> newParcels = Spigot<DustTest, NX, NY, NZ, NH>::extract2replicate(pool, Parcels, "W", expG);
  for(DustTest P: newParcels) Parcels.push_back(P);

  printf("Spigot Control Test after      %4zu parcels with fill percentages = ", Parcels.size());
  for(DustTest P: Parcels) printf("%9.6f ", float(P.getcount(DustTest::HEALTHY))/DustTest::NDUST);
  printf("\n");

  /* ---------------------------------------------------------------------- */
  {
  const size_t NP=NX*NY*NZ;
  const size_t NG=(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH);
  Yarn::ScalarFieldType F("F", NG);
  Yarn::ScalarFieldType G("G", NP);
  for(DustTest P: Parcels)
      Goethite<NX,NY,NZ,NH>::deposit(F, P.loc, P.state, P.ScalarPointVariables.find("W")->second);
  Tuck<NX,NY,NZ,NH>::unfill_ghost(F,G);

  double minG, maxG, sumG;
  expG=0.0;
  for(DustTest P: Parcels) expG+=P.getcount(DustTest::HEALTHY);
  Kokkos::parallel_reduce(NP, calcmin(G), minG);
  Kokkos::parallel_reduce(NP, calcmax(G), maxG);
  Kokkos::parallel_reduce(NP,  KOKKOS_LAMBDA(const size_t& n, double& lsum) { lsum+=G(n); }, sumG);

  printf("Expected deposit: %12.5e Actual sum %12.5e \n", expG, sumG);
  if(fabs(sumG-expG)>1e-8*expG) return(1);
  expG/=double(NX*NY*NZ);
  printf("Expected per point: %12.5e Actual min %12.5e max %12.5e\n", expG, minG, maxG);
  }

  return(0);
}
