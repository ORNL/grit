#ifndef SPIGOT_H
#define SPIGOT_H

#include "Dust.h"
#include "Lint.h"

template <typename T, int NX, int NY, int NZ, int NH=0>
class Spigot {
  public:
      typedef Kokkos::Random_XorShift64_Pool<> GeneratorPool;
    /* ---------------------------------------------------------------------- */
    // W is a weight function to be deposited on the grid
    // E is the expected distribution of weight function
    // Kill or extract (to replicate) particles to make the deposited W approach E.
    // After extracting, either kill the source particle or leave as it is (bool kill argument)
    static Lint<T> extract2replicate(GeneratorPool pool, Lint<T> Parcels,
                std::string Wname, Yarn::ScalarFieldType E, bool kill=false){
      assert(E.dimension_0()==NX*NY*NZ);
      const size_t NP=NX*NY*NZ;
      const size_t NG=(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH);
      Yarn::ScalarFieldType H("H", NG);
      Yarn::ScalarFieldType W("W", NG);

      {
      Yarn::ScalarFieldType D("D", NP);
      Yarn::ScalarFieldType F("F", NG);
      Yarn::ScalarFieldType G("G", NP);

      for(T P: Parcels)
          Goethite<NX,NY,NZ,NH>::deposit(F, P.loc, P.state, P.ScalarPointVariables.find(Wname)->second);
      Tuck<NX,NY,NZ,NH>::unfill_ghost(F,G);
      Kokkos::parallel_for(NP, KOKKOS_LAMBDA(const size_t &n) { D(n)=E(n)-G(n); } );
      Gossamer<double,NX,NY,NZ,NH,1+NH,NH,1+NH,NH,1+NH>::fill_ghost(D, H, true);

      Dust::ScalarPointType I("I");
      Kokkos::parallel_for(Dust::NDUST, KOKKOS_LAMBDA(const size_t& n) { I(n)=1.0; } );
      for(T P: Parcels)
          Goethite<NX,NY,NZ,NH>::deposit(F, P.loc, P.state, I);
      Tuck    <NX,NY,NZ,NH>::unfill_ghost(F,G);
      Gossamer<double,NX,NY,NZ,NH,1+NH,NH,1+NH,NH,1+NH>::fill_ghost(G, W, true);
      }

      Dust::ScalarPointType Q("Q");
      for(T P: Parcels) {
        Goethite<NX,NY,NZ,NH>::undepositW(H, W, P.loc, P.state, Q);
        Kokkos::parallel_for(T::NDUST, KOKKOS_LAMBDA(const size_t& n) { if (P.state(n)==T::HEALTHY) {
            GeneratorPool::generator_type gen = pool.get_state();
            float prob=gen.frand(1.0);
            if(prob<fabs(Q(n))) {
              if(Q(n)<0) P.state(n)=T::UNOCCUPIED;
              if(Q(n)>0) P.state(n)=T::EXTRACT2REPLICATE;
            }
        } } );
      }
      Lint<T> newParcels= Parcels.extract(T::EXTRACT2REPLICATE, kill);
      if(!kill) for(T P:Parcels) Kokkos::parallel_for(NP, KOKKOS_LAMBDA(const size_t &n) {
          if(P.state(n)==T::EXTRACT2REPLICATE) P.state(n)=T::HEALTHY;
      } );
      for(T P:newParcels) Kokkos::parallel_for(NP, KOKKOS_LAMBDA(const size_t &n) {
          if(P.state(n)==T::EXTRACT2REPLICATE) P.state(n)=T::HEALTHY;
      } );
      return (newParcels);
    }
    /* ---------------------------------------------------------------------- */
    static Lint<T> extract2replicate(GeneratorPool pool, Lint<T> Parcels,
                std::string Wname, double e, bool kill=false){
      const size_t NP=NX*NY*NZ;
      Yarn::ScalarFieldType E("E", NP);
      Kokkos::parallel_for(NP, KOKKOS_LAMBDA(const size_t &n) { E(n)=e;} );
      return (extract2replicate(pool, Parcels, Wname, E, kill));
    }
    /* ---------------------------------------------------------------------- */
};

#endif
