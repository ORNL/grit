#ifndef TUCK_H
#define TUCK_H
#include "FDM/Yarn.h"
#include "FDM/Gossamer.h"

#if defined(__CUDACC__)
  #include <thrust/tuple.h>
  namespace TUP=thrust;
#else
  #include <tuple>
  namespace TUP=std;
#endif
template<int NX, int NY, int NZ, int NH=0>
class Tuck {
  private:
    typedef Yarn::ScalarFieldType ScalarFieldType;

  public:
    /* ---------------------------------------------------------------------- */
    static void tuckx(ScalarFieldType Fg, ScalarFieldType Fi) {
      assert(   (NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH) ==  Fg.dimension_0());
      assert(   (NX       )*(NY+1+2*NH)*(NZ+1+2*NH) ==  Fi.dimension_0());
      size_t NT=(NX+2+4*NH)*(NY+1+2*NH)*(NZ+1+2*NH);

      ScalarFieldType Ft("Ft", NT);
      Gossamer<double, NX+1+2*NH, NY+1+2*NH, NZ+1+2*NH, 1+NH,   NH,    0,    0,    0,    0>::fill_ghost(Fg, Ft, true);

      Kokkos::parallel_for((NX     )*(NY+1+2*NH)*(NZ+1+2*NH), KOKKOS_LAMBDA (const size_t& n) {
        int      i,j,k;
        TUP::tie(i,j,k)=Inego<NX       ,NY+1+2*NH,NZ+1+2*NH,      0,      0,      0>(n);
        size_t ni      =Inego<NX+2+4*NH,NY+1+2*NH,NZ+1+2*NH,-1-2*NH,      0,      0>(        i,        j,        k);
        size_t nl      =Inego<NX+2+4*NH,NY+1+2*NH,NZ+1+2*NH,-1-2*NH,      0,      0>(-1-2*NH+i,        j,        k);
        size_t nr      =Inego<NX+2+4*NH,NY+1+2*NH,NZ+1+2*NH,-1-2*NH,      0,      0>( 1+2*NH+i,        j,        k);
        Fi(n)=Ft(ni);
        if(i<  1+NH) Fi(n)+=Ft(nl);
        if(i>=NX-NH) Fi(n)+=Ft(nr);
      } );
    }

    /* ---------------------------------------------------------------------- */
    static void tucky(ScalarFieldType Fg, ScalarFieldType Fi) {
      assert(   (NX       )*(NY+1+2*NH)*(NZ+1+2*NH) ==  Fg.dimension_0());
      assert(   (NX       )*(NY       )*(NZ+1+2*NH) ==  Fi.dimension_0());
      size_t NT=(NX       )*(NY+2+4*NH)*(NZ+1+2*NH);

      ScalarFieldType Ft("Ft", NT);
      Gossamer<double, NX       , NY+1+2*NH, NZ+1+2*NH,    0,    0, 1+NH,   NH,    0,    0>::fill_ghost(Fg, Ft, true);

      Kokkos::parallel_for((NX     )*(NY       )*(NZ+1+2*NH), KOKKOS_LAMBDA (const size_t& n) {
        int      i,j,k;
        TUP::tie(i,j,k)=Inego<NX       ,NY       ,NZ+1+2*NH,      0,      0,      0>(n);
        size_t ni      =Inego<NX       ,NY+2+4*NH,NZ+1+2*NH,      0,-1-2*NH,      0>(        i,        j,        k);
        size_t nl      =Inego<NX       ,NY+2+4*NH,NZ+1+2*NH,      0,-1-2*NH,      0>(        i,-1-2*NH+j,        k);
        size_t nr      =Inego<NX       ,NY+2+4*NH,NZ+1+2*NH,      0,-1-2*NH,      0>(        i, 1+2*NH+j,        k);
        Fi(n)=Ft(ni);
        if(j<  1+NH) Fi(n)+=Ft(nl);
        if(j>=NY-NH) Fi(n)+=Ft(nr);
      } );
    }

    /* ---------------------------------------------------------------------- */
    static void tuckz(ScalarFieldType Fg, ScalarFieldType Fi) {
      assert(   (NX       )*(NY       )*(NZ+1+2*NH) ==  Fg.dimension_0());
      assert(   (NX       )*(NY       )*(NZ       ) ==  Fi.dimension_0());
      size_t NT=(NX       )*(NY       )*(NZ+2+4*NH);

      ScalarFieldType Ft("Ft", NT);
      Gossamer<double, NX       , NY       , NZ+1+2*NH,    0,    0,    0,    0, 1+NH,   NH>::fill_ghost(Fg, Ft, true);

      Kokkos::parallel_for((NX     )*(NY       )*(NZ       ), KOKKOS_LAMBDA (const size_t& n) {
        int      i,j,k;
        TUP::tie(i,j,k)=Inego<NX       ,NY       ,NZ       ,      0,      0,      0>(n);
        size_t ni      =Inego<NX       ,NY       ,NZ+2+4*NH,      0,      0,-1-2*NH>(        i,        j,        k);
        size_t nl      =Inego<NX       ,NY       ,NZ+2+4*NH,      0,      0,-1-2*NH>(        i,        j,-1-2*NH+k);
        size_t nr      =Inego<NX       ,NY       ,NZ+2+4*NH,      0,      0,-1-2*NH>(        i,        j, 1+2*NH+k);
        Fi(n)=Ft(ni);
        if(k<  1+NH) Fi(n)+=Ft(nl);
        if(k>=NZ-NH) Fi(n)+=Ft(nr);
      } );
    }

    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    static void unfill_ghost(ScalarFieldType Fg, ScalarFieldType Fi) {
      assert(   (NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH) == Fg.dimension_0());
      assert(   (NX       )*(NY       )*(NZ       ) == Fi.dimension_0());
      ScalarFieldType Fx("Fx", (NX       )*(NY+1+2*NH)*(NZ+1+2*NH));
      ScalarFieldType Fy("Fy", (NX       )*(NY       )*(NZ+1+2*NH));
      tuckx(Fg,Fx);
      tucky(Fx,Fy);
      tuckz(Fy,Fi);
    }
};
#endif
