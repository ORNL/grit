#ifndef INEGO_H
#define INEGO_H

#include <assert.h>
#include <Kokkos_Core.hpp>

#if defined(__CUDACC__)
  #include <thrust/tuple.h>
  namespace TUP=thrust;
#else
  #include <tuple>
  namespace TUP=std;
#endif

/* ---------------------------------------------------------------------- */
// Provides a fortran style indexing operator back and forth from 3D to 1D.
// The left most index varies the fastest.
// The starting index is (SX,SY,SZ)
// The ending index is (SX+NX-1, SY+NY-1, SZ+NZ-1)

// To go through (NX,NY,NZ) in fortran order, the index conversion is:
// (i,j,k) -> n  is given by n=k*NX*NY+j*NX+i
// n -> (i,j,k) is given by k=n/NX/NY j=(n-k*NX*NY)/NX i=(n-k*NX*NY)%NX
/* ---------------------------------------------------------------------- */
template<int NX  , int NY  , int NZ,
         int SX=0, int SY=0, int SZ=0>
KOKKOS_FUNCTION
static size_t Inego (const int i, const int j, const int k) {
  assert(i>=SX && i<SX+NX);
  assert(j>=SY && j<SY+NY);
  assert(k>=SZ && k<SZ+NZ);
  return (k-SZ)*NX*NY+(j-SY)*NX+i-SX;
}

template<int NX  , int NY  , int NZ,
         int SX=0, int SY=0, int SZ=0>
KOKKOS_FUNCTION
static TUP::tuple<int, int, int> Inego (const size_t n) {
  assert(n<NX*NY*NZ);
  int k=n/NX/NY;
  int j=(n-k*NX*NY)/NX;
  int i=(n-k*NX*NY)%NX;
  k+=SZ;
  j+=SY;
  i+=SX;
  return TUP::make_tuple(i,j,k);
}

#endif
