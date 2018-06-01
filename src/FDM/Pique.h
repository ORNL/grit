#ifndef PIQUE_H
#define PIQUE_H

#include <stdlib.h>
#include <assert.h>

/* ---------------------------------------------------------------------- */
// Provides a fortran style array indexing operator on a pointer
// as if it was dimensioned in fortran as ptr(NX, NY, NZ)
// The indices go in fortran style as ptr(SX:EX, SY:EY, SZ:EZ, SV:EV)
// The left most index varies the fastest
// The default starting index is 0 in C-style unless specified
// No memory management is provided 
/* ---------------------------------------------------------------------- */
template<class T>
class Pique {
private:
  T* ptr;
  int NX, NY, NZ, NV;
  int SX, SY, SZ, SV;
  int EX, EY, EZ, EV;
public:
  /* ---------------------------------------------------------------------- */
  Pique(T* ptr_, int NX_  , int NY_  , int NZ_=1, int NV_=1,
                 int SX_=0, int SY_=0, int SZ_=0, int SV_=0) :
        ptr(ptr_), NX(NX_), NY(NY_), NZ(NZ_), NV(NV_),
                   SX(SX_), SY(SY_), SZ(SZ_), SV(SV_),
                   EX(SX_+NX_-1), EY(SY_+NY_-1), EZ(SZ_+NZ_-1), EV(SV_+NV_-1) { } ;

  /* ---------------------------------------------------------------------- */
  T& operator () (const int i, const int j, const int k=0, const int m=0) const {
    assert(i>=SX && i<=EX);
    assert(j>=SY && j<=EY);
    assert(k>=SZ && k<=EZ);
    assert(m>=SV && m<=EV);
    int i0=i-SX;
    int j0=j-SY;
    int k0=k-SZ;
    int m0=m-SV;
    return (*(ptr+m0*NX*NY*NZ+k0*NX*NY+j0*NX+i0));
  };
};

#endif
