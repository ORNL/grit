#ifndef GOETHITE_H
#define GOETHITE_H

#include "FDM/Yarn.h"
#include "FDM/Tuck.h"
#include "Dust.h"

#if defined(__CUDACC__)
  #include <thrust/tuple.h>
  namespace TUP=thrust;
#else
  #include <tuple>
  namespace TUP=std;
#endif

template<int NX, int NY, int NZ, int NH=0>
class Goethite {
  private:
    typedef Yarn::ScalarFieldType ScalarFieldType;

  public:
    /* ---------------------------------------------------------------------- */
    static void deposit(Yarn::StridedScalarFieldType F,
          Dust::LocationVecType loc, Dust::PointHealthType state, Dust::ScalarPointType P){
      assert(F.dimension_0()==(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH));

      Kokkos::parallel_for(Dust::NDUST, KOKKOS_LAMBDA(const size_t& n) { if(state(n)==Dust::HEALTHY) {
          int ix=floor(loc(n,0));
          int jy=floor(loc(n,1));
          int kz=floor(loc(n,2));
          double rx=loc(n,0)-double(ix);
          double ry=loc(n,1)-double(jy);
          double rz=loc(n,2)-double(kz);

          double filtersum=0.0;
          for(int kk=-NH; kk<2+NH; kk++) {
            for(int jj=-NH; jj<2+NH; jj++) {
              for(int ii=-NH; ii<2+NH; ii++) {
                double delx=ii-rx;
                double dely=jj-ry;
                double delz=kk-rz;
                filtersum +=exp(-0.5*(delx*delx+dely*dely+delz*delz));
          } } }
          for(int kk=-NH; kk<2+NH; kk++) {
            for(int jj=-NH; jj<2+NH; jj++) {
              for(int ii=-NH; ii<2+NH; ii++) {
                double delx=ii-rx;
                double dely=jj-ry;
                double delz=kk-rz;
                double filtercoeff=exp(-0.5*(delx*delx+dely*dely+delz*delz) )/filtersum;
                size_t nn   =   Inego<NX+1+2*NH, NY+1+2*NH, NZ+1+2*NH, -NH, -NH, -NH>(ix+ii,jy+jj,kz+kk) ;
                Kokkos::atomic_add(&F(nn), filtercoeff*P(n));
          } } }
      } } );
    }

    /* ---------------------------------------------------------------------- */
    static void undeposit(Yarn::StridedScalarFieldType F,
          Dust::LocationVecType loc, Dust::PointHealthType state, Dust::ScalarPointType P){
      assert(F.dimension_0()==(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH));
      const size_t NP=NX*NY*NZ;
      const size_t NG=(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH);

      Dust::ScalarPointType I("I");
      Yarn::ScalarFieldType G("G", NG);
      Yarn::ScalarFieldType T("T", NP);
      Yarn::ScalarFieldType W("W", NG);
      Kokkos::parallel_for(Dust::NDUST, KOKKOS_LAMBDA(const size_t& n) { I(n)=1.0; } );
      deposit(G, loc, state, I);
      Tuck    <NX,NY,NZ,NH>::unfill_ghost(G,T);
      Gossamer<double,NX,NY,NZ,NH,1+NH,NH,1+NH,NH,1+NH>::fill_ghost(T, W, true);
      undepositW(F, W, loc, state, P);
    }

    /* ---------------------------------------------------------------------- */
    static void undepositW(Yarn::StridedScalarFieldType F, Yarn::ScalarFieldType W,
          Dust::LocationVecType loc, Dust::PointHealthType state, Dust::ScalarPointType P){
      assert(F.dimension_0()==(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH));
      assert(W.dimension_0()==(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH));

      Kokkos::parallel_for(Dust::NDUST, KOKKOS_LAMBDA(const size_t& n) { if(state(n)==Dust::HEALTHY) {
          int ix=floor(loc(n,0));
          int jy=floor(loc(n,1));
          int kz=floor(loc(n,2));
          double rx=loc(n,0)-double(ix);
          double ry=loc(n,1)-double(jy);
          double rz=loc(n,2)-double(kz);

          double filtersum=0.0;
          for(int kk=-NH; kk<2+NH; kk++) {
            for(int jj=-NH; jj<2+NH; jj++) {
              for(int ii=-NH; ii<2+NH; ii++) {
                double delx=ii-rx;
                double dely=jj-ry;
                double delz=kk-rz;
                filtersum +=exp(-0.5*(delx*delx+dely*dely+delz*delz));
          } } }
          P(n)=0.0;
          for(int kk=-NH; kk<2+NH; kk++) {
            for(int jj=-NH; jj<2+NH; jj++) {
              for(int ii=-NH; ii<2+NH; ii++) {
                double delx=ii-rx;
                double dely=jj-ry;
                double delz=kk-rz;
                double filtercoeff=exp(-0.5*(delx*delx+dely*dely+delz*delz))/filtersum;
                size_t nn   =   Inego<NX+1+2*NH, NY+1+2*NH, NZ+1+2*NH, -NH, -NH, -NH>(ix+ii,jy+jj,kz+kk) ;
                P(n)+=filtercoeff*F(nn)/W(nn);
          } } }
      } } );
    }
    /* ---------------------------------------------------------------------- */
  };
#endif

