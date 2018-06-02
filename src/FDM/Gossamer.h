#ifndef GOSSAMER_H
#define GOSSAMER_H

#include <Kokkos_Core.hpp>
#include "FDM/Inego.h"
#include "GlobalVariables.h"

#if defined(__CUDACC__)
  #include <thrust/tuple.h>
  namespace TUP=thrust;
#else
  #include <tuple>
  namespace TUP=std;
#endif

template<typename T,
    int NX, int NY, int NZ, int GXL, int GXR, int GYL, int GYR, int GZL, int GZR>
class Gossamer {
  public:
    typedef Kokkos::View<T *> ScalarFieldType;
  public:
    /* ---------------------------------------------------------------------- */
    static void fill_intrr(ScalarFieldType Fi, ScalarFieldType Fg){
      Kokkos::parallel_for(NX*NY*NZ, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX        , NY        , NZ        ,    0,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, j, k);
        Fg(ng) = Fi(n);
      } );
    }
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    template<class S=void>
    static typename std::enable_if<(GXL==0), S>::type fill_xleft(ScalarFieldType Fg, const bool fill, const T value){ }

    template<class S=void>
    static typename std::enable_if<(GXL!=0), S>::type fill_xleft(ScalarFieldType Fg, const bool fill, const T value){
      size_t NGHOST=(   GXL    )*(NY        )*(NZ        );
      ScalarFieldType grsend("grsend", NGHOST);
      ScalarFieldType glrecv("glrecv", NGHOST);
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<   GXL    , NY        , NZ        ,    0,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(NX-GXL+i, j, k);
        grsend(n) = Fg(ng);
      } );
      int rank=cartcomm.x.rank();
      int size=cartcomm.x.size();
      if(size==1){
        glrecv=grsend;
      } else {
        int lid = (rank-1+size)%size;
        int rid = (rank+1+size)%size;
        boost::mpi::request reqs[2];
        typename ScalarFieldType::HostMirror Hostgrsend=Kokkos::create_mirror_view(grsend);
        typename ScalarFieldType::HostMirror Hostglrecv=Kokkos::create_mirror_view(glrecv);
        Kokkos::deep_copy(Hostgrsend, grsend);
        reqs[0]=cartcomm.x.isend(rid, 61, (T*) Hostgrsend.ptr_on_device(), NGHOST);
        reqs[1]=cartcomm.x.irecv(lid, 61, (T*) Hostglrecv.ptr_on_device(), NGHOST);
        boost::mpi::wait_all(reqs, reqs+2);
        Kokkos::deep_copy(glrecv,Hostglrecv);
      };
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<   GXL    , NY        , NZ        ,    0,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>( -GXL+i, j, k);
        Fg(ng)=glrecv(n);
      } );
      if(rank!=0) return;
      if(!param.present("periodic_x") || param.getbool("periodic_x")) return;
      if(fill){
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<   GXL    , NY        , NZ        ,    0,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>( -GXL+i, j, k);
        Fg(ng)=value;
      } );
      return;
      };
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<   GXL    , NY        , NZ        ,    0,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>( -GXL+i, j, k);
        size_t ni         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(      0, j, k);
        Fg(ng)=Fg(ni);
      } );
    }
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    template<class S=void>
    static typename std::enable_if<(GXR==0), S>::type fill_xrght(ScalarFieldType Fg, const bool fill, const T value){ }

    template<class S=void>
    static typename std::enable_if<(GXR!=0), S>::type fill_xrght(ScalarFieldType Fg, const bool fill, const T value){
      size_t NGHOST=(       GXR)*(NY        )*(NZ        );
      ScalarFieldType glsend("glsend", NGHOST);
      ScalarFieldType grrecv("grrecv", NGHOST);
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<       GXR, NY        , NZ        ,    0,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(       i, j, k);
        glsend(n) = Fg(ng);
      } );
      int rank=cartcomm.x.rank();
      int size=cartcomm.x.size();
      if(size==1){
        grrecv=glsend;
      } else {
        int lid = (rank-1+size)%size;
        int rid = (rank+1+size)%size;
        boost::mpi::request reqs[2];
        typename ScalarFieldType::HostMirror Hostglsend=Kokkos::create_mirror_view(glsend);
        typename ScalarFieldType::HostMirror Hostgrrecv=Kokkos::create_mirror_view(grrecv);
        Kokkos::deep_copy(Hostglsend, glsend);
        reqs[0]=cartcomm.x.isend(lid, 62, (T*) Hostglsend.ptr_on_device(), NGHOST);
        reqs[1]=cartcomm.x.irecv(rid, 62, (T*) Hostgrrecv.ptr_on_device(), NGHOST);
        boost::mpi::wait_all(reqs, reqs+2);
        Kokkos::deep_copy(grrecv,Hostgrrecv);
      };
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<       GXR, NY        , NZ        ,    0,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(NX+    i, j, k);
        Fg(ng)=grrecv(n);
      } );
      if(rank!=size-1) return;
      if(!param.present("periodic_x") || param.getbool("periodic_x")) return;
      if(fill){
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<       GXR, NY        , NZ        ,    0,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(NX+    i, j, k);
        Fg(ng)=value;
      } );
      return;
      };
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<       GXR, NY        , NZ        ,    0,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(NX+    i, j, k);
        size_t ni         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(NX-    1, j, k);
        Fg(ng)=Fg(ni);
      } );
    }
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    template<class S=void>
    static typename std::enable_if<(GYL==0), S>::type fill_yleft(ScalarFieldType Fg, const bool fill, const T value){ }

    template<class S=void>
    static typename std::enable_if<(GYL!=0), S>::type fill_yleft(ScalarFieldType Fg, const bool fill, const T value){
      size_t NGHOST=(NX+GXL+GXR)*(   GYL    )*(NZ        );
      ScalarFieldType grsend("grsend", NGHOST);
      ScalarFieldType glrecv("glrecv", NGHOST);
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR,    GYL    , NZ        , -GXL,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, NY-GYL+j, k);
        grsend(n) = Fg(ng);
      } );
      int rank=cartcomm.y.rank();
      int size=cartcomm.y.size();
      if(size==1){
        glrecv=grsend;
      } else {
        int lid = (rank-1+size)%size;
        int rid = (rank+1+size)%size;
        boost::mpi::request reqs[2];
        typename ScalarFieldType::HostMirror Hostgrsend=Kokkos::create_mirror_view(grsend);
        typename ScalarFieldType::HostMirror Hostglrecv=Kokkos::create_mirror_view(glrecv);
        Kokkos::deep_copy(Hostgrsend, grsend);
        reqs[0]=cartcomm.y.isend(rid, 63, (T*) Hostgrsend.ptr_on_device(), NGHOST);
        reqs[1]=cartcomm.y.irecv(lid, 63, (T*) Hostglrecv.ptr_on_device(), NGHOST);
        boost::mpi::wait_all(reqs, reqs+2);
        Kokkos::deep_copy(glrecv,Hostglrecv);
      };
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR,    GYL    , NZ        , -GXL,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, -GYL+j, k);
        Fg(ng)=glrecv(n);
      } );
      if(rank!=0) return;
      if(!param.present("periodic_y") || param.getbool("periodic_y")) return;
      if(fill){
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR,    GYL    , NZ        , -GXL,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, -GYL+j, k);
        Fg(ng)=value;
      } );
      return;
      };
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR,    GYL    , NZ        , -GXL,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, -GYL+j, k);
        size_t ni         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i,      0, k);
        Fg(ng)=Fg(ni);
      } );
    }
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    template<class S=void>
    static typename std::enable_if<(GYR==0), S>::type fill_yrght(ScalarFieldType Fg, const bool fill, const T value){ }

    template<class S=void>
    static typename std::enable_if<(GYR!=0), S>::type fill_yrght(ScalarFieldType Fg, const bool fill, const T value){
      size_t NGHOST=(NX+GXL+GXR)*(   GYR    )*(NZ        );
      ScalarFieldType glsend("glsend", NGHOST);
      ScalarFieldType grrecv("grrecv", NGHOST);
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR,        GYR, NZ        , -GXL,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i,        j, k);
        glsend(n) = Fg(ng);
      } );
      int rank=cartcomm.y.rank();
      int size=cartcomm.y.size();
      if(size==1){
        grrecv=glsend;
      } else {
        int lid = (rank-1+size)%size;
        int rid = (rank+1+size)%size;
        boost::mpi::request reqs[2];
        typename ScalarFieldType::HostMirror Hostglsend=Kokkos::create_mirror_view(glsend);
        typename ScalarFieldType::HostMirror Hostgrrecv=Kokkos::create_mirror_view(grrecv);
        Kokkos::deep_copy(Hostglsend, glsend);
        reqs[0]=cartcomm.y.isend(lid, 64, (T*) Hostglsend.ptr_on_device(), NGHOST);
        reqs[1]=cartcomm.y.irecv(rid, 64, (T*) Hostgrrecv.ptr_on_device(), NGHOST);
        boost::mpi::wait_all(reqs, reqs+2);
        Kokkos::deep_copy(grrecv,Hostgrrecv);
      };
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR,        GYR, NZ        , -GXL,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, NY+    j, k);
        Fg(ng)=grrecv(n);
      } );
      if(rank!=size-1) return;
      if(!param.present("periodic_y") || param.getbool("periodic_y")) return;
      if(fill){
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR,        GYR, NZ        , -GXL,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, NY+    j, k);
        Fg(ng)=value;
      } );
      return;
      };
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR,        GYR, NZ        , -GXL,    0,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, NY+    j, k);
        size_t ni         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, NY-    1, k);
        Fg(ng)=Fg(ni);
      } );
    }
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    template<class S=void>
    static typename std::enable_if<(GZL==0), S>::type fill_zleft(ScalarFieldType Fg, const bool fill, const T value){ }

    template<class S=void>
    static typename std::enable_if<(GZL!=0), S>::type fill_zleft(ScalarFieldType Fg, const bool fill, const T value){
      size_t NGHOST=(NX+GXL+GXR)*(NY+GYL+GYR)*(   GZL    );
      ScalarFieldType grsend("grsend", NGHOST);
      ScalarFieldType glrecv("glrecv", NGHOST);
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR, NY+GYL+GYR,    GZL    , -GXL, -GYL,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, j, NZ-GZL+k);
        grsend(n) = Fg(ng);
      } );
      int rank=cartcomm.z.rank();
      int size=cartcomm.z.size();
      if(size==1){
        glrecv=grsend;
      } else {
        int lid = (rank-1+size)%size;
        int rid = (rank+1+size)%size;
        boost::mpi::request reqs[2];
        typename ScalarFieldType::HostMirror Hostgrsend=Kokkos::create_mirror_view(grsend);
        typename ScalarFieldType::HostMirror Hostglrecv=Kokkos::create_mirror_view(glrecv);
        Kokkos::deep_copy(Hostgrsend, grsend);
        reqs[0]=cartcomm.z.isend(rid, 65, (T*) Hostgrsend.ptr_on_device(), NGHOST);
        reqs[1]=cartcomm.z.irecv(lid, 65, (T*) Hostglrecv.ptr_on_device(), NGHOST);
        boost::mpi::wait_all(reqs, reqs+2);
        Kokkos::deep_copy(glrecv,Hostglrecv);
      };
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR, NY+GYL+GYR,    GZL    , -GXL, -GYL,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, j,   -GZL+k);
        Fg(ng)=glrecv(n);
      } );
      if(rank!=0) return;
      if(!param.present("periodic_z") || param.getbool("periodic_z")) return;
      if(fill){
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR, NY+GYL+GYR,    GZL    , -GXL, -GYL,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, j,   -GZL+k);
        Fg(ng)=value;
      } );
      return;
      };
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR, NY+GYL+GYR,    GZL    , -GXL, -GYL,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, j,   -GZL+k);
        size_t ni         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, j,        0);
        Fg(ng)=Fg(ni);
      } );
    }
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    template<class S=void>
    static typename std::enable_if<(GZR==0), S>::type fill_zrght(ScalarFieldType Fg, const bool fill, const T value){ }

    template<class S=void>
    static typename std::enable_if<(GZR!=0), S>::type fill_zrght(ScalarFieldType Fg, const bool fill, const T value){
      size_t NGHOST=(NX+GXL+GXR)*(NY+GYL+GYR)*(       GZR);
      ScalarFieldType glsend("glsend", NGHOST);
      ScalarFieldType grrecv("grrecv", NGHOST);
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR, NY+GYL+GYR,        GZR, -GXL, -GYL,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, j,        k);
        glsend(n) = Fg(ng);
      } );
      int rank=cartcomm.z.rank();
      int size=cartcomm.z.size();
      if(size==1){
        grrecv=glsend;
      } else {
        int lid = (rank-1+size)%size;
        int rid = (rank+1+size)%size;
        boost::mpi::request reqs[2];
        typename ScalarFieldType::HostMirror Hostglsend=Kokkos::create_mirror_view(glsend);
        typename ScalarFieldType::HostMirror Hostgrrecv=Kokkos::create_mirror_view(grrecv);
        Kokkos::deep_copy(Hostglsend, glsend);
        reqs[0]=cartcomm.z.isend(lid, 66, (T*) Hostglsend.ptr_on_device(), NGHOST);
        reqs[1]=cartcomm.z.irecv(rid, 66, (T*) Hostgrrecv.ptr_on_device(), NGHOST);
        boost::mpi::wait_all(reqs, reqs+2);
        Kokkos::deep_copy(grrecv,Hostgrrecv);
      };
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR, NY+GYL+GYR,        GZR, -GXL, -GYL,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, j, NZ+    k);
        Fg(ng)=grrecv(n);
      } );
      if(rank!=size-1) return;
      if(!param.present("periodic_z") || param.getbool("periodic_z")) return;
      if(fill){
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR, NY+GYL+GYR,        GZR, -GXL, -GYL,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, j,NZ+     k);
        Fg(ng)=value;
      } );
      return;
      };
      Kokkos::parallel_for(NGHOST, KOKKOS_LAMBDA (const size_t & n) {
        int      i, j, k;
        TUP::tie(i, j, k) = Inego<NX+GXL+GXR, NY+GYL+GYR,        GZR, -GXL, -GYL,    0>(n);
        size_t ng         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, j,NZ+     k);
        size_t ni         = Inego<NX+GXL+GXR, NY+GYL+GYR, NZ+GZL+GZR, -GXL, -GYL, -GZL>(i, j,NZ-     1);
        Fg(ng)=Fg(ni);
      } );
    }
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
  public:
    static void fill_ghost(ScalarFieldType Fi, ScalarFieldType Fg, const bool fill=false, const T value=0){
      fill_intrr(Fi, Fg);
      if(GXL>0) fill_xleft(Fg, fill, value);
      if(GXR>0) fill_xrght(Fg, fill, value);
      if(GYL>0) fill_yleft(Fg, fill, value);
      if(GYR>0) fill_yrght(Fg, fill, value);
      if(GZL>0) fill_zleft(Fg, fill, value);
      if(GZR>0) fill_zrght(Fg, fill, value);
    }
};
#endif
