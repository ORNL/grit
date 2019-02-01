#ifndef YARN_H
#define YARN_H

#include <Kokkos_Core.hpp>
#include "Pique.h"

class Yarn {
  private:
  public:
    typedef Kokkos::DefaultExecutionSpace::size_type size_type; 

    typedef Kokkos::View<double *>      ScalarFieldType;
    typedef Kokkos::View<double **>     VectorFieldType;
    typedef Kokkos::View<double ***>    TensorFieldType;

    typedef Kokkos::View<double *, Kokkos::LayoutStride> StridedScalarFieldType;

    typedef Kokkos::View<const double *>      ConstScalarFieldType;
    typedef Kokkos::View<const double **>     ConstVectorFieldType;
    typedef Kokkos::View<const double ***>    ConstTensorFieldType;

    //----------------------------------------
    static void NativeArrayToScalarField(int NP, double *A, ScalarFieldType F){
      FortranArrayToScalarField(NP, A, F);
    }
    //----------------------------------------
    static void ScalarFieldToNativeArray(int NP, double *A, ScalarFieldType F){
      ScalarFieldToFortranArray(NP, A, F);
    }
    //----------------------------------------
    static void FortranArrayToScalarField(int NP, double *A, ScalarFieldType F){
      ScalarFieldType::HostMirror HostF=Kokkos::create_mirror_view(F);
      for(int i=0; i<NP; i++){
        HostF(i)=A[i];
      }
      Kokkos::deep_copy(F, HostF);
    }
    //----------------------------------------
    static void ScalarFieldToFortranArray(int NP, double *A, ScalarFieldType F){
      ScalarFieldType::HostMirror HostF=Kokkos::create_mirror_view(F);
      Kokkos::deep_copy(HostF, F); 
      for(int i=0; i<NP; i++){
        A[i]=HostF(i); 
      }
    }
    //----------------------------------------
    static void NativeArrayToVectorField(int NP, int NV, double *A, VectorFieldType F){
      Pique<double> Aq(A, NV, NP); //Aq gives a fortran array view of A
      VectorFieldType::HostMirror HostF=Kokkos::create_mirror_view(F);
      for(int i=0; i<NP; i++){
        for(int j=0; j<NV; j++){
          HostF(i,j)=Aq(j,i);
        }
      }
      Kokkos::deep_copy(F, HostF);
    }
    //----------------------------------------
    static void VectorFieldToNativeArray(int NP, int NV, double *A, VectorFieldType F){
      Pique<double> Aq(A, NV, NP); //Aq gives a fortran array view of A
      VectorFieldType::HostMirror HostF=Kokkos::create_mirror_view(F);
      Kokkos::deep_copy(HostF, F); 
      for(int i=0; i<NP; i++){
        for(int j=0; j<NV; j++){
          Aq(j,i)=HostF(i,j);
        }
      }
    }
    //----------------------------------------
    static void FortranArrayToVectorField(int NP, int NV, double *A, VectorFieldType F){
      Pique<double> Aq(A, NP, NV);
      VectorFieldType::HostMirror HostF=Kokkos::create_mirror_view(F);
      for(int j=0; j<NV; j++){
        for(int i=0; i<NP; i++){
          HostF(i,j)=Aq(i,j);
        }
      }
      Kokkos::deep_copy(F, HostF);
    }
    //----------------------------------------
    static void VectorFieldToFortranArray(int NP, int NV, double *A, VectorFieldType F){
      Pique<double> Aq(A, NP, NV);
      VectorFieldType::HostMirror HostF=Kokkos::create_mirror_view(F);
      Kokkos::deep_copy(HostF, F);
      for(int j=0; j<NV; j++){
        for(int i=0; i<NP; i++){
          Aq(i,j)=HostF(i,j);
        }
      }
    }
};

#endif
