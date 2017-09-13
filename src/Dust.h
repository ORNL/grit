#ifndef DUST_H
#define DUST_H

#include <cstdint>
#include <Kokkos_Core.hpp>

class Dust {
  public:
    static const size_t NDUST=1024;
    enum STATE : uint32_t;
    using st_type = typename std::underlying_type<STATE>::type;

    Kokkos::View<double    [NDUST]    > age  ;
    Kokkos::View<double    [NDUST]    > dob  ;
    Kokkos::View<uint64_t  [NDUST]    > ssn  ;
    Kokkos::View<st_type   [NDUST]    > state;
    Kokkos::View<double    [NDUST][3] > loc  ;

    enum STATE : uint32_t {
       HEALTHY          =  0, //
       UNOCCUPIED           , //
       EXIT_LEFT_X_BDRY     , //
       EXIT_RGHT_X_BDRY     , //
       EXIT_LEFT_Y_BDRY     , //
       EXIT_RGHT_Y_BDRY     , //
       EXIT_LEFT_Z_BDRY     , //
       EXIT_RGHT_Z_BDRY     , //
       WENT_TOO_FAR         , // !Went farther than the nearest neighbor
       WENT_LEFT_X          , //
       WENT_RGHT_X          , //
       WENT_LEFT_Y          , //
       WENT_RGHT_Y          , //
       WENT_LEFT_Z          , //
       WENT_RGHT_Z          , //
       DEATH_HOLE           , // !Kill particles that defy logic
       NBR_NOSPACE          , // !Neighbor has no space to receive
       DIVISION_BY_ZERO     , //
       DRIFTED                // !Drifted away from the intended location
    };

  public:
    Dust(uint64_t ssn_start=0);

  protected:
    void init_ssn(uint64_t ssn_start=0);

};

#endif
