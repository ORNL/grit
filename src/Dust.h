#ifndef DUST_H
#define DUST_H

#include <cstdint>
#include <Kokkos_Core.hpp>

class Dust {
  public:
    static const size_t NDUST=1024;
    enum STATE : uint32_t;
    using st_type = typename std::underlying_type<STATE>::type;

    typedef Kokkos::View<double  [NDUST]    > ScalarPointType;
    typedef Kokkos::View<double  [NDUST][3] > LocationVecType;
    typedef Kokkos::View<st_type [NDUST]    > PointHealthType;

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
       DRIFTED              , // !Drifted away from the intended location
       INVALID                //
    };

  public:
    // constructor
    Dust(uint64_t ssn_start=0) {
      age   = Kokkos::View<double    [NDUST]    > ("age"  );
      dob   = Kokkos::View<double    [NDUST]    > ("dob"  );
      ssn   = Kokkos::View<uint64_t  [NDUST]    > ("ssn"  );
      state = Kokkos::View<st_type   [NDUST]    > ("state");
      loc   = Kokkos::View<double    [NDUST][3] > ("loc"  );
      Kokkos::parallel_for(NDUST, init_ssn(ssn, ssn_start));
      Kokkos::fence();
    }
    /* ---------------------------------------------------------------------- */
    struct init_ssn {
      uint64_t ssn_start;
      Kokkos::View<uint64_t  [NDUST]    > ssn  ;
      //constructor
      init_ssn(Kokkos::View<uint64_t [NDUST]> ssn_, uint64_t ssn_start_)
        : ssn(ssn_), ssn_start(ssn_start_) { };
      KOKKOS_INLINE_FUNCTION
      void operator()(const size_t n) const {
        ssn(n) = ssn_start+n;
      }
    };
    /* ---------------------------------------------------------------------- */
};

#endif
