#include "Dust.h"

/* ---------------------------------------------------------------------- */
Dust::Dust(uint64_t ssn_start) {
    age   = Kokkos::View<double    [NDUST]    > ("age"  );
    dob   = Kokkos::View<double    [NDUST]    > ("dob"  );
    ssn   = Kokkos::View<uint64_t  [NDUST]    > ("ssn"  );
    pob   = Kokkos::View<uint32_t  [NDUST]    > ("pob"  );
    state = Kokkos::View<st_type   [NDUST]    > ("state");
    loc   = Kokkos::View<double    [NDUST][3] > ("loc"  );

    Kokkos::parallel_for(NDUST, KOKKOS_LAMBDA (const size_t n) {
        age  (n) = 0.0;
        dob  (n) = 0.0; //fix
        ssn  (n) = ssn_start+n;
        pob  (n) = 0;
        state(n) = HEALTHY;
        loc(n,0) = loc(n,1) = loc(n,2) = 0.0;
    } );
}
