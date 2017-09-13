#include "Dust.h"

/* ---------------------------------------------------------------------- */
Dust::Dust(uint64_t ssn_start) {
    age   = Kokkos::View<double    [NDUST]    > ("age"  );
    dob   = Kokkos::View<double    [NDUST]    > ("dob"  );
    ssn   = Kokkos::View<uint64_t  [NDUST]    > ("ssn"  );
    state = Kokkos::View<st_type   [NDUST]    > ("state");
    loc   = Kokkos::View<double    [NDUST][3] > ("loc"  );
    init_ssn(ssn_start);
}

/* ---------------------------------------------------------------------- */
void Dust::init_ssn(uint64_t ssn_start) {
    Kokkos::parallel_for(NDUST, KOKKOS_LAMBDA (const size_t n) {
      ssn(n) = ssn_start+n;
    } );
}
