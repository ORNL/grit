#ifndef DUST_H
#define DUST_H

#include <cstdint>
#include <map>
#include <silo.h>
#include <boost/serialization/map.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_free.hpp>
#include <Kokkos_Core.hpp>
#include "GlobalVariables.h"

class Dust {
  public:
    static const size_t NDUST=1024;
    enum STATE : uint32_t;
    using st_type = typename std::underlying_type<STATE>::type;

    typedef Kokkos::View<double   [NDUST]    > ScalarPointType;
    typedef Kokkos::View<double   [NDUST][3] > Vectr3PointType;
    typedef Kokkos::View<st_type  [NDUST]    > HealthPointType;
    typedef Kokkos::View<uint64_t [NDUST]    > SSNumbPointType;
    typedef Vectr3PointType LocatnPointType;
    typedef Vectr3PointType LocationVecType;
    typedef HealthPointType PointHealthType;

    SSNumbPointType ssn  ;
    HealthPointType state;
    LocatnPointType loc  ;

    std::map<std::string, ScalarPointType> ScalarPointVariables;
    std::map<std::string, Vectr3PointType> Vectr3PointVariables;

    enum STATE : uint32_t {
       UNOCCUPIED       =  0, //
       HEALTHY          =  1, //
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
      ssn   = SSNumbPointType ("ssn"  );
      state = HealthPointType ("state"); //Kokkos initializes to 0 Unoccupied
      loc   = LocatnPointType ("loc"  );
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
    int getcount (STATE s=HEALTHY) {
      int nfilled;
      size_t ND=NDUST;
      HealthPointType state_=state;
      Kokkos::parallel_reduce(ND, KOKKOS_LAMBDA (const size_t& n, int& count) {
          if(state_(n)==s) count++;
      }, nfilled);
      return(nfilled);
    }
    /* ---------------------------------------------------------------------- */
    void kill_unhealthy() {
      size_t ND=NDUST;
      HealthPointType state_=state;
      Kokkos::parallel_for(ND, KOKKOS_LAMBDA (const size_t& n) {
          if(state_(n)!=HEALTHY) state_(n)=UNOCCUPIED;
      } );
    }
    /* ---------------------------------------------------------------------- */
    void write_silo(std::string prefix="Dust") const {
      char filename[1000];
      sprintf(filename, "%s.silo", prefix.c_str());
      DBfile *file=nullptr;
      file=DBCreate(filename, DB_CLOBBER, DB_LOCAL, NULL, DB_HDF5);
      write_silo_mesh(file);
      DBClose(file);
      return;
    }
    /* ---------------------------------------------------------------------- */
    void write_silo_mesh(DBfile *file, std::string prefix="Points") const {
      std::vector<double> xvVec, yvVec, zvVec;
      get_vertex_coordinates(xvVec, yvVec, zvVec);
      double* coords[3]={xvVec.data(), yvVec.data(), zvVec.data()};
      char dirname[1000];
      sprintf(dirname, "%s", prefix.c_str());
      DBMkDir (file, dirname);
      DBSetDir(file, dirname);
      char meshname[1000];
      sprintf(meshname, "%s", prefix.c_str());
      DBPutPointmesh(file, meshname, 3, coords, NDUST, DB_DOUBLE, NULL);

      SSNumbPointType::HostMirror   ssnHost = Kokkos::create_mirror_view(  ssn);
      HealthPointType::HostMirror stateHost = Kokkos::create_mirror_view(state);
      Kokkos::deep_copy(   ssnHost,   ssn);
      Kokkos::deep_copy( stateHost, state);

      DBPutPointvar1 (file,   "ssn", meshname,   ssnHost.data(), NDUST, DB_LONG_LONG, NULL);
      DBPutPointvar1 (file, "state", meshname, stateHost.data(), NDUST, DB_INT   , NULL);

      for(auto it : ScalarPointVariables) {
        ScalarPointType v=it.second;
        ScalarPointType::HostMirror vHost = Kokkos::create_mirror_view(v);
        Kokkos::deep_copy(vHost, v);
        DBPutPointvar1 (file, it.first.c_str(), meshname, vHost.data(), NDUST, DB_DOUBLE, NULL);
      }

      for(auto it : Vectr3PointVariables) {
        Vectr3PointType w=it.second;
        ScalarPointType v("v");
        for(int m=0; m<3; m++) {
        Kokkos::parallel_for(NDUST, KOKKOS_LAMBDA(const size_t &n) { v(n)=w(n,m); } );
        ScalarPointType::HostMirror vHost = Kokkos::create_mirror_view(v);
        Kokkos::deep_copy(vHost, v);
        DBPutPointvar1 (file, (it.first+"vec"+char('0'+m)).c_str(), meshname, vHost.data(), NDUST, DB_DOUBLE, NULL);
        }
      }

      DBSetDir(file, "/");
      return;
    }
    /* ---------------------------------------------------------------------- */
    void get_vertex_coordinates(std::vector<double> &xvVec,
                                std::vector<double> &yvVec,
                                std::vector<double> &zvVec) const {
      xvVec.resize(NDUST);
      yvVec.resize(NDUST);
      zvVec.resize(NDUST);

      Vectr3PointType::HostMirror locHost =Kokkos::create_mirror_view(loc);
      Kokkos::deep_copy(locHost, loc);

      for(int n=0; n<NDUST; n++) {
        xvVec[n]=localgrid.x(locHost(n,0));
        yvVec[n]=localgrid.y(locHost(n,1));
        zvVec[n]=localgrid.z(locHost(n,2));
      }
      return;
    }
    /* ---------------------------------------------------------------------- */
  private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
      ar & ssn;
      ar & state;
      ar & loc;
      //ar & ScalarPointVariables;
      //ar & Vectr3PointVariables;
    }
};

namespace boost { namespace serialization {
    /* ---------------------------------------------------------------------- */
    template<class Archive>
    void save(Archive &ar, Dust::SSNumbPointType v, const unsigned int version) {
      Dust::SSNumbPointType::HostMirror vHost = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(vHost, v);
      ar & boost::serialization::make_array(vHost.ptr_on_device(), Dust::NDUST);
    }
    template<class Archive>
    void load(Archive &ar, Dust::SSNumbPointType v, const unsigned int version) {
      Dust::SSNumbPointType::HostMirror vHost = Kokkos::create_mirror_view(v);
      ar & boost::serialization::make_array(vHost.ptr_on_device(), Dust::NDUST);
      Kokkos::deep_copy(v, vHost);
    }
    /* ---------------------------------------------------------------------- */
    template<class Archive>
    void save(Archive &ar, Dust::HealthPointType v, const unsigned int version) {
      Dust::HealthPointType::HostMirror vHost = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(vHost, v);
      ar & boost::serialization::make_array(vHost.ptr_on_device(), Dust::NDUST);
    }
    template<class Archive>
    void load(Archive &ar, Dust::HealthPointType v, const unsigned int version) {
      Dust::HealthPointType::HostMirror vHost = Kokkos::create_mirror_view(v);
      ar & boost::serialization::make_array(vHost.ptr_on_device(), Dust::NDUST);
      Kokkos::deep_copy(v, vHost);
    }
    /* ---------------------------------------------------------------------- */
    template<class Archive>
    void save(Archive &ar, Dust::ScalarPointType v, const unsigned int version) {
      Dust::ScalarPointType::HostMirror vHost = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(vHost, v);
      printf("rsa reached a782 %zu %p\n", v.dimension_0(), v.ptr_on_device());
      ar & boost::serialization::make_array(vHost.ptr_on_device(), Dust::NDUST);
    }
    template<class Archive>
    void load(Archive &ar, Dust::ScalarPointType v, const unsigned int version) {
      Dust::ScalarPointType::HostMirror vHost = Kokkos::create_mirror_view(v);
      printf("rsa reached a247 %zu %p\n", v.dimension_0(), v.ptr_on_device());
      ar & boost::serialization::make_array(vHost.ptr_on_device(), Dust::NDUST);
      Kokkos::deep_copy(v, vHost);
    }
    /* ---------------------------------------------------------------------- */
    template<class Archive>
    void save(Archive &ar, Dust::Vectr3PointType v, const unsigned int version) {
      Dust::Vectr3PointType::HostMirror vHost = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(vHost, v);
      ar & boost::serialization::make_array(vHost.ptr_on_device(), Dust::NDUST*3);
    }
    template<class Archive>
    void load(Archive &ar, Dust::Vectr3PointType v, const unsigned int version) {
      Dust::Vectr3PointType::HostMirror vHost = Kokkos::create_mirror_view(v);
      ar & boost::serialization::make_array(vHost.ptr_on_device(), Dust::NDUST*3);
      Kokkos::deep_copy(v, vHost);
    }
    /* ---------------------------------------------------------------------- */
} }

BOOST_SERIALIZATION_SPLIT_FREE(Dust::SSNumbPointType)
BOOST_SERIALIZATION_SPLIT_FREE(Dust::HealthPointType)
BOOST_SERIALIZATION_SPLIT_FREE(Dust::ScalarPointType)
BOOST_SERIALIZATION_SPLIT_FREE(Dust::Vectr3PointType)

#endif
