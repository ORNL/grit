#ifndef LINT_H
#define LINT_H

#include <list>
#include <map>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/serialization/list.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include "GlobalVariables.h"
#include "Dust.h"

template <typename T>
class Lint : public std::list<T> {
  public:
    /* ---------------------------------------------------------------------- */
    void compact() {
      if(this->empty()) return;
      typedef std::pair<T, int>  CountedT;
      auto comp = [] (const CountedT &p1, const CountedT &p2)->bool { return p1.second>p2.second; };
      std::list<CountedT> CountedList;
      for(T parcel: *this) {
        CountedList.push_back(std::make_pair(parcel, T::NDUST-parcel.getcount(T::UNOCCUPIED)));
      }
      CountedList.remove_if( [] (const CountedT p1) { return p1.second==0; } );
      if(CountedList.size()<2) {
        this->clear();
        for(auto parcel: CountedList) this->push_back(parcel.first);
        return;
      }
      CountedList.sort(comp);

      while( CountedList.back().second + (*(std::prev(CountedList.end(),2))).second < T::NDUST) {
        auto pr2=CountedList.back();
        auto it1=std::prev(CountedList.end(),2);
        if(CountedList.size()==2) {
          merge_parcels((*it1).first, pr2.first);
          (*it1).second+=pr2.second;
          CountedList.pop_back();
          break;
        }
        for(auto it=CountedList.begin(); it!=std::prev(CountedList.end(),2); ++it) {
          if((*it).second+pr2.second<T::NDUST) { it1=it; break; }
        }
        merge_parcels((*it1).first, pr2.first);
        (*it1).second+=CountedList.back().second;
        CountedList.pop_back();
      }
      this->clear();
      for(auto parcel: CountedList) this->push_back(parcel.first);
      return;
    }

    /* ---------------------------------------------------------------------- */
    void merge_parcels(T p1, T p2) {  //Merges p2 into p1. Keep p1 and discard p2
      Kokkos::View<uint32_t [T::NDUST]> ix1("ix1"), ix2("ix2"), ix3("ix3");
      Kokkos::parallel_scan(T::NDUST, KOKKOS_LAMBDA(const size_t& n, uint32_t& upd, const bool& final) {
          if(final) ix1(n)=upd;
          if(p1.state(n)==T::UNOCCUPIED) upd+=1;
      } );
      Kokkos::parallel_scan(T::NDUST, KOKKOS_LAMBDA(const size_t& n, uint32_t& upd, const bool& final) {
          if(final) ix2(n)=upd;
          if(p2.state(n)!=T::UNOCCUPIED) upd+=1;
      } );
      Kokkos::parallel_for(T::NDUST, KOKKOS_LAMBDA(const size_t& n) {
          if(p1.state(n)==T::UNOCCUPIED) ix3(ix1(n))=n;
      } );
      Kokkos::parallel_for(T::NDUST, KOKKOS_LAMBDA(const size_t& n) {
          if(p2.state(n)!=T::UNOCCUPIED) {
            p1.ssn  (ix3(ix2(n)))=p2.ssn  (n);
            p1.state(ix3(ix2(n)))=p2.state(n);
            for(int m=0; m<3; m++) p1.loc(ix3(ix2(n)),m)=p2.loc(n, m);
          }
      } );
      for(auto var2 : p2.ScalarPointVariables) {
        auto s1=p1.ScalarPointVariables.find(var2.first)->second;
        auto s2=var2.second;
        Kokkos::parallel_for(T::NDUST, KOKKOS_LAMBDA(const size_t& n) {
            if(p2.state(n)!=T::UNOCCUPIED) s1(ix3(ix2(n)))=s2(n);
        } );
      }
      for(auto var2 : p2.Vectr3PointVariables) {
        auto s1=p1.Vectr3PointVariables.find(var2.first)->second;
        auto s2=var2.second;
        Kokkos::parallel_for(T::NDUST, KOKKOS_LAMBDA(const size_t& n) {
            if(p2.state(n)!=T::UNOCCUPIED) for(int m=0; m<3; m++) s1(ix3(ix2(n)),m)=s2(n,m);
        } );
      }
    }

    /* ---------------------------------------------------------------------- */
    Lint<T> extract(Dust::STATE s=T::HEALTHY, bool kill=false) {
      Lint <T> newlist;
      for(T p2: *this) {
        if(p2.getcount(s)==0) continue;
        T p1;
        Kokkos::parallel_for(T::NDUST, KOKKOS_LAMBDA(const size_t& n) {
            if(p2.state(n)==s) {
              p1.ssn  (n)=p2.ssn  (n);
              p1.state(n)=p2.state(n);
              for(int m=0; m<3; m++) p1.loc(n,m)=p2.loc(n, m);
            }
        } );
        for(auto var2 : p2.ScalarPointVariables) {
          auto s1=p1.ScalarPointVariables.find(var2.first)->second;
          auto s2=var2.second;
          Kokkos::parallel_for(T::NDUST, KOKKOS_LAMBDA(const size_t& n) {
              if(p2.state(n)==s) s1(n)=s2(n);
          } );
        }
        for(auto var2 : p2.Vectr3PointVariables) {
          auto s1=p1.Vectr3PointVariables.find(var2.first)->second;
          auto s2=var2.second;
          Kokkos::parallel_for(T::NDUST, KOKKOS_LAMBDA(const size_t& n) {
              if(p2.state(n)==s) for(int m=0; m<3; m++) s1(n,m)=s2(n,m);
          } );
        }
        newlist.push_back(p1);
        if (kill) Kokkos::parallel_for(T::NDUST, KOKKOS_LAMBDA(const size_t& n) {
            if(p2.state(n)==s) p2.state(n)=T::UNOCCUPIED;
        } );
      }
      newlist.compact();
      return(newlist);
    }

    /* ---------------------------------------------------------------------- */
    void write_silo(std::string prefix="Lint") const {
      if(globalcomm.rank()==0) {
        std::string dirname=prefix+"_silodir";
        if(!boost::filesystem::exists(dirname)) {
          boost::filesystem::path dir(dirname);
          boost::filesystem::create_directory(dir);
        }
      }
      globalcomm.barrier();
      char filename[1000];
      sprintf(filename, "%s_silodir/%s%06d.silo", prefix.c_str(), prefix.c_str(), globalcomm.rank());
      DBfile *file=nullptr;
      file=DBCreate(filename, DB_CLOBBER, DB_LOCAL, NULL, DB_HDF5);

      int count=-1;
      for(T parcel: *this) {
        count++;
        char meshname[1000];
        sprintf(meshname, "Points%06d", count);
        parcel.write_silo_mesh(file, meshname);
      }
      DBClose(file);

      size_t numparcels=this->size();//number of dust parcels on this rank
      std::vector<size_t> numparcels_g;
      boost::mpi::gather(globalcomm, numparcels, numparcels_g, 0);
      if(globalcomm.rank()!=0) return;

      std::vector<std::string> meshnames;
      std::vector<char*> meshnamesptr;
      std::vector<int> meshtypes;

      std::vector<std::string> pathnames;
      std::vector<int> vartypes;

      for(int n=0; n<globalcomm.size(); n++) {
        sprintf(filename, "%s_silodir/%s%06d.silo", prefix.c_str(), prefix.c_str(), n);
        for(int m=0; m<numparcels_g[n]; m++) {
          count++;
          char meshname[1000];
          sprintf(meshname, "Points%06d", m);

          char name[1000];
          sprintf(name,"%s:%s/%s", filename, meshname, meshname);
          meshnames.push_back(std::string(name));
          meshnamesptr.push_back((char*)meshnames.back().c_str());
          meshtypes.push_back(DB_POINTMESH);

          sprintf(name,"%s:%s", filename, meshname);
          pathnames.push_back(std::string(name));
          vartypes.push_back(DB_POINTVAR);
        }
      }

      sprintf(filename, "%s.silo", prefix.c_str());
      file=DBCreate(filename, DB_CLOBBER, DB_LOCAL, NULL, DB_HDF5);
      DBPutMultimesh(file, "Points", meshnamesptr.size(), meshnamesptr.data(), meshtypes.data(), NULL);

      auto writemultivar = [=](std::string varname) {
        std::vector<std::string> varnames;
        std::vector<char*> varnamesptr;
        for(auto str: pathnames) {
          varnames.push_back(str+"/"+varname);
          varnamesptr.push_back((char*)varnames.back().c_str());
        }
        DBPutMultivar (file, varname.c_str(), varnamesptr.size(), varnamesptr.data(), vartypes.data(), NULL);
      };

      writemultivar(  "ssn");
      writemultivar("state");
      T dummy; //Use a dummy to find what variables exist
      for(auto it : dummy.ScalarPointVariables)
        writemultivar(it.first);
      for(auto it : dummy.Vectr3PointVariables) for(int m=0; m<3; m++)
        writemultivar(it.first+"vec"+char('0'+m));
      DBClose(file);
    }

    /* ---------------------------------------------------------------------- */
    void exchange(int NX, int NY, int NZ) {
      exchange_onedirn(0, NX, cartcomm.x, param.getbool("periodic_x"));
      exchange_onedirn(1, NY, cartcomm.y, param.getbool("periodic_y"));
      exchange_onedirn(2, NZ, cartcomm.z, param.getbool("periodic_z"));
      this->compact();
    }

    /* ---------------------------------------------------------------------- */
    void exchange_onedirn(int dim, int nmax, boost::mpi::communicator comm, bool periodic) {
      assert(dim==0 || dim==1 || dim==2);
      Dust::STATE WENT_LEFT, WENT_RGHT;
      switch(dim) {
        case 0: WENT_LEFT=T::WENT_LEFT_X; WENT_RGHT=T::WENT_RGHT_X; break;
        case 1: WENT_LEFT=T::WENT_LEFT_Y; WENT_RGHT=T::WENT_RGHT_Y; break;
        case 2: WENT_LEFT=T::WENT_LEFT_Z; WENT_RGHT=T::WENT_RGHT_Z; break;
      }

      for(T p: *this) {
        Kokkos::parallel_for(T::NDUST, KOKKOS_LAMBDA(const size_t &n) {
          if( p.state(n)==T::HEALTHY && p.loc(n, dim)< 0    ) {p.state(n)=WENT_LEFT; p.loc(n, dim)+=nmax;}
          if( p.state(n)==T::HEALTHY && p.loc(n, dim)>=nmax ) {p.state(n)=WENT_RGHT; p.loc(n, dim)-=nmax;}
          if((p.state(n)==WENT_LEFT || p.state(n)==WENT_RGHT) && (p.loc(n, dim)<0 || p.loc(n, dim)>=nmax) )
              p.state(n)=T::WENT_TOO_FAR;
        } );
      }

      Lint<T> sleft=this->extract(WENT_LEFT, true);
      Lint<T> srght=this->extract(WENT_RGHT, true);
      Lint<T> rleft;
      Lint<T> rrght;

      for(T p: sleft) {
        Kokkos::parallel_for(T::NDUST, KOKKOS_LAMBDA(const size_t &n) {
          if(p.state(n)==WENT_LEFT) p.state(n)=T::HEALTHY;
        } );
      }
      for(T p: srght) {
        Kokkos::parallel_for(T::NDUST, KOKKOS_LAMBDA(const size_t &n) {
          if(p.state(n)==WENT_RGHT) p.state(n)=T::HEALTHY;
        } );
      }

      int rank=comm.rank();
      int size=comm.size();
      int lid = (rank-1+size)%size;
      int rid = (rank+1+size)%size;
      if(!periodic && rank==0     ) sleft.clear();
      if(!periodic && rank==size-1) srght.clear();

      boost::mpi::request reqs[4];
      reqs[0]=comm.isend(lid, 791, sleft);
      reqs[1]=comm.isend(rid, 792, srght);
      reqs[2]=comm.irecv(lid, 792, rleft);
      reqs[3]=comm.irecv(rid, 791, rrght);
      boost::mpi::wait_all(reqs, reqs+4);

      this->splice(this->end(), rleft);
      this->splice(this->end(), rrght);
      return;
    }

    /* ---------------------------------------------------------------------- */
    void checkpoint(std::string prefix="Lint") const {
      if(globalcomm.rank()==0) {
        std::string dirname=prefix+"_ckptdir";
        if(!boost::filesystem::exists(dirname)) {
          boost::filesystem::path dir(dirname);
          boost::filesystem::create_directory(dir);
        }
      }
      globalcomm.barrier();
      char filename[1000];
      sprintf(filename, "%s_ckptdir/%06d.ckpt", prefix.c_str(), globalcomm.rank());
      std::ofstream ofs(filename, std::ios::binary);
      boost::archive::binary_oarchive oa(ofs);
      oa << *this;
    }

    /* ---------------------------------------------------------------------- */
    void restore(std::string prefix) {
      char filename[1000];
      sprintf(filename, "%s_ckptdir/%06d.ckpt", prefix.c_str(), globalcomm.rank());
      std::ifstream ifs(filename, std::ios::binary);
      boost::archive::binary_iarchive ia(ifs);
      ia >> *this;
    }

    /* ---------------------------------------------------------------------- */
  private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
      ar & boost::serialization::base_object<std::list<T>>(*this);
    }
};

#endif
