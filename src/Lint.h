#ifndef LINT_H
#define LINT_H

#include <list>
#include <map>
#include "GlobalVariables.h"

template <typename T>
class Lint : public std::list<T> {
  public:
    /* ---------------------------------------------------------------------- */
    void compact() {
      if(this->size()<2) return;
      typedef std::pair<T, int>  CountedT;
      auto comp = [] (const CountedT &p1, const CountedT &p2)->bool { return p1.second>p2.second; };
      std::list<CountedT> CountedList;
      for(T parcel: *this) {
        CountedList.push_back(std::make_pair(parcel, T::NDUST-parcel.getcount(T::UNOCCUPIED)));
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
            p1.age  (ix3(ix2(n)))=p2.age  (n);
            p1.dob  (ix3(ix2(n)))=p2.dob  (n);
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
    void write_silo(std::string prefix="Lint") const {
      char filename[1000];
      sprintf(filename, "%s%06d.silo", prefix.c_str(), globalcomm.rank());
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
        sprintf(filename, "%s%06d.silo", prefix.c_str(), n);
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
      DBPutMultimesh(file, prefix.c_str(), meshnamesptr.size(), meshnamesptr.data(), meshtypes.data(), NULL);

      auto writemultivar = [=](std::string varname) {
        std::vector<std::string> varnames;
        std::vector<char*> varnamesptr;
        for(auto str: pathnames) {
          varnames.push_back(str+"/"+varname);
          varnamesptr.push_back((char*)varnames.back().c_str());
        }
        DBPutMultivar (file, varname.c_str(), varnamesptr.size(), varnamesptr.data(), vartypes.data(), NULL);
      };

      writemultivar(  "age");
      writemultivar(  "dob");
      writemultivar(  "ssn");
      writemultivar("state");
      for(auto it : this->front().ScalarPointVariables) //RSA problem if this was empty
        writemultivar(it.first);
      DBClose(file);
    }
    /* ---------------------------------------------------------------------- */
};

#endif
