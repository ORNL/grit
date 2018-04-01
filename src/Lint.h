#ifndef LINT_H
#define LINT_H

#include <list>

template <typename T>
class Lint : public std::list<T> {
  public:
    /* ---------------------------------------------------------------------- */
    void write_silo(std::string prefix="Lint") const {
      char filename[1000];
      sprintf(filename, "%sdata.silo", prefix.c_str());
      DBfile *file=nullptr;
      file=DBCreate(filename, DB_CLOBBER, DB_LOCAL, NULL, DB_HDF5);

      std::vector<std::string> meshnames;
      std::vector<char*> meshnamesptr;
      std::vector<int> meshtypes;

      std::vector<std::string> pathnames;
      std::vector<int> vartypes;

      int count=-1;
      for(T parcel: *this) {
        count++;
        char meshname[1000];
        sprintf(meshname, "Points%06d", count);
        parcel.write_silo_mesh(file, meshname);

        char name[1000];
        sprintf(name,"%s:%s/%s", filename, meshname, meshname);
        meshnames.push_back(std::string(name));
        meshnamesptr.push_back((char*)meshnames.back().c_str());
        meshtypes.push_back(DB_POINTMESH);

        sprintf(name,"%s:%s", filename, meshname);
        pathnames.push_back(std::string(name));
        vartypes.push_back(DB_POINTVAR);
      }
      DBClose(file);

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
      for(auto it : this->front().ScalarPointVariables) {
        writemultivar(it.first);
      }
      DBClose(file);
    }
    /* ---------------------------------------------------------------------- */
};

#endif
