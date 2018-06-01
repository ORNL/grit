#ifndef RUNPARAMS_H
#define RUNPARAMS_H

#include <map>
#include <boost/mpi.hpp>
#include <boost/serialization/map.hpp>

class RunParams : private std::map<std::string, std::string> {
  public:
    RunParams() {
      // This is the place to set the defaults
      set("filterfrequency","1");
      set("periodic_x", "true");
      set("periodic_y", "true");
      set("periodic_z", "true");
    }
    RunParams(boost::mpi::communicator comm, std::string filename);
    RunParams(std::map<std::string, std::string> p);
    void  set     (const std::string& key, const std::string& value) {(*this)[key]=value;}
    bool  present (const std::string& key) const {return this->count(key);}
    int   getint  (const std::string& key) const {return std::stoi(this->at(key));}
    float getfloat(const std::string& key) const {return std::stof(this->at(key));}
    bool  getbool (const std::string& key) const {
      std::string value=this->at(key);
      if(value=="false"||value=="False"||value=="0") return(false);
      return(true);
    }
    std::string getstring(const std::string& key) const {return this->at(key);}
    void  print() const {for(auto p: *this) std::cout << p.first << " is " << p.second << std::endl;}

  private:
    friend class boost::serialization::access;
    template <class Archive> void serialize(Archive &ar, const unsigned int version) {
      ar & boost::serialization::base_object<std::map<std::string, std::string>>(*this);
    }
};

#endif
