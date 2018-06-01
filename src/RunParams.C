#include <fstream>
#include <regex>
#include "RunParams.h"

/* ---------------------------------------------------------------------- */
RunParams::RunParams(boost::mpi::communicator comm, std::string filename) : RunParams() {
  if(comm.rank()==0) {
    std::ifstream file;
    std::stringstream ss;
    std::string line;
    file.open(filename.c_str());
    while(std::getline(file, line)) {
      line=std::regex_replace(line,std::regex("#.*"),"");//Treat anything after # as a comment
      ss<<line<<std::endl;
    }
    std::string key, value;
    while(ss>>key>>value) (*this)[key]=value;
  }
  broadcast(comm, *this, 0);
}

/* ---------------------------------------------------------------------- */
RunParams::RunParams(std::map<std::string, std::string> p) : RunParams() {
  this->insert(p.begin(), p.end());
}
