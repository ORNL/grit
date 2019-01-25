#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <ctime>

class Utils {
  public:
    static std::string now(const char* format = "%T" );
};

#endif
