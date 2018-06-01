#ifndef COLLECTIVESINGLEFILE_H
#define COLLECTIVESINGLEFILE_H

#include "FDM/Yarn.h"

void CollectiveArraytoSingleFile(char mode, std::string filename, int NX, int NY, int NZ, int NV, double *Fa);
void CollectiveWriteScalartoSingleFile (std::string filename, int NX, int NY, int NZ,         Yarn::ScalarFieldType F);
void CollectiveWriteVectortoSingleFile (std::string filename, int NX, int NY, int NZ, int NV, Yarn::VectorFieldType F);
void CollectiveReadScalarfromSingleFile(std::string filename, int NX, int NY, int NZ,         Yarn::ScalarFieldType F);
void CollectiveReadVectorfromSingleFile(std::string filename, int NX, int NY, int NZ, int NV, Yarn::VectorFieldType F);
void WriteBOVforScalarinSingleFile     (std::string filename, int NX, int NY, int NZ,
                                        std::string datfilename, std::string varname);
#endif
