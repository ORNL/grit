#include <fstream>
#include <mpi.h>
#include "IO/CollectiveSingleFile.h"
#include "GlobalVariables.h"

/* ---------------------------------------------------------------------- */
void CollectiveArraytoSingleFile(char mode, std::string filename, int NX, int NY, int NZ, int NV, double *Fa){
  int l_size[3], g_size[3], starts[3];
  // local array dimensions
  l_size[0] = NX;
  l_size[1] = NY;
  l_size[2] = NZ;
  // global array dimensions
  g_size[0] = NX*cartcomm.x.size();
  g_size[1] = NY*cartcomm.y.size();
  g_size[2] = NZ*cartcomm.z.size();
  // local array's start offsets in global array
  starts[0] = NX*cartcomm.x.rank();
  starts[1] = NY*cartcomm.y.rank();
  starts[2] = NZ*cartcomm.z.rank();

  MPI_Datatype subviewType;
  MPI_Type_create_subarray(3, g_size, l_size, starts, MPI_ORDER_FORTRAN, MPI_DOUBLE, &subviewType);
  MPI_Type_commit(&subviewType);

  // set MPI I/O hints
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "romio_ds_write", "disable");

  // open the file
  MPI_File fp;
  int open_mode;
  if(mode == 'w')
    open_mode = MPI_MODE_WRONLY|MPI_MODE_CREATE;
  else
    open_mode = MPI_MODE_RDONLY;

  int err=MPI_File_open(globalcomm, filename.c_str(), open_mode, info, &fp);
  if (err != MPI_SUCCESS) {
    char error_string[MPI_MAX_ERROR_STRING];
    int length_of_error_string;
    MPI_Error_string(err, error_string, &length_of_error_string);
    fprintf(stderr, "%3d: %s\n", globalcomm.rank(), error_string);
    MPI_Abort(globalcomm, 1);
  }
  MPI_File_set_view(fp, 0, MPI_DOUBLE, subviewType, "native", MPI_INFO_NULL);

  // read or write to file
  MPI_Status status;
  if (mode == 'w')
      MPI_File_write_all(fp, Fa, NX*NY*NZ*NV, MPI_DOUBLE, &status);
  else
      MPI_File_read_all (fp, Fa, NX*NY*NZ*NV, MPI_DOUBLE, &status);

  // close file
  MPI_File_close(&fp);

  // Type and info cleanup
  MPI_Type_free(&subviewType);
  MPI_Info_free(&info);
}

/* ---------------------------------------------------------------------- */
void CollectiveWriteScalartoSingleFile(std::string filename, int NX, int NY, int NZ,         Yarn::ScalarFieldType F){
  size_t NP=NX*NY*NZ;
  double *Fa = new double[NP];
  Yarn::ScalarFieldToFortranArray (NP,     Fa, F);
  CollectiveArraytoSingleFile ('w', filename, NX, NY, NZ,  1, Fa);
  delete[] Fa;
}

/* ---------------------------------------------------------------------- */
void CollectiveWriteVectortoSingleFile(std::string filename, int NX, int NY, int NZ, int NV, Yarn::VectorFieldType F){
  size_t NP=NX*NY*NZ;
  double *Fa = new double[NP*NV];
  Yarn::VectorFieldToFortranArray (NP, NV, Fa, F);
  CollectiveArraytoSingleFile ('w', filename, NX, NY, NZ, NV, Fa);
  delete[] Fa;
}

/* ---------------------------------------------------------------------- */
void CollectiveReadScalarfromSingleFile(std::string filename, int NX, int NY, int NZ,         Yarn::ScalarFieldType F){
  size_t NP=NX*NY*NZ;
  double *Fa = new double[NP];
  CollectiveArraytoSingleFile ('r', filename, NX, NY, NZ,  1, Fa);
  Yarn::FortranArrayToScalarField (NP,     Fa, F);
  delete[] Fa;
}

/* ---------------------------------------------------------------------- */
void CollectiveReadVectorfromSingleFile(std::string filename, int NX, int NY, int NZ, int NV, Yarn::VectorFieldType F){
  size_t NP=NX*NY*NZ;
  double *Fa = new double[NP*NV];
  CollectiveArraytoSingleFile ('r', filename, NX, NY, NZ, NV, Fa);
  Yarn::FortranArrayToVectorField (NP, NV, Fa, F);
  delete[] Fa;
}

/* ---------------------------------------------------------------------- */
void WriteBOVforScalarinSingleFile (std::string filename, int NX, int NY, int NZ,
                                        std::string datfilename, std::string varname){
  int px = param.getint("px");
  int py = param.getint("py");
  int pz = param.getint("pz");
  FILE * bovfile = fopen(filename.c_str(), "w");
  fprintf(bovfile, "DATA_FILE: %s\n", datfilename.c_str());
  fprintf(bovfile, "DATA_SIZE: %4d %4d %4d\n", NX*px, NY*py, NZ*pz);
  fprintf(bovfile, "DATA_FORMAT: DOUBLE\n");
  fprintf(bovfile, "VARIABLE: %s\n", varname.c_str());
  fprintf(bovfile, "BRICK_ORIGIN: %+5.1f %+5.1f %+5.1f\n", grid.x(0),grid.y(0),grid.z(0) );
  fprintf(bovfile, "BRICK_SIZE:   %+5.1f %+5.1f %+5.1f\n",
      grid.x(NX*px-1)-grid.x(0),
      grid.y(NY*py-1)-grid.y(0),
      grid.z(NZ*pz-1)-grid.z(0) );
  fclose(bovfile);
}
