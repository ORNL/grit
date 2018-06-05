#/bin/bash
#BSUB -P CSC190RAPTOR
#BSUB -J grit
#BSUB -o /ccs/home/rsankar/grit/workspace/summit_nightly.o%J.txt
#BSUB -W 60
#BSUB -nnodes 1
#BSUB -alloc_flags gpumps

module purge
module load DefApps
module unload xalt
module swap xl gcc/6.4.0
module load git
module load cmake
module load cuda
module list

set -x
date
NEXTRUNTIME=`date --date="tomorrow 1AM" +%m:%d:%H:%M`
bsub -b $NEXTRUNTIME /ccs/home/rsankar/grit/examples/scripts/summit_nightly_job.sh

export CTEST_SITE="summit.ccs.ornl.gov"
export   BOOST_DIR=$HOME/mysw/summit_gcc640/boost-1.67.0
export  KOKKOS_DIR=$HOME/mysw/summit_gcc640/kokkosCuda9.2.64_d3a9419
export    SILO_DIR=$HOME/mysw/summit_gcc640/silo-4.10.2
export    HDF5_DIR=$HOME/mysw/summit_gcc640/hdf5-1.10.1

ctest -S $HOME/grit/examples/scripts/summit_nightly_cuda.cmake -VV

date
