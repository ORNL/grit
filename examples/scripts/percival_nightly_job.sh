#!/bin/bash
#PBS -N grit
#PBS -A STF006
#PBS -j oe
#PBS -o /ccs/home/rsankar/grit/workspace/percival_nightly.$PBS_JOBID.txt
#PBS -l nodes=1
#PBS -l walltime=1:00:00

source /opt/cray/pe/modules/3.2.10.5/init/bash
module list
module swap PrgEnv-intel PrgEnv-gnu
module swap gcc gcc/6.3.0
module load cray-hdf5
module list

set -x 
echo $PBS_JOBID
date
NEXTRUNTIME=`date --date="tomorrow 1AM" +%m%d%H%M`
qsub -a $NEXTRUNTIME /ccs/home/rsankar/grit/examples/scripts/percival_nightly_job.sh

export CTEST_SITE="percival"
export   BOOST_DIR=$HOME/mysw/percival_gcc630/boost-1.67.0
export  KOKKOS_DIR=$HOME/mysw/percival_gcc630/kokkosOMP_d3a9419
export    SILO_DIR=$HOME/mysw/percival_gcc630/silo-4.10.2
ctest -S $HOME/grit/examples/scripts/cray_nightly_gnu_openmp.cmake -VV

date
