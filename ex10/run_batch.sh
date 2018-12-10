#!/usr/bin/env bash
# File       : run_batch.sh
# Created    : Mon Dec 10 2018 10:00:00 AM (+0100)
# Description:  Batch execute floating point compressor for tolerance = 0, 0.2, ... 10
#              ./run_batch.sh 
# bhubmann@student.ethz.ch
set -Eeuo pipefail

dim=4096

for tol in $(seq 0 0.2 10)
do
    # run compressor
    result=$(mpirun -n 6 ./mpi_float_compression "${tol}" ${dim} cyclone.bin.gz | grep "Compression rate" | awk -F":" '{print $NF}')
    echo "$tol $result"
done