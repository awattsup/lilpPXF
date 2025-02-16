#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/N4383_logs.out
#SBATCH --error=logs/N4383_logs.err
#SBATCH --mem=50GB


mpirun -np 1 python spectral_fitting.py parameterfiles/read_data.param
mpirun -np 1 python spectral_fitting.py parameterfiles/vorbin_SN40.param
mpirun -np 48 python spectral_fitting.py parameterfiles/SN40_stelkin_blue_MILES.param
mpirun -np 1 python spectral_fitting.py parameterfiles/vorbin_continuum_Halpha.param
