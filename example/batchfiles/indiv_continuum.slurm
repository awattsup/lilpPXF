#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/HAcontinuum.out
#SBATCH --error=logs/HAcontinuum.err
#SBATCH --mem=50GB


mpirun -np 1 python spectral_fitting.py parameterfiles/vorbin_continuum_Halpha.param
