#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=24:00:00
#SBATCH --output=logs/SN40_stelkin.out
#SBATCH --error=logs/SN40_stelkin.err
#SBATCH --mem=50GB


mpirun -np 48 python spectral_fitting.py parameterfiles/SN40_stelkin.param


