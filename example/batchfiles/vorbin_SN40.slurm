#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/vorbin.out
#SBATCH --error=logs/vorbin.err
#SBATCH --mem=30GB


python spectral_fitting.py parameterfiles/vorbin_SN40.param
