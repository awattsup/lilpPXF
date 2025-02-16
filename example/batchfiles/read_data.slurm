#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/read.out
#SBATCH --error=logs/read.err
#SBATCH --mem=30GB


python spectral_fitting.py parameterfiles/read_data.param
