#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sda@cwi.nl
#SBATCH --array=1-16

# Note:
# - gpu_a100: 18 cores
# - gpu_h100: 16 cores
# https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting

mkdir -p /scratch-shared/$USER

export JULIA_DEPOT_PATH=/scratch-shared/$USER/.julia:

cd $HOME/projects/IncompressibleNavierStokes/lib/PaperDC

julia --project postanalysis.jl
