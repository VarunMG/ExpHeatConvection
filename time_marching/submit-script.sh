#!/bin/bash
#This file is called submit-script.sh
#SBATCH --partition=shared       # default "shared", if not specified
#SBATCH --time=0-08:30:00       # run time in days-hh:mm:ss
#SBATCH --nodes=2              # require 1 nodes
#SBATCH --ntasks-per-node=32    # cpus per node (by default, "ntasks"="cpus")
#SBATCH --mem=8000             # RAM per node in megabytes
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
# Make sure to change the above two lines to reflect your appropriate
# file locations for standard error and output

# Now list your executable command (or a string of them).
# Example for code compiled with a software module:
eval "$(conda shell.bash hook)"
conda activate Convection
module load hdf5/1.12.2-gcc-11.3.0-openmpi-4.1.3
srun --mpi=pmix -n32 -N2 python3 /home/gudibanda/expHeating_Convection/time_marching/expHeating_timemarching.py
conda deactivate
