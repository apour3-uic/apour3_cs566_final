#!/bin/bash
#SBATCH --job-name=psort_phase1
#SBATCH --output=psort_phase1_%j.log
#SBATCH --error=psort_phase1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=csece566_dutt

cd /home/apour3/csece566_dutt_link/apour3/final

module load OpenMPI

# Compile
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi


echo "Phase 1 Test: A_vector_100000_8_2 (skewed pivots)"
srun --mpi=pmix -n 8 ./parallel_sort A_vector_100000_8_2.txt pivot_vector_100000_8_2.txt 100000

echo "Phase 1 Test: A_vector_100000_8_8 (varied pivots)"
srun --mpi=pmix -n 8 ./parallel_sort A_vector_100000_8_8.txt pivot_vector_100000_8_8.txt 100000

echo "Job completed at $(date)"
