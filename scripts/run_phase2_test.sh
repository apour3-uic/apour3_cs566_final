#!/bin/bash
#SBATCH --job-name=psort_phase2
#SBATCH --output=outputs/psort_phase2_%j.log
#SBATCH --error=outputs/psort_phase2_%j.err
#SBATCH --nodes=1

# Number of nodes to allocate this job
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=csece566_dutt

cd /home/apour3/csece566_dutt_link/apour3/final

module load OpenMPI

make clean && make
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

echo "Phase 1+2 Test: A_vector_100000_8_2 (skewed pivots)"
srun --mpi=pmix -n 8 ./parallel_sort A_vector_100000_8_2.txt pivot_vector_100000_8_2.txt

echo ""
echo "Phase 1+2 Test: A_vector_100000_8_8 (varied pivots)"
srun --mpi=pmix -n 8 ./parallel_sort A_vector_100000_8_8.txt pivot_vector_100000_8_8.txt

echo ""
echo "Job completed at $(date)"
