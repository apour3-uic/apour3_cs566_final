#!/bin/bash
#SBATCH --job-name=psort_phase3
#SBATCH --output=outputs/psort_phase3_%j.log
#SBATCH --error=outputs/psort_phase3_%j.err
#SBATCH --nodes=1
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

echo "============================================"
echo "Phase 1+2+3 Test: A_vector_100000_8_2 (skewed pivots)"
echo "============================================"
srun --mpi=pmix -n 8 ./parallel_sort A_vector_100000_8_2.txt pivot_vector_100000_8_2.txt 100000

echo ""
echo "============================================"
echo "Phase 1+2+3 Test: A_vector_100000_8_8 (varied pivots)"
echo "============================================"
srun --mpi=pmix -n 8 ./parallel_sort A_vector_100000_8_8.txt pivot_vector_100000_8_8.txt 100000

echo ""
echo "Job completed at $(date)"
