#!/bin/bash
#SBATCH --job-name=psort_rand
#SBATCH --output=outputs/psort_rand_%j.log
#SBATCH --error=outputs/psort_rand_%j.err
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
echo "Deterministic = Random Test: A_vector_100000_8_2 (skewed pivots)"
echo "============================================"
srun --mpi=pmix -n 8 ./parallel_sort Inputs/A_vector_100000_8_2.txt Inputs/pivot_vector_100000_8_2.txt --mode=with_lb
srun --mpi=pmix -n 8 ./parallel_sort Inputs/A_vector_100000_8_2.txt Inputs/pivot_vector_100000_8_2.txt --mode=with_lb --cl=rand

echo ""
echo "============================================"
echo "Random + Deterministic Test: A_vector_100000_8_8 (varied pivots)"
echo "============================================"
srun --mpi=pmix -n 8 ./parallel_sort Inputs/A_vector_100000_8_8.txt Inputs/pivot_vector_100000_8_8.txt --mode=with_lb
srun --mpi=pmix -n 8 ./parallel_sort Inputs/A_vector_100000_8_8.txt Inputs/pivot_vector_100000_8_8.txt --mode=with_lb --cl=rand

echo ""
echo "Job completed at $(date)"
