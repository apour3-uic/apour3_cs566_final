#!/bin/bash
# Quick interactive test — run from within an salloc session
# Usage: bash scripts/run_quick_test.sh
#
# First get an interactive allocation:
#   salloc --job-name "ParSort_Test" --cpus-per-task 1 --ntasks=8 --time 00:10:00 --account=csece566_dutt
#   cd /home/yqu30/csece566_dutt_link/yqu30/final
#   module load OpenMPI
#   make clean && make
#   bash scripts/run_quick_test.sh

mkdir -p outputs

echo "=== Quick test: P=8, no_lb, skewed pivots ==="
srun --mpi=pmix -n 8 ./parallel_sort \
    A_vector_100000_8_2.txt pivot_vector_100000_8_2.txt --mode=no_lb

echo ""
echo "=== Quick test: P=8, with_lb, skewed pivots ==="
srun --mpi=pmix -n 8 ./parallel_sort \
    A_vector_100000_8_2.txt pivot_vector_100000_8_2.txt --mode=with_lb

echo ""
echo "Done. Check outputs/ for sorted arrays and stats."
