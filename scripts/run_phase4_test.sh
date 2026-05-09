#!/bin/bash
#SBATCH --job-name=psort_phase4
#SBATCH --output=outputs/psort_phase4_%j.log
#SBATCH --error=outputs/psort_phase4_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=csece566_dutt

cd /home/apour3/csece566_dutt_link/apour3/final

module load OpenMPI

make clean && make
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

mkdir -p outputs

P_VALUES="8 16 32 64 128"
N_VALUES="100000 1000000"

echo "============================================"
echo "  Phase 1-6 Full Pipeline Test"
echo "  P=8..128, N=100000/1000000, s=2/P"
echo "============================================"

for N in $N_VALUES; do
    for P in $P_VALUES; do
        for s in 2 $P; do
            for mode in no_lb with_lb; do
                AFILE="Inputs/A_vector_${N}_${P}_${s}.txt"
                PFILE="Inputs/pivot_vector_${N}_${P}_${s}.txt"

                if [ ! -f "$AFILE" ] || [ ! -f "$PFILE" ]; then
                    echo "SKIP: missing $AFILE or $PFILE"
                    continue
                fi

                echo ""
                echo "============================================"
                echo "Run: N=$N P=$P s=$s mode=$mode"
                echo "============================================"
                srun --mpi=pmix -n ${P} ./parallel_sort \
                    "$AFILE" "$PFILE" --mode=${mode}
            done
        done
    done
done

echo ""
echo "============================================"
echo "All experiments completed at $(date)"
echo "============================================"
echo ""
echo "Output files in outputs/ directory:"
ls -la outputs/sorted_*.txt outputs/stats_*.txt 2>/dev/null
