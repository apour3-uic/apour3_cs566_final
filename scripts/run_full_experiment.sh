#!/bin/bash
#SBATCH --job-name=psort_full
#SBATCH --output=outputs/psort_full_%j.log
#SBATCH --error=outputs/psort_full_%j.err
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
echo "  Full Experiment Grid"
echo "  P=1 (baseline), then P=8..128 x s=2/P x {no_lb, with_lb, with_lb+rand}"
echo "  N=100000/1000000"
echo "============================================"

CONFIGS="no_lb:default with_lb:default with_lb:rand"

# Sequential baseline (P=1, one run per N) for speedup denominator.
# LB and --cl=rand are no-ops at P=1, so only the default config is run.
for N in $N_VALUES; do
    AFILE="Inputs/A_vector_${N}_1_1.txt"
    PFILE="Inputs/pivot_vector_${N}_1_1.txt"
    if [ ! -f "$AFILE" ] || [ ! -f "$PFILE" ]; then
        echo "SKIP: missing $AFILE or $PFILE (P=1 baseline)"
        continue
    fi
    echo ""
    echo "============================================"
    echo "Sequential baseline: N=$N P=1"
    echo "============================================"
    srun --mpi=pmix -n 1 ./parallel_sort \
        "$AFILE" "$PFILE" ${N} --mode=no_lb --cl=default
done

for N in $N_VALUES; do
    for P in $P_VALUES; do
        for s in 2 $P; do
            for config in $CONFIGS; do
                mode=${config%:*}
                cl=${config#*:}

                AFILE="Inputs/A_vector_${N}_${P}_${s}.txt"
                PFILE="Inputs/pivot_vector_${N}_${P}_${s}.txt"

                if [ ! -f "$AFILE" ] || [ ! -f "$PFILE" ]; then
                    echo "SKIP: missing $AFILE or $PFILE"
                    continue
                fi

                echo ""
                echo "============================================"
                echo "Run: N=$N P=$P s=$s mode=$mode cl=$cl"
                echo "============================================"
                srun --mpi=pmix -n ${P} ./parallel_sort \
                    "$AFILE" "$PFILE" ${N} --mode=${mode} --cl=${cl}
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
