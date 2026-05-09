#!/bin/bash
# Generates all input/pivot file pairs for the parallel sort experiments.
# Usage: run from the project root directory (where input_gen lives).
#
# File naming: A_vector_{N}_{P}_{s}.txt / pivot_vector_{N}_{P}_{s}.txt
# s values tested: 2 (highly skewed) and P (balanced)
# P values: 8, 16, 32, 64, 128
# N values: 100000, 1000000

GEN=./input_gen
OUTDIR=Inputs

mkdir -p "$OUTDIR"

if [ ! -x "$GEN" ]; then
    echo "Error: $GEN not found or not executable"
    exit 1
fi

P_VALUES="8 16 32 64 128"
N_VALUES="100000 1000000"

for N in $N_VALUES; do
    for P in $P_VALUES; do
        for s in 2 $P; do
            # Skip duplicate when P=2 (s=2 and s=P are the same)
            if [ "$s" -eq 2 ] && [ "$P" -eq 2 ]; then
                continue
            fi

            AFILE="$OUTDIR/A_vector_${N}_${P}_${s}.txt"
            PFILE="$OUTDIR/pivot_vector_${N}_${P}_${s}.txt"

            echo "Generating N=$N P=$P s=$s ..."
            $GEN $N $P $s

            # Generator may write files to cwd with the final name already embedded.
            # Move any A_vector / pivot_vector files from cwd into OUTDIR.
            for f in A_vector_${N}_${P}_${s}.txt pivot_vector_${N}_${P}_${s}.txt; do
                if [ -f "$f" ]; then
                    mv "$f" "$OUTDIR/$f"
                fi
            done
            # Fallback: generator uses generic names
            [ -f "A_vector.txt" ]     && mv A_vector.txt     "$AFILE"
            [ -f "pivot_vector.txt" ] && mv pivot_vector.txt "$PFILE"

            echo "  -> $AFILE"
            echo "  -> $PFILE"
        done
    done
done

echo ""
echo "Done. Files in $OUTDIR/:"
ls "$OUTDIR/" | sort
