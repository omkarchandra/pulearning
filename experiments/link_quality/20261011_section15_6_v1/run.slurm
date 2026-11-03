#!/usr/bin/env bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=pu_linkq_15_6

set -euo pipefail
PROJECT="/home/mes7956/projects/pulearning"
RUN="$PROJECT/experiments/link_quality/20261011_section15_6_v1"
OUTPUT="$RUN/results"
SCRATCH="$RUN/scratch"
PY="$PROJECT/envs/20261005_quest_cre_phase3_v6/bin/python"
BUNDLE="$PROJECT/datasets/processed/cre/phase3_bundles/20261005_quest_cre_phase3_v4"
RE2G="$PROJECT/datasets/raw/cre/encode_re2g/V4-hg38.Gene-Links.Computational-Methods.rE2G.txt.gz"

export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

sha256sum "$RUN/run_section15_6.py" "$RUN/run.slurm" > "$RUN/code_sha256_${SLURM_JOB_ID}.txt"
/usr/bin/time -v -o "$RUN/time_${SLURM_JOB_ID}.txt" \
    "$PY" -u "$RUN/run_section15_6.py" \
    --bundle-dir "$BUNDLE" \
    --re2g "$RE2G" \
    --output-dir "$OUTPUT" \
    --scratch-dir "$SCRATCH" \
    --threads 4 \
    --memory-limit 20GB
sha256sum "$OUTPUT"/* > "$RUN/result_sha256_${SLURM_JOB_ID}.txt"
