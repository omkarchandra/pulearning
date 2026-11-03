#!/usr/bin/env bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=pu_lncrna_verify
#SBATCH --output=/home/mes7956/projects/pulearning/experiments/link_quality/20261011_lncrna_strata_v1/evidence/verify_%j.out
#SBATCH --error=/home/mes7956/projects/pulearning/experiments/link_quality/20261011_lncrna_strata_v1/evidence/verify_%j.err

set -euo pipefail
: "${BUILD_JOB_ID:?BUILD_JOB_ID must name the successful lncRNA-strata job}"
PROJECT="/home/mes7956/projects/pulearning"
RUN="$PROJECT/experiments/link_quality/20261011_lncrna_strata_v1"
OUT="$RUN/runs/$BUILD_JOB_ID"
SCRATCH="/scratch/mes7956/pulearning/lncrna_strata_verify_${SLURM_JOB_ID}"
PY="$PROJECT/envs/20261005_quest_cre_phase3_v6/bin/python"

export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

sha256sum "$RUN/verify_lncrna_strata.py" "$RUN/run_verify.slurm" \
  > "$OUT/verification_code_sha256_${SLURM_JOB_ID}.txt"
/usr/bin/time -v -o "$OUT/verification_time_${SLURM_JOB_ID}.txt" \
  "$PY" -u "$RUN/verify_lncrna_strata.py" \
    --report "$OUT/results/lncrna_strata_report.json" \
    --table "$OUT/results/lncrna_strata.parquet" \
    --rules "$RUN/RULES.md" \
    --output "$OUT/verification.json" \
    --scratch-dir "$SCRATCH" \
    --threads 4 \
    --memory-limit 16GB
