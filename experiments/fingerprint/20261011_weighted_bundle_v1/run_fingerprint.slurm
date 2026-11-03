#!/usr/bin/env bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --job-name=pu_weighted_fp
#SBATCH --output=/home/mes7956/projects/pulearning/experiments/fingerprint/20261011_weighted_bundle_v1/evidence/fingerprint_%j.out
#SBATCH --error=/home/mes7956/projects/pulearning/experiments/fingerprint/20261011_weighted_bundle_v1/evidence/fingerprint_%j.err

set -euo pipefail
: "${BUILD_JOB_ID:?BUILD_JOB_ID must name the successful weighted-bundle job}"
PROJECT="/home/mes7956/projects/pulearning"
RUN="$PROJECT/experiments/fingerprint/20261011_weighted_bundle_v1"
OUT="$RUN/runs/$BUILD_JOB_ID"
BUNDLE="$OUT/weighted_bundle"
SCRATCH="/scratch/mes7956/pulearning/weighted_fingerprint_${SLURM_JOB_ID}"
PY="$PROJECT/envs/20261005_quest_cre_phase3_v6/bin/python"

export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

sha256sum "$RUN/fingerprint_weighted_bundle.py" "$RUN/record_fingerprint_receipt.py" \
  "$RUN/run_fingerprint.slurm" > "$OUT/fingerprint_code_sha256_${SLURM_JOB_ID}.txt"
/usr/bin/time -v -o "$OUT/fingerprint_time_${SLURM_JOB_ID}.txt" \
  "$PY" -u "$RUN/fingerprint_weighted_bundle.py" \
    --bundle "$BUNDLE" \
    --scratch-dir "$SCRATCH" \
    --output "$OUT/sorted_fingerprints.json" \
    --memory-limit 12GB \
    --threads 4 \
    --batch-size 10000
"$PY" -u "$RUN/record_fingerprint_receipt.py" \
  --report "$OUT/sorted_fingerprints.json" \
  --fingerprint-script "$RUN/fingerprint_weighted_bundle.py" \
  --output "$OUT/fingerprint_receipt.json"
