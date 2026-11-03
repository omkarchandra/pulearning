#!/usr/bin/env bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=pu_weighted_bundle
#SBATCH --output=/home/mes7956/projects/pulearning/experiments/fingerprint/20261011_weighted_bundle_v1/evidence/build_%j.out
#SBATCH --error=/home/mes7956/projects/pulearning/experiments/fingerprint/20261011_weighted_bundle_v1/evidence/build_%j.err

set -euo pipefail
PROJECT="/home/mes7956/projects/pulearning"
RUN="$PROJECT/experiments/fingerprint/20261011_weighted_bundle_v1"
OUT="$RUN/runs/$SLURM_JOB_ID"
SCRATCH="/scratch/mes7956/pulearning/weighted_bundle_${SLURM_JOB_ID}"
PY="$PROJECT/envs/20261005_quest_cre_phase3_v6/bin/python"
BUNDLE="$PROJECT/datasets/processed/cre/phase3_bundles/20261005_quest_cre_phase3_v4"

export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PU_ATOMIC_DIRECTORY_MODE=symlink
mkdir -p "$OUT" "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

sha256sum "$RUN/build_weighted_bundle.py" "$RUN/DESIGN.md" "$RUN/run_build.slurm" \
  > "$OUT/build_code_sha256.txt"
/usr/bin/time -v -o "$OUT/build_time.txt" \
  "$PY" -u "$RUN/build_weighted_bundle.py" \
    --source-bundle "$BUNDLE" \
    --design "$RUN/DESIGN.md" \
    --run-dir "$OUT" \
    --scratch-dir "$SCRATCH" \
    --lambda-bp 68152.45706 \
    --threads 4 \
    --memory-limit 8GB
sha256sum "$OUT/build_report.json" "$OUT/full_replay.json" \
  "$OUT/weighted_links.parquet" "$OUT/weighted_bundle/manifest.json" \
  > "$OUT/build_result_sha256.txt"
