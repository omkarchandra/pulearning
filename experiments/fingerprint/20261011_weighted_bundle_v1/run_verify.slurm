#!/usr/bin/env bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=pu_weighted_verify
#SBATCH --output=/home/mes7956/projects/pulearning/experiments/fingerprint/20261011_weighted_bundle_v1/evidence/verify_%j.out
#SBATCH --error=/home/mes7956/projects/pulearning/experiments/fingerprint/20261011_weighted_bundle_v1/evidence/verify_%j.err

set -euo pipefail
: "${BUILD_JOB_ID:?BUILD_JOB_ID must name the successful weighted-bundle job}"
PROJECT="/home/mes7956/projects/pulearning"
RUN="$PROJECT/experiments/fingerprint/20261011_weighted_bundle_v1"
OUT="$RUN/runs/$BUILD_JOB_ID"
PY="$PROJECT/envs/20261005_quest_cre_phase3_v6/bin/python"
SOURCE_FP="$PROJECT/experiments/fingerprint/20261011_eval_repair_v1/v4_fingerprint.json"

export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
sha256sum "$RUN/verify_weighted_bundle.py" "$RUN/run_verify.slurm" \
  > "$OUT/verification_code_sha256_${SLURM_JOB_ID}.txt"
"$PY" -u "$RUN/verify_weighted_bundle.py" \
  --build-report "$OUT/build_report.json" \
  --fingerprint-report "$OUT/sorted_fingerprints.json" \
  --fingerprint-receipt "$OUT/fingerprint_receipt.json" \
  --source-fingerprint "$SOURCE_FP" \
  --output "$OUT/verification.json"
