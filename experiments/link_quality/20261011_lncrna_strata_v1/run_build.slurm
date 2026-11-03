#!/usr/bin/env bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=pu_lncrna_strata
#SBATCH --output=/home/mes7956/projects/pulearning/experiments/link_quality/20261011_lncrna_strata_v1/evidence/build_%j.out
#SBATCH --error=/home/mes7956/projects/pulearning/experiments/link_quality/20261011_lncrna_strata_v1/evidence/build_%j.err

set -euo pipefail
PROJECT="/home/mes7956/projects/pulearning"
RUN="$PROJECT/experiments/link_quality/20261011_lncrna_strata_v1"
OUT="$RUN/runs/$SLURM_JOB_ID"
SCRATCH="/scratch/mes7956/pulearning/lncrna_strata_${SLURM_JOB_ID}"
PY="$PROJECT/envs/20261005_quest_cre_phase3_v6/bin/python"
BUNDLE="$PROJECT/datasets/processed/cre/phase3_bundles/20261005_quest_cre_phase3_v4"
GTF="$PROJECT/datasets/raw/cre/gencode/v50/gencode.v50.annotation.gtf.gz"
TSS="$PROJECT/datasets/processed/cre/gencode_v50_gene_tss.parquet"

export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT" "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

sha256sum "$RUN/build_lncrna_strata.py" "$RUN/RULES.md" "$RUN/run_build.slurm" \
  > "$OUT/build_code_sha256.txt"
/usr/bin/time -v -o "$OUT/build_time.txt" \
  "$PY" -u "$RUN/build_lncrna_strata.py" \
    --bundle-dir "$BUNDLE" \
    --gtf "$GTF" \
    --tss "$TSS" \
    --rules "$RUN/RULES.md" \
    --output-dir "$OUT/results" \
    --scratch-dir "$SCRATCH" \
    --threads 4 \
    --memory-limit 16GB
sha256sum "$OUT/results/lncrna_strata.parquet" \
  "$OUT/results/lncrna_strata_report.json" > "$OUT/build_result_sha256.txt"
