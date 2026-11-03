#!/usr/bin/env bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=pu_linkq_context

set -euo pipefail
PROJECT="/home/mes7956/projects/pulearning"
RUN="$PROJECT/experiments/link_quality/20261011_section15_6_v1"
OUTPUT="$RUN/context_results_corrected"
SCRATCH="${TMPDIR:-/tmp}/pu_linkq_context_${SLURM_JOB_ID}"
PY="$PROJECT/envs/20261005_quest_cre_phase3_v6/bin/python"
BUNDLE="$PROJECT/datasets/processed/cre/phase3_bundles/20261005_quest_cre_phase3_v4"
RE2G="$PROJECT/datasets/raw/cre/encode_re2g/V4-hg38.Gene-Links.Computational-Methods.rE2G.txt.gz"
TSS="$PROJECT/datasets/processed/cre/gencode_v50_gene_tss.parquet"

export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

sha256sum "$RUN/revalidate_section15_context.py" "$RUN/run_context.slurm" \
    > "$RUN/context_code_sha256_${SLURM_JOB_ID}.txt"
/usr/bin/time -v -o "$RUN/context_time_${SLURM_JOB_ID}.txt" \
    "$PY" -u "$RUN/revalidate_section15_context.py" \
    --bundle-dir "$BUNDLE" \
    --re2g "$RE2G" \
    --tss "$TSS" \
    --output-dir "$OUTPUT" \
    --scratch-dir "$SCRATCH" \
    --threads 4 \
    --memory-limit 48GB
sha256sum "$OUTPUT"/* > "$RUN/context_result_sha256_${SLURM_JOB_ID}.txt"
