#!/usr/bin/env bash
# Entrypoint of the training image. Vertex AI mounts Cloud Storage buckets under /gcs/<bucket>.
#
#   vertex_entrypoint.sh <dataset> <command...>
#   e.g. vertex_entrypoint.sh tinystories-10k scripts/train_tinystories_base.sh --device cuda --run_name a100-base
#
# Expects the bucket (name in $GCS_BUCKET) to mirror the repo layout: data/<dataset>/*.npy and tokenizers/<name>/,
# and writes runs/ there. The dataset and all tokenizers are copied to local disk first: np.memmap reads small random
# pieces, which is slow over the mount. The run directory is written locally and copied to the bucket when the command
# ends, also when it fails.
set -euo pipefail

: "${GCS_BUCKET:?set GCS_BUCKET to the bucket name}"
dataset="$1"
shift
bucket_dir="/gcs/${GCS_BUCKET}"

ls /usr/local/nvidia/lib64 2>/dev/null | grep -m1 libcuda || echo "no libcuda in /usr/local/nvidia/lib64"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || echo "nvidia-smi not available"
uv run python -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, GPU available: {torch.cuda.is_available()}')"

echo "Copying ${bucket_dir}/data/${dataset} and ${bucket_dir}/tokenizers to local disk..."
mkdir -p "data/${dataset}" tokenizers runs
cp "${bucket_dir}/data/${dataset}/"*.npy "data/${dataset}/"
cp -r "${bucket_dir}/tokenizers/." tokenizers/

copy_runs_to_bucket() {
    echo "Copying runs/ to ${bucket_dir}/runs..."
    mkdir -p "${bucket_dir}/runs"
    cp -r runs/. "${bucket_dir}/runs/"
}
trap copy_runs_to_bucket EXIT

"$@"
