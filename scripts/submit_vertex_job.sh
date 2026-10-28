#!/usr/bin/env bash
# Submits a Vertex AI custom training job that runs the training image.
#   scripts/submit_vertex_job.sh <cpu|a100> <dataset> <command...>
#   e.g. scripts/submit_vertex_job.sh a100 tinystories-10k scripts/train_tinystories_base.sh --device cuda --run_name a100-base
# The job is named after --run_name if the command has one. Results land in gs://<bucket>/runs/.
# IMAGE_TAG (default: latest) picks an image variant built with the same variable by scripts/build_image.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

machine="$1"
dataset="$2"
shift 2

tf_output() { terraform -chdir=infra output -raw "$1"; }
project="$(tf_output project_id)"
region="$(tf_output region)"
bucket="$(tf_output bucket)"
image="$(tf_output image)"
image="${image%:*}:${IMAGE_TAG:-latest}"
trainer_sa="$(tf_output trainer_sa)"
wandb_secret="$(tf_output wandb_secret)"

case "$machine" in
  cpu) machine_spec=("machineType: n1-standard-4") ;;
  a100) machine_spec=("machineType: a2-highgpu-1g" "acceleratorType: NVIDIA_TESLA_A100" "acceleratorCount: 1") ;;
  *) echo "machine must be cpu or a100, got: $machine" >&2; exit 1 ;;
esac

job_args=("$dataset" "$@")
display_name="cs336-$(date +%Y%m%d-%H%M%S)"
for ((i = 0; i < ${#job_args[@]} - 1; i++)); do
  if [[ "${job_args[i]}" == "--run_name" ]]; then
    display_name="${job_args[i + 1]}"
  fi
done

# A job copies runs/<run_name> back into the bucket when it ends, overwriting a run with the same name.
# ALLOW_EXISTING_RUN=1 skips this check, e.g. when resuming into the same run directory on purpose.
if [ -z "${ALLOW_EXISTING_RUN:-}" ] && gcloud storage ls "gs://${bucket}/runs/${display_name}/" >/dev/null 2>&1; then
  echo "gs://${bucket}/runs/${display_name}/ already exists; pick a new --run_name (or set ALLOW_EXISTING_RUN=1)" >&2
  exit 1
fi

# CustomJobSpec: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec
config="$(mktemp -d)/job.yaml"
{
  echo "workerPoolSpecs:"
  echo "  - replicaCount: 1"
  echo "    machineSpec:"
  printf '      %s\n' "${machine_spec[@]}"
  echo "    containerSpec:"
  echo "      imageUri: ${image}"
  echo "      env:"
  echo "        - name: GCS_BUCKET"
  echo "          value: ${bucket}"
  echo "        - name: WANDB_API_KEY_SECRET"
  echo "          value: ${wandb_secret}"
  echo "      args:"
  for arg in "${job_args[@]}"; do
    # Single-quoted YAML strings: a quote inside is written twice, an empty argument stays ''.
    printf "        - '%s'\n" "${arg//\'/\'\'}"
  done
  echo "serviceAccount: ${trainer_sa}"
} > "$config"
cat "$config"

gcloud ai custom-jobs create \
  --project "$project" \
  --region "$region" \
  --display-name "$display_name" \
  --config "$config"
