#!/usr/bin/env bash
# Submits a Vertex AI custom training job that runs the training image.
#   scripts/submit_vertex_job.sh <cpu|a100> <dataset> <command...>
#   e.g. scripts/submit_vertex_job.sh a100 tinystories-10k scripts/train_tinystories_base.sh --device cuda --run_name a100-base
# The job is named after --run_name if the command has one. Results land in gs://<bucket>/runs/.
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
trainer_sa="$(tf_output trainer_sa)"

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
