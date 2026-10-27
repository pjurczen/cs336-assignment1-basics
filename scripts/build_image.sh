#!/usr/bin/env bash
# Builds the training image in Cloud Build and pushes it to Artifact Registry.
# Only the source is uploaded (filtered by .gitignore); torch and the CUDA libraries are downloaded inside GCP.
# The working tree is built as is, uncommitted changes included. IMAGE_TAG (default: latest) names a variant,
# e.g. for an ablation:  IMAGE_TAG=no-rmsnorm scripts/build_image.sh
set -euo pipefail
cd "$(dirname "$0")/.."

tf_output() { terraform -chdir=infra output -raw "$1"; }
project="$(tf_output project_id)"
region="$(tf_output region)"
bucket="$(tf_output bucket)"
latest_image="$(tf_output image)"
image="${latest_image%:*}:${IMAGE_TAG:-latest}"
builder_sa="$(tf_output builder_sa)"

echo "Building ${image}"
gcloud builds submit \
  --project "$project" \
  --region "$region" \
  --config cloudbuild.yaml \
  --substitutions "_IMAGE=${image},_CACHE_IMAGE=${latest_image}" \
  --service-account "projects/${project}/serviceAccounts/${builder_sa}" \
  --gcs-source-staging-dir "gs://${bucket}/cloudbuild-source" \
  .
