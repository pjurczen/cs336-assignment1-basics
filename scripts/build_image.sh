#!/usr/bin/env bash
# Builds the training image in Cloud Build and pushes it to Artifact Registry.
# Only the source is uploaded (filtered by .gitignore); torch and the CUDA libraries are downloaded inside GCP.
set -euo pipefail
cd "$(dirname "$0")/.."

tf_output() { terraform -chdir=infra output -raw "$1"; }
project="$(tf_output project_id)"
region="$(tf_output region)"
bucket="$(tf_output bucket)"
image="$(tf_output image)"
builder_sa="$(tf_output builder_sa)"

gcloud builds submit \
  --project "$project" \
  --region "$region" \
  --config cloudbuild.yaml \
  --substitutions "_IMAGE=${image}" \
  --service-account "projects/${project}/serviceAccounts/${builder_sa}" \
  --gcs-source-staging-dir "gs://${bucket}/cloudbuild-source" \
  .
