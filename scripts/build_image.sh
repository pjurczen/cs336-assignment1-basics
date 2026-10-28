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

# The code version goes into the image (GIT_COMMIT, recorded in every run's config.json) and is an extra image tag.
# "-dirty" marks uncommitted changes to tracked files, which the build includes.
git_commit="$(git rev-parse --short HEAD)"
git diff --quiet HEAD -- || git_commit="${git_commit}-dirty"
version_image="${latest_image%:*}:${git_commit}"

echo "Building ${image} and ${version_image}"
gcloud builds submit \
  --project "$project" \
  --region "$region" \
  --config cloudbuild.yaml \
  --substitutions "_IMAGE=${image},_CACHE_IMAGE=${latest_image},_VERSION_IMAGE=${version_image},_GIT_COMMIT=${git_commit}" \
  --service-account "projects/${project}/serviceAccounts/${builder_sa}" \
  --gcs-source-staging-dir "gs://${bucket}/cloudbuild-source" \
  .
