#!/usr/bin/env bash
# Runs terraform in infra/ as the active gcloud account. Without this the provider would use
# $GOOGLE_APPLICATION_CREDENTIALS (a low-privilege service account key). The token lasts about an hour.
#   infra/tf.sh plan
#   infra/tf.sh apply
set -euo pipefail
cd "$(dirname "$0")"

token="$(gcloud auth print-access-token)"
exec env -u GOOGLE_APPLICATION_CREDENTIALS GOOGLE_OAUTH_ACCESS_TOKEN="$token" terraform "$@"
