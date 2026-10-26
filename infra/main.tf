locals {
    apis = [
        "aiplatform.googleapis.com",
        "artifactregistry.googleapis.com",
        "cloudbuild.googleapis.com",
        "iam.googleapis.com",
        "logging.googleapis.com",
        "storage.googleapis.com",
    ]
}

# The project is shared with other services, so never turn these APIs off on destroy.
resource "google_project_service" "apis" {
    for_each = toset(local.apis)
    service            = each.value
    disable_on_destroy = false
}

# Training data in, runs out. The layout mirrors the repo: data/<dataset>/*.npy and runs/<run_name>/.
resource "google_storage_bucket" "cs336" {
    name                        = "${var.project_id}-cs336"
    location                    = var.region
    uniform_bucket_level_access = true
    public_access_prevention = "enforced"
    # Keeps `terraform destroy` from deleting training runs.
    force_destroy               = false

    # Cloud Build uploads the source here on every build; old uploads are never read again.
    lifecycle_rule {
        condition {
            age = 7
            matches_prefix = ["cloudbuild-source/"]
        }
        action {
            type = "Delete"
        }
    }

    depends_on = [google_project_service.apis]
}

resource "google_artifact_registry_repository" "cs336" {
    repository_id = "cs336"
    location      = var.region
    format        = "DOCKER"
    description = "CS336 training images"

    # Images are ~6 GB each: delete versions older than 7 days, except the 3 most recent.
    cleanup_policy_dry_run = false
    cleanup_policies {
        id     = "delete-older-than-7-days"
        action = "DELETE"
        condition {
            tag_state  = "ANY"
            older_than = "604800s"
        }
    }
    cleanup_policies {
        id     = "keep-3-most-recent"
        action = "KEEP"
        most_recent_versions {
            keep_count = 3
        }
    }

    depends_on = [google_project_service.apis]
}

# Identity of the Vertex AI training jobs.
resource "google_service_account" "trainer" {
    account_id   = "cs336-trainer"
    display_name = "CS336 Vertex AI training jobs"
    depends_on = [google_project_service.apis]
}

# Identity of Cloud Build when it builds the training image.
resource "google_service_account" "builder" {
    account_id   = "cs336-builder"
    display_name = "CS336 training image builds"
    depends_on = [google_project_service.apis]
}

# Jobs read data from and write runs to the bucket through the /gcs mount, which also needs bucket metadata.
resource "google_storage_bucket_iam_member" "trainer_objects" {
    bucket = google_storage_bucket.cs336.name
    role   = "roles/storage.objectAdmin"
    member = "serviceAccount:${google_service_account.trainer.email}"
}

resource "google_storage_bucket_iam_member" "trainer_bucket_read" {
    bucket = google_storage_bucket.cs336.name
    role   = "roles/storage.legacyBucketReader"
    member = "serviceAccount:${google_service_account.trainer.email}"
}

resource "google_artifact_registry_repository_iam_member" "trainer_pull" {
    repository = google_artifact_registry_repository.cs336.name
    location   = google_artifact_registry_repository.cs336.location
    role       = "roles/artifactregistry.reader"
    member     = "serviceAccount:${google_service_account.trainer.email}"
}

# Builds read the uploaded source from the bucket, push the image and write build logs.
resource "google_storage_bucket_iam_member" "builder_source" {
    bucket = google_storage_bucket.cs336.name
    role   = "roles/storage.objectViewer"
    member = "serviceAccount:${google_service_account.builder.email}"
}

resource "google_artifact_registry_repository_iam_member" "builder_push" {
    repository = google_artifact_registry_repository.cs336.name
    location   = google_artifact_registry_repository.cs336.location
    role       = "roles/artifactregistry.writer"
    member     = "serviceAccount:${google_service_account.builder.email}"
}

resource "google_project_iam_member" "builder_logs" {
    project = var.project_id
    role    = "roles/logging.logWriter"
    member  = "serviceAccount:${google_service_account.builder.email}"
}
