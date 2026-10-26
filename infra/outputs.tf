# Read by scripts/build_image.sh and scripts/submit_vertex_job.sh.

output "project_id" {
    value = var.project_id
}

output "region" {
    value = var.region
}

output "bucket" {
    value = google_storage_bucket.cs336.name
}

output "image" {
    value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.cs336.repository_id}/cs336-train:latest"
}

output "trainer_sa" {
    value = google_service_account.trainer.email
}

output "builder_sa" {
    value = google_service_account.builder.email
}
