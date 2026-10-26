variable "project_id" {
    description = "GCP project to create the training resources in."
    type        = string
}

variable "region" {
    description = "Region of the bucket, the image repository and the training jobs."
    type        = string
    default     = "us-central1"
}
