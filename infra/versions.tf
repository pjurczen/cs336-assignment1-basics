terraform {
    required_version = ">= 1.6"

    required_providers {
        google = {
            source  = "hashicorp/google"
            version = ">= 6.0, < 8.0"
        }
    }
}

provider "google" {
    project = var.project_id
    region = var.region

    # Calls are made with user credentials (see tf.sh); bill their API quota to this project.
    user_project_override = true
    billing_project       = var.project_id
}
