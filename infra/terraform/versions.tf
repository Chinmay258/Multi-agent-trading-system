# Terraform + provider versions. The OCI provider talks to Oracle Cloud, whose
# Always-Free tier (ARM Ampere A1) can run this whole stack at no cost, forever.
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 5.0.0"
    }
  }
}

# Auth via the standard OCI CLI config (~/.oci/config). Run `oci setup config` once.
# (Alternatively, set the provider's tenancy_ocid/user_ocid/fingerprint/private_key
#  variables directly — see the OCI provider docs.)
provider "oci" {
  region              = var.region
  config_file_profile = var.config_file_profile
}
