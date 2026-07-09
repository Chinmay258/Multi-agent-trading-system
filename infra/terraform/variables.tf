variable "region" {
  description = "OCI region, e.g. ap-mumbai-1, us-ashburn-1, eu-frankfurt-1."
  type        = string
}

variable "compartment_ocid" {
  description = "OCID of the compartment to deploy into (often your tenancy/root OCID)."
  type        = string
}

variable "config_file_profile" {
  description = "Profile name in ~/.oci/config used for auth."
  type        = string
  default     = "DEFAULT"
}

variable "ssh_public_key" {
  description = "SSH public key (contents, not path) for the 'ubuntu' user."
  type        = string
}

variable "domain" {
  description = <<-EOT
    Hostname Caddy will obtain HTTPS for. Use a real domain pointed at the VM's
    public IP, OR a free wildcard-DNS host like "<IP>.sslip.io" (resolves to the IP,
    gets a real Let's Encrypt cert, no domain purchase needed). Leave as a sslip.io
    placeholder and update after first apply once you know the IP.
  EOT
  type        = string
  default     = "CHANGEME.sslip.io"
}

variable "repo_url" {
  description = "Git URL of the project to clone on the VM."
  type        = string
  default     = "https://github.com/Chinmay258/Multi-agent-trading-system.git"
}

variable "branch" {
  description = "Git branch to deploy."
  type        = string
  default     = "main"
}

# Shape. Two Always-Free options:
#   - "VM.Standard.A1.Flex"      ARM Ampere (up to 4 OCPU / 24 GB) — best, but capacity-
#                                constrained in busy regions ("out of host capacity").
#   - "VM.Standard.E2.1.Micro"   AMD x86, fixed 1 OCPU / 1 GB — always available, but tiny
#                                (use with use_lite=true + the swap the cloud-init adds).
variable "instance_shape" {
  description = "Compute shape."
  type        = string
  default     = "VM.Standard.A1.Flex"
}

# Only used for Flex shapes (ignored for the fixed E2.1.Micro).
variable "instance_ocpus" {
  description = "OCPUs for Flex shapes (Always-Free ARM budget is 4)."
  type        = number
  default     = 2
}

variable "instance_memory_gb" {
  description = "Memory in GB for Flex shapes (Always-Free ARM budget is 24)."
  type        = number
  default     = 12
}

# On a 1 GB micro, run the trimmed 'lite' compose overlay.
variable "use_lite" {
  description = "Add docker-compose.lite.yml (memory-tight profile) on the VM."
  type        = bool
  default     = false
}

variable "boot_volume_gb" {
  description = "Boot volume size in GB (Always-Free includes up to 200 GB total)."
  type        = number
  default     = 50
}
