# =============================================================================
# Single Always-Free ARM VM running the dockerized stack, fronted by Caddy HTTPS.
# =============================================================================
# Network: a minimal VCN + public subnet + internet gateway, with a security list
# that opens only 22/80/443. Compute: one VM.Standard.A1.Flex (ARM, Always-Free)
# booting Ubuntu 22.04 with the cloud-init in ../cloud-init.yaml.
# =============================================================================

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.compartment_ocid
}

# Latest Canonical Ubuntu 22.04 image matching the chosen shape (ARM or x86).
data "oci_core_images" "ubuntu" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "22.04"
  shape                    = var.instance_shape
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

resource "oci_core_vcn" "vcn" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = ["10.20.0.0/16"]
  display_name   = "trading-vcn"
  dns_label      = "tradingvcn"
}

resource "oci_core_internet_gateway" "igw" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.vcn.id
  display_name   = "trading-igw"
}

resource "oci_core_route_table" "rt" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.vcn.id
  display_name   = "trading-rt"
  route_rules {
    destination       = "0.0.0.0/0"
    network_entity_id = oci_core_internet_gateway.igw.id
  }
}

resource "oci_core_security_list" "sl" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.vcn.id
  display_name   = "trading-sl"

  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }

  # SSH, HTTP, HTTPS only.
  dynamic "ingress_security_rules" {
    for_each = [22, 80, 443]
    content {
      protocol = "6" # TCP
      source   = "0.0.0.0/0"
      tcp_options {
        min = ingress_security_rules.value
        max = ingress_security_rules.value
      }
    }
  }
}

resource "oci_core_subnet" "subnet" {
  compartment_id             = var.compartment_ocid
  vcn_id                     = oci_core_vcn.vcn.id
  cidr_block                 = "10.20.1.0/24"
  display_name               = "trading-subnet"
  route_table_id             = oci_core_route_table.rt.id
  security_list_ids          = [oci_core_security_list.sl.id]
  prohibit_public_ip_on_vnic = false
  dns_label                  = "tradingsub"
}

resource "oci_core_instance" "vm" {
  compartment_id      = var.compartment_ocid
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = "trading-demo"
  shape               = var.instance_shape

  # shape_config only applies to Flex shapes; the fixed E2.1.Micro rejects it.
  dynamic "shape_config" {
    for_each = can(regex("Flex", var.instance_shape)) ? [1] : []
    content {
      ocpus         = var.instance_ocpus
      memory_in_gbs = var.instance_memory_gb
    }
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.subnet.id
    assign_public_ip = true
  }

  source_details {
    source_type             = "image"
    source_id               = data.oci_core_images.ubuntu.images[0].id
    boot_volume_size_in_gbs = var.boot_volume_gb
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data = base64encode(templatefile("${path.module}/../cloud-init.yaml", {
      domain    = var.domain
      repo_url  = var.repo_url
      branch    = var.branch
      lite_flag = var.use_lite ? "-f docker-compose.lite.yml" : ""
    }))
  }
}
