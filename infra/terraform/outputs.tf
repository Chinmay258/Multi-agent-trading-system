output "public_ip" {
  description = "Public IP of the VM."
  value       = oci_core_instance.vm.public_ip
}

output "ssh_command" {
  description = "SSH into the VM."
  value       = "ssh ubuntu@${oci_core_instance.vm.public_ip}"
}

output "url" {
  description = "Where the demo will be reachable once Caddy obtains its cert."
  value       = "https://${var.domain}"
}

output "sslip_hint" {
  description = "If you used a sslip.io domain, set var.domain to this and re-apply."
  value       = "${oci_core_instance.vm.public_ip}.sslip.io"
}

output "next_steps" {
  value = <<-EOT
    1. Point your domain's A record at ${oci_core_instance.vm.public_ip}
       (or use the sslip.io host above — no DNS setup needed).
    2. First build on the VM takes ~10-15 min (TA-Lib + images compile on ARM).
       Watch: ssh ubuntu@${oci_core_instance.vm.public_ip} 'tail -f /var/log/cloud-init-output.log'
    3. Open https://${var.domain}  (Caddy fetches the cert on first request).
    4. Tear down everything with:  make destroy   (or  terraform destroy)
  EOT
}
