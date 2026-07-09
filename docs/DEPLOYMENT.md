# Deployment

This deploys the **keyless, paper-only** demo to a public HTTPS URL: a single small VM
runs the whole Docker Compose stack, with **Caddy** terminating TLS (free Let's Encrypt
cert) in front of the dashboard + API. **No secrets are required** to deploy.

> 💸 **You will not be charged if you follow the recommended path.** The recommended
> target (Oracle Cloud Always Free) is free *forever*, and a prominent **teardown** is
> documented at the bottom. Always run `make destroy` when you're done experimenting.

```
Internet ──HTTPS──▶ Caddy (:80/:443, auto-cert)
                      └─▶ dashboard (nginx: SPA + proxies /api,/ws)
                            └─▶ api (FastAPI) ──▶ Redis / Postgres / 7 agents (internal only)
```

---

## Recommended: Oracle Cloud Always Free (free forever) ⭐

Oracle's Always-Free tier includes an **ARM Ampere A1** allowance of **4 OCPU / 24 GB RAM**
that never expires — plenty for this stack, unlike the 1 GB micros on other clouds.

### Option A — Terraform (one command)

**Prerequisites (one-time, ~5 min):**
1. Create a free Oracle Cloud account: <https://www.oracle.com/cloud/free/> (needs a card for
   identity verification; Always-Free resources are never charged).
2. Install the OCI CLI and run `oci setup config` → creates `~/.oci/config`. Note your
   **tenancy OCID** and **home region**.
3. Install [Terraform](https://developer.hashicorp.com/terraform/install).

**Deploy:**
```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # fill in region, compartment_ocid, ssh_public_key
make deploy            # = terraform init && terraform apply   (run from repo root: `make deploy`)
```
Terraform provisions a VCN (only 22/80/443 open), an Always-Free ARM VM, and runs the
cloud-init in [`infra/cloud-init.yaml`](../infra/cloud-init.yaml) which installs Docker,
clones the repo, and starts the production stack. **First build takes ~10–15 min** (TA-Lib
and images compile on ARM). Watch progress:
```bash
ssh ubuntu@<public-ip> 'tail -f /var/log/cloud-init-output.log'
```

**Get HTTPS with no domain to buy:** after the first apply, take the output `public_ip`,
set `domain = "<public-ip>.sslip.io"` in `terraform.tfvars`, and `make deploy` again.
`sslip.io` resolves that host to your IP, so Caddy obtains a **real** Let's Encrypt cert.
(Or point your own domain's A-record at the IP and use that.)

Open **`https://<your-domain>`** → the showcase + live paper-trading dashboard.

### If you hit "Out of host capacity" (common!)

Free ARM (A1.Flex) capacity in busy regions is frequently exhausted — `apply` returns
`500 Out of host capacity`. The network is created on the first apply, so only the instance
needs retrying. Two options:

- **Auto-retry** (recommended): `bash infra/oci_retry_apply.sh 180 0` loops `terraform apply`
  every 180s until the instance is created (can take minutes to hours). A smaller shape
  (`instance_ocpus = 1`, `instance_memory_gb = 6`) improves your odds.
- **Console fallback (Option B below).**

### Option B — Console (if ARM capacity is unavailable via API)

If you'd rather not wait, create the VM in the OCI Console (Compute → Instances → Create:
shape `VM.Standard.A1.Flex`, image *Canonical Ubuntu 22.04*, paste
[`infra/cloud-init.yaml`](../infra/cloud-init.yaml) — with `${domain}/${repo_url}/${branch}`
filled in — as **user-data**; open 80/443 in the subnet's security list). Retrying a
different availability domain or region usually succeeds.

---

## Test the production stack locally first (optional, recommended)

Before paying for any cloud time, prove the prod overlay works on your machine:
```bash
make prod-up          # builds + runs base + prod overlay; Caddy on :80/:443 (DOMAIN=localhost)
# open https://localhost  (browser will warn — Caddy uses a local CA for 'localhost')
make prod-down
```
This is the exact stack that runs in the cloud, minus the public cert.

---

## Easier managed alternatives (tradeoffs)

If you'd rather not manage a VM, these work — but each has caveats for an always-on,
stateful, multi-container app:

| Platform | Free? | Good fit? | Notes |
|---|---|---|---|
| **Oracle Always Free** ⭐ | Free forever | **Yes** | This guide. Full stack, always on. Best option. |
| **Fly.io** | Small free allowance | Partial | Deploy each service as a Fly machine + a Fly Postgres/Redis. More wiring; free tier is tight for 10 services. |
| **Render** | Free tier | **No (24/7)** | Free web services **spin down after ~15 min idle** — that kills the live trading loop. Paid tier ($7/mo) avoids it. Good for the *static dashboard* only. |
| **Railway** | ~$5 trial credit | Short-lived | Easiest UX (deploys compose-like), but not free long-term once credit runs out. |
| **GCP Cloud Run / AWS App Runner** | Generous free reqs | **No** | Built for *stateless, scale-to-zero* HTTP services. The agents are long-running stateful loops; scale-to-zero stops them. Not suitable. |
| **GCP e2-micro Always Free** | Free forever | With the lite overlay | 1 shared vCPU / **1 GB** — too small for the full stack. Use [`docker-compose.lite.yml`](../docker-compose.lite.yml) (see below) and expect it to be tight. |
| **Static hosts** (Cloudflare/GitHub/Netlify/Vercel Pages) | Free | Frontend only | Great for the SPA, but the backend (agents + API) still needs a 24/7 host. |

### Lite profile (1 GB micro)
```bash
DOMAIN=<host> docker compose \
  -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.lite.yml up -d --build
```
The lite overlay caps memory and shrinks Redis. It's best-effort — a 1 GB box is genuinely
tight for ten containers. A ≥ 2 GB VM (or the Oracle ARM box) needs no lite overlay.

---

## Cost estimate

| Path | Monthly cost |
|---|---|
| **Oracle Always Free (recommended)** | **$0** (forever, within 4 OCPU / 24 GB / 200 GB) |
| GCP e2-micro Always Free + lite | **$0** (forever, 1 region; egress beyond 1 GB billed) |
| AWS t3.micro | ~$0 for 12 months (free tier), then ~$7–9/mo |
| A small paid VM (DigitalOcean/Hetzner/Linode) | ~$4–6/mo |
| `sslip.io` domain / Let's Encrypt cert | **$0** |

There are **no API keys and no broker fees** — the demo is keyless and paper-only.

---

## 🔻 Teardown (do this when finished)

**Terraform-provisioned infra:**
```bash
make destroy          # = terraform destroy   — removes the VM, network, everything
```
**Console-created VM:** terminate the instance (and its boot volume) in the OCI Console;
optionally delete the VCN.

**Local stack:** `make prod-down` (stop) or `make clean` (stop + remove volumes).

> Removing the VM removes the only thing that could ever bill you. Verify in the cloud
> console that the instance is gone.

---

## Optional: MT5 (local only — never in the cloud)

The public demo never uses MetaTrader 5. If *you* want to run the system locally against
your own MT5 terminal for data or execution, do it on your own machine (not the cloud VM):
set `DATA_SOURCE=mt5` (read-only data) and/or `EXECUTION_BROKER=mt5` in a local `.env`, and
keep your broker login inside the MetaTrader terminal — **never** put credentials in this
repo or the deployed environment. See [`data_sources/mt5_source.py`](../data_sources/mt5_source.py)
and [`agents/execution/mt5_bridge.py`](../agents/execution/mt5_bridge.py).

---

## Verifying the deployment

```bash
curl -I https://<your-domain>/                 # 200 + valid TLS
curl    https://<your-domain>/api/overview      # JSON: trading_mode "paper", agents_healthy
curl    https://<your-domain>/health            # system health
```
Then open `https://<your-domain>` and confirm the **Live Dashboard** shows agents healthy,
a websocket connection, and paper balances updating. Logs on the VM:
```bash
ssh ubuntu@<ip> 'cd /opt/app && docker compose -f docker-compose.yml -f docker-compose.prod.yml ps'
ssh ubuntu@<ip> 'cd /opt/app && docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api dashboard caddy'
```
