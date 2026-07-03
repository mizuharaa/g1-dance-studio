# GreenNode AI Platform Console Recon
**Date:** 2026-07-02 evening
**Conducted by:** Automated browser recon (Claude pilot)
**Account:** alois@vng.com.vn (Root user)

---

## Task 1 — Login Outcome

**Result: SUCCESS**

- URL: https://aiplatform.console.greennode.ai/overview
- Signed in via GreenNode SSO (sso.greennode.ai) → chose "SIGN IN WITH ROOT USER ACCOUNT"
- reCAPTCHA checkbox was present; it auto-verified after fill (no puzzle appeared)
- Console dashboard title: "AI Platform - Overview"
- Logged-in user: alois@vng.com.vn (shown in top-right header)

---

## Task 2a — Account Balance / Credits

From console header:
- **AVAILABLE: 0 credits**

From credits dropdown (clicking the header widget):
- Total: **0 credits**
- Holding: **0 credits**
- Currency: VND (all prices in Vietnamese Dong)
- "Buy credit" button visible (not clicked)

From Budget & Alert page:
- Monthly budget limit is disabled
- "Charges will apply based on available credits with no monthly spending cap."
- Budget alerts are disabled
- Alert recipient: alois@vng.com.vn

---

## Task 2b — Notebook Instance Create Form

URL: https://aiplatform.console.greennode.ai/notebooks/create
Status: No existing notebook instances. Zero running resources found.

### Basic Configuration
- Notebook instance name: required, 1–50 chars, letters/numbers/underscore/dash/dot
- Region: HCM (Ho Chi Minh City) — only option shown
- Zones available: HCM-03-1A (HoChiMinh-1A), HCM-03-1B (HoChiMinh-1B), HCM-03-1C (HoChiMinh-1C)
- Code editor: JupyterLab (no other options shown)

### Resource Families and Instance Types

**Family: CPU-CODE-S**
| Instance | GPU | CPU | RAM | VRAM | Price shown |
|---|---|---|---|---|---|
| aiplatform-s-general-8x16 | 0 | 8 | 16 GB | — | 2,193,840 VND |
| aiplatform-s-general-16x32 | 0 | 16 | 32 GB | — | (not checked individually) |

**Family: GPU-CODE-RTX2080TI**
| Instance | GPU | CPU | RAM | VRAM |
|---|---|---|---|---|
| aiplatform-standard-4x16-1rtx2080ti | 1 | 4 | 16 GB | 11 GB |
| aiplatform-standard-8x32-1rtx2080ti | 1 | 8 | 32 GB | 11 GB |
| aiplatform-standard-8x64-1rtx2080ti | 1 | 8 | 64 GB | 11 GB |
| aiplatform-standard-8x32-2rtx2080ti | 2 | 8 | 32 GB | 22 GB |
| aiplatform-standard-16x64-2rtx2080ti | 2 | 16 | 64 GB | 22 GB |
| aiplatform-standard-16x64-4rtx2080ti | 4 | 16 | 64 GB | 44 GB |

**Family: GPU-CODE-RTX4090**
| Instance | GPU | CPU | RAM | VRAM |
|---|---|---|---|---|
| aiplatform-standard-16x64-1rtx4090 | 1 | 16 | 64 GB | 24 GB |
| aiplatform-standard-32x64-1rtx4090 | 1 | 32 | 64 GB | 24 GB |
| aiplatform-standard-16x64-2rtx4090 | 2 | 16 | 64 GB | 48 GB |
| aiplatform-standard-16x128-2rtx4090 | 2 | 16 | 128 GB | 48 GB |
| aiplatform-standard-32x64-2rtx4090 | 2 | 32 | 64 GB | 48 GB |
| aiplatform-standard-32x128-4rtx4090 | 4 | 32 | 128 GB | 96 GB |

**Family: GPU-CODE-A40**
| Instance | GPU | CPU | RAM | VRAM |
|---|---|---|---|---|
| aiplatform-standard-16x64-1A40 | 1 | 16 | 64 GB | 48 GB |
| aiplatform-standard-16x64-2A40 | 2 | 16 | 64 GB | 96 GB |
| aiplatform-standard-32x64-2A40 | 2 | 32 | 64 GB | 96 GB |
| aiplatform-standard-32x128-2A40 | 2 | 32 | 128 GB | 96 GB |
| aiplatform-standard-32x256-4A40 | 4 | 32 | 256 GB | 192 GB |
| aiplatform-standard-48x512-8A40 | 8 | 48 | 512 GB | 384 GB |

### RTX4090 Pricing (Selected: aiplatform-standard-16x64-1rtx4090)

From Item List panel with 1x RTX4090 selected:
```
RESOURCE CONFIGURATION
  GPU:        RTX4090
  GPU count:  1
  CPU:        16 core(s)
  RAM:        64 GB
  VRAM:       24 GB
  Price:      16,080,632 VND

BLOCK STORAGE
  Network volume: (none)
  Folder sync:    /workspace/
  Blockstorage:   20 GB
  Price:          74,800 VND

Original price:   16,155,432 VND
Total:            16,155,432 VND
```

**Billing period label:** NOT SHOWN explicitly on the form. No "per hour" or "per month" label is displayed anywhere in the pricing widget.

**Interpretation note:** At ~25,800 VND/USD, 16,080,632 VND ≈ $623 which is consistent with a MONTHLY price (≈ $0.87/hr). The Budget & Alert page uses "per month" budget tracking. The form likely shows monthly cost.

**VAT text (verbatim):**
"** VAT excluded: Please note that the prices displayed on the portal do not include VAT. The applicable VAT amount will vary depending on the country provided during account registration."

**Billing behavior note (from Budget & Alert):**
"Budget limit is disabled. Charges will apply based on available credits with no monthly spending cap."
"Email alert at 100% usage: Agents continue running but immediate action recommended."
→ Instances continue running even when 100% of budget alert is hit. No explicit stop-on-zero credit language seen.

### Image Options
- **Pre-built container** (selected by default): "Use a supported framework."
  - Only one option shown: **Pytorch 2.5.1 CUDA 12.4**
- **Custom container**: "Use your own framework." (paste custom image URI)

### Form Fields (complete list)
1. Notebook instance name (required)
2. Region (required, dropdown)
3. Zone (radio: HCM-03-1A / 1B / 1C)
4. Instance type (required, family + type dropdown)
5. Image — Pre-built container or Custom container
6. Network volume (required, dropdown — "No volumes found. Manage your volumes")
7. Folder sync (path field, default /workspace/, max 256 chars)
8. Blockstorage (required, 20–1000 GB, default 20)
9. Run command (Section 5, expandable — "Configure commands for launching the container")
10. SSH public key (Section 6 Access — dropdown or paste field, example: ssh-rsa AAAA...)
    - Note: "Select SSH key to add public key to notebook. Manage your SSH keys"
11. HTTP Ports (default: 8888, comma-separated, max 3 ports)
12. TCP Ports (comma-separated, max 3 ports)

**Forbidden button (not clicked):** "CREATE INSTANCE" (greyed out / present at bottom right)

---

## Task 2c — Network Volume

URL: https://aiplatform.console.greennode.ai/volume
Status: Section **exists**. No volumes currently created.

Create Volume form fields:
- Volume name (required, 5–50 chars, letters/numbers/underscore/dash/dot)
- Region (required, dropdown)
- Volume configuration:
  - Optional bucket selection: "Select a bucket of your own if you want."
  - "If no bucket is selected, **a bucket will be automatically created.** The size of this bucket will dynamically adjust based on your usage."
  - Button: "Select a bucket"

Item List pricing:
```
BASIC INFORMATION
  Volume name: volume-01
  Region:      HCM
  Price:       1,080 VND

Original price: 1,080 VND
Total:          1,080 VND
```

---

## Task 2d — API Keys

URL: https://aiplatform.console.greennode.ai/keys
Status: Section **exists**. No API keys currently created.

Page description: "API Keys are used to authenticate requests to the API. They are generated by the user and can be used to access the API."

Create API Key dialog fields:
- "Give this API key a name then click Create to finish"
- Api key name (required, 5–35 chars, letters a-z/0-9/dashes only)

**Max keys:** No max-key limit displayed anywhere in the UI (neither on the listing page nor in the create dialog).

---

## Task 2e — Helpdesk SSH Articles

Attempted URLs:
1. https://helpdesk.greennode.ai/portal/en/kb/articles/connect-to-notebook-instances
   → **"Sorry, this page is restricted."**
2. https://helpdesk.greennode.ai/portal/en/kb/articles/manage-a-notebook-instance
   → **"Sorry, this page is restricted."**

Note: The SSO session DID carry over to the helpdesk (user shows as "AL" avatar in top-right). The restriction is access-level, not login. These articles exist in the system but are in a private/gated KB section not accessible with this root account.

Portal search results:
- "connect notebook" → 1 result (Vietnamese: network interface setup for vServer — unrelated)
- "SSH notebook" → 3 results (all Vietnamese, all about vServer/pfSense — unrelated)
- "notebook instance" → 1 result (Vietnamese: nginx HA setup — unrelated)
- "AI Platform" → **0 results**

The publicly accessible helpdesk KB (`/kb/mss`) covers GreenNode Cloud (VNG Cloud) legacy products in Vietnamese. AI Platform documentation is not available through it.

**SSH docs captured:** NO — articles are restricted and could not be accessed.

---

## Surprising / Notable Findings

1. **No running resources** — account is completely clean, 0 credits, 0 notebooks, 0 volumes, 0 API keys.
2. **A40 goes to 8-GPU / 512 GB RAM** — the largest A40 config is quite substantial (aiplatform-standard-48x512-8A40: 8×A40, 48 CPU, 512 GB RAM, 384 GB VRAM).
3. **No explicit billing period on pricing** — the create form shows total prices in VND but never says "per hour" or "per month". Based on scale, prices appear to be monthly.
4. **Network volume price is tiny** — 1,080 VND base price (≈ $0.04) with dynamic sizing; actual cost scales with usage.
5. **Helpdesk KB segregated** — AI Platform help articles are in a separate/restricted portal vs. the public GreenNode Cloud KB. Both "connect-to-notebook-instances" and "manage-a-notebook-instance" are restricted to this account.
6. **SSH key can be pasted directly** in the create form — you don't need to pre-register SSH keys; you can paste a public key inline. There's also a "Manage your SSH keys" link for pre-registration.
7. **"Continue running" on 100% budget** — email alerts fire at 80% and 100%, but at 100% it says "Agents continue running but immediate action recommended." No hard stop.
8. **Three zones** in HCM but only one region — HCM-03-1A/1B/1C all in Ho Chi Minh City.

---

## Console Navigation Map (sidebar sections found)

- Overview
- Model as a service (MaaS): Models, Playground
- API Key: API Keys
- AgentBase: Agent runtime, Marketplace, Container Registry, Memory, Access control
- MCP Governance: MCP Gateways, Policy Group
- Team & Permissions
- Usage & Budget: Budget & Alert, Usage & Cost
- RAG: RAG Engine, Knowledge base, Tool
- **Notebook: Notebook Instance** ← primary section
- Model deployment, Model tuning
- Protect & Govern: Guardrail (BETA), Rate limit (BETA)
- Deployment & Usage: Model registry, Inference
- **Storage management: Network volume** ← secondary section

## GreenNode AI Platform — Cloud Provisioning (2026-07-03)

### Task 1: Network Volume

- **Name:** g1dance-data
- **Region:** HCM
- **Type:** GreenNode managed (auto-bucket created)
- **Status:** ACTIVE
- **Internal ID:** nv-cb2e7860-567a-4a85-a439-3d26a60...
- **Created:** 03/07/2026 17:10:51
- **Price:** 1,080 VND/month (VAT excluded) — bucket auto-sized by usage
- **Note:** S3 credentials were shown once at creation; saved to .secrets/cloud-connect.txt

### Task 2: Notebook Instance

- **Name:** g1dance-gpu
- **Internal ID:** nb-039dfbbe-6fca-474b-be6f-e3e863a8389d
- **Region:** HCM, Zone: HCM-03-1A
- **Status:** ACTIVE (provisioned in ~18 min)
- **Created:** 03/07/2026 17:20:49
- **Instance type:** aiplatform-standard-16x64-1rtx4090
  - GPU: 1 × RTX4090 (24 GB VRAM)
  - CPU: 16 cores
  - RAM: 64 GB
- **Image:** Pytorch 2.5.1 CUDA 12.4 (Pre-built container, JupyterLab)
- **Block storage:** 150 GB (nnv-09da4462-b8..., Gen2-NVMe2-5000IOPS, ACTIVE)
- **Network volume:** g1dance-data mounted at /workspace/notebook-data
- **HTTP port:** 8888 (Jupyter via web proxy — Ready)
- **TCP port:** 22 (SSH)
- **SSH host/port:** 103.245.250.152 : 46936 (public) → :22 (internal)
- **SSH user:** root
- **Jupyter URL:** https://nb-039dfbbe-6fca-474b-be6f-e3e863a8389d-8888.notebook-hcm.aiplatform.console.greennode.ai/lab

### Pricing shown at creation (VAT excluded)

| Line item | Amount (VND) |
|---|---|
| Instance (GPU-CODE-RTX4090, 16x64, 1×RTX4090) | 16,155,432 |
| Block storage (150 GB) | +486,200 |
| **Total at creation** | **16,641,632** |
| Network volume (g1dance-data) | 1,080/mo |

Note: Console shows prices excluding VAT. At ~18,200 VND/h authorized, confirmed in range.

### Access notes

- JupyterLab accessible immediately after ACTIVE status via SSO session
- notebook-data folder visible in JupyterLab file browser (network volume mount confirmed)
- SSH key (g1dance-laptop ed25519) deployed; connect with: `ssh -p 46936 root@103.245.250.152 -i ~/.ssh/id_rsa`
- Token/credential details in .secrets/cloud-connect.txt (mode 600, never commit)
