# GreenNode setup — complete step-by-step guide

Written 2026-07-02 from a multi-agent research sweep over GreenNode's live pages, their
official docs (docs.vngcloud.vn — VNG Cloud is GreenNode's parent), web archives, and
release notes. Markers: **✅ verified** = read directly from an official page;
**⚠️ unverified** = not documented publicly, expect to confirm on screen.
If any screen differs from this guide, tell Claude what you see — he adapts live.

**The one-line summary:** create account → top up credit → create a *Network Volume*
(persistent disk) → create the notebook (RTX 4090) → click Connect → hand access to
Claude. About 30–40 minutes total, most of it waiting.

---

## Before you start — 2 things to know

1. **GreenNode has two consoles for the same platform** ✅:
   - **International**: sign up at `register.greennode.ai`, console at
     `aiplatform.console.greennode.ai`, prices in **USD**, top-up by **credit card via
     Stripe**.
   - **Vietnam domestic**: console at `aiplatform.console.vngcloud.vn`, prices in
     **VND** (1 credit = 1 VND), top-up via **MoMo / ZaloPay** wallet gateways.
   This guide follows the **international** path; the notebook screens are the same.
   If you'd rather pay with MoMo/ZaloPay in VND, say so and we'll take the domestic path.
2. **Ask VNG first.** GreenNode *is* VNG's cloud. One internal message may get you a
   company tenant or POC credits (GreenNode has a sales-arranged "Resource POC" program
   ✅ — free proof-of-concept resources, convertible to paid later). There are **no
   automatic free trial credits** for self-service signups ✅.

---

## Part A — Create the account (~10 min + possible activation wait)

1. Open **https://register.greennode.ai/signup**.
2. The form is titled "Create new account" and asks for ✅:
   - **Email** (this becomes your account identity)
   - **Full name**
   - **Create a password** — at least 8 characters with upper-case, lower-case, a
     number and a special character from `!@#$%^&*`
   - **Confirm your password**
   - **Mobile phone** — country-code dropdown, defaults to Vietnam (+84)
   - **Security code** — type the captcha image
   - Tick both checkboxes: *Terms and Conditions* and *Personal Data Protection Policy*
3. Click **Register**. (Shortcut: there's also **"Continue with Google"** ✅ if you
   prefer no password at all.)
4. Check your inbox (and Spam) for an email titled **"GreenNode – Account
   Verification"** ✅ — click the link, complete the final registration form.
5. OTP verification ✅: with a +84 phone number the code arrives **by SMS**; with a
   foreign number or no phone, **by email**. No ID documents/KYC are required ✅.
6. Activation is usually minutes; one (low-quality ⚠️) help page says it can take up
   to 24 h — don't worry if it's not instant.
7. Sign in afterwards via **greennode.ai → Sign In** (SSO at `sso.greennode.ai`,
   Google login supported ✅). Password reset: `register.greennode.ai/resetpwd`.

## Part B — Put money on the account (~5 min)

GreenNode is **prepaid by default** ✅ — you deposit credits, usage draws them down.
(Postpaid/monthly-invoice exists but requires contacting their sales team ✅.)

1. In the console, find the **balance / credit** area (header) or the **Billing**
   section → look for **Deposit / Buy Credits / Top up**.
2. International console: deposits go **via Stripe** (credit card) ✅. Their public
   FAQ: *"we only accept prepaid payments... credit card (Visa, Mastercard) and bank
   transfer"* ✅.
3. Amount: **no minimum is documented** ✅. Start with **$50–100** — enough for
   provisioning plus the first training benchmark. Rough price anchors ✅: RTX 4090
   GPU instance lists at **$610/month** (≈ $0.84/hour); the AI-platform category
   starts at ~33,400 VND/hour (≈ $1.3/hour). The exact hourly price of the 4090
   notebook flavor is **only shown in the console at creation time** ✅ — read it on
   the create screen before clicking (tell Claude the number; it sets our
   cost-per-dance math).
4. A coupon field exists at the payment step ✅ if VNG gives you one.

**⚠️ Billing gotcha to check on screen:** docs say prepaid notebooks are "charged
immediately upon notebook creation" with the remainder refunded **only when the
notebook is deleted** — it is not clear whether *stopping* (not deleting) pauses
prepaid charges. When you reach the create screen, note what the price/billing text
says; Claude will factor it into the on/off strategy.

## Part C — Create a Network Volume FIRST (~3 min) — don't skip

**Why:** the notebook's own disk is **ephemeral — everything on it is LOST when the
instance stops** ✅ (their docs: *"Data stored on the instance's local storage will be
lost when the instance is stopped."*). The **Network Volume** is the persistent disk:
it survives stops, is billed only per GB actually used ✅, and auto-syncs with the
notebook ✅ — on start its contents are copied into a folder inside the notebook (e.g.
`/workspace/notebook-data`); on stop that folder is copied back and **overwrites** the
volume.

1. In the left menu, open **Storage management / Network Volume** → **Create Network
   Volume** ✅.
2. Fields ✅: **Volume Name** (e.g. `g1dance-data`), **Size (GB)** — it auto-adjusts
   to actual usage ✅, **Region: HCM**.
3. Create it. (Bonus: each Network Volume maps to an internal S3 bucket whose access
   keys are shown on its details page ✅ — that gives Claude a second way to move files
   in/out.)

## Part D — Create the notebook (~5 min + boot time)

1. Left menu → **Notebook instance** → click **Create Notebook** ✅.
2. Fill the form ✅:
   - **Notebook instance name**: e.g. `g1dance-gpu` (letters, digits, `_ - .` only,
     1–50 chars)
   - **Region**: HCM (currently the only option)
   - **Resource configuration**: pick the family **`GPU-CODE-RTX4090`** ("Instances
     with NVIDIA RTX 4090 GPU"), then the instance type (it shows GPU/CPU/RAM —
     pick the 1-GPU type; **note the hourly price shown here** and tell Claude)
   - **Image**: there is exactly one — **PyTorch 2.5.1 CUDA 12.4** ✅ (no choice to make)
   - **Data mount**: select your Network Volume from Part C; **Mount folder name**
     e.g. `/workspace/notebook-data`; **Block storage size**: **150 GB**
     (allowed 20–1000 ✅; must be larger than the Network Volume's current size ✅ —
     Isaac Lab and checkpoints are huge, don't go small; it can be grown later via
     "Resize" but never shrunk ✅)
3. There is **no SSH-key field** on this form ✅ — that's normal, ignore that earlier
   suggestion of mine.
4. Click **Create Instance**. The notebook **starts automatically** ✅ and begins
   billing while running ✅. Wait for status **Active**.

## Part E — Connect and hand over to Claude (~5 min)

1. When status is **Active**, click **Connect** (via the instance name or the Action
   column) — it opens the **Jupyter** web interface ✅.
2. **Look at what the Connect option offers.** GreenNode supports three connection
   methods ✅: **Code Editor** (in-browser), **TCP Port**, and **SSH** (their SSH
   how-to is login-gated, so the exact dialog is ⚠️). What to do:
   - If you see **SSH or TCP-port connection details** (a host/port/command/key):
     copy ALL of it to Claude — that's the best hands-off channel.
   - **Either way, also do this** (guaranteed fallback): in the Jupyter page, open
     **File → New → Terminal**, paste this single line, and press Enter:

     ```bash
     bash -c 'D=${NB_DATA:-/workspace/notebook-data}; [ -d "$D" ] || D=$HOME; mkdir -p "$D/bin"; [ -x "$D/bin/cloudflared" ] || (curl -fsSL -o "$D/bin/cloudflared" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x "$D/bin/cloudflared"); T=$(python3 -c "import secrets;print(secrets.token_hex(16))"); (jupyter server --ip=127.0.0.1 --port=8899 --ServerApp.token="$T" --no-browser >/tmp/g1jup.log 2>&1 &); sleep 4; echo; echo "=== COPY THIS TOKEN: $T"; "$D/bin/cloudflared" tunnel --url http://127.0.0.1:8899 --no-autoupdate 2>&1 | grep --line-buffered -o "https://[a-z0-9-]*\.trycloudflare\.com"'
     ```

     After ~10 seconds it prints a **TOKEN** line and a **https://…trycloudflare.com**
     URL. Paste BOTH into the app: *Cloud GPU → Connection settings → Jupyter/tunnel
     URL (plan B)*, then click **Save & test** — the status dot should turn green.
     Keep that terminal tab open (closing it closes the tunnel; the tunnel dies
     anyway when the instance stops, which is fine — re-paste to reconnect).
3. Note: the Jupyter session is reached through your logged-in console ✅ (notebooks
   have **no public IP** ✅), so a copy-pasted browser URL alone may not work from
   outside — that's why step 2 matters.
4. After the channel is up, **everything else is Claude's job**: installing GVHMR and
   Isaac Lab (or the mjlab fallback), wiring Weights & Biases, running the first
   training benchmark, and syncing results to the laptop.

## Part F — Running costs and the on/off routine

- **Stop the instance when Claude says a work session is done** — charges accrue
  while running ✅. Start/Stop buttons are on the instance row / detail page ✅.
- **Never store anything important outside the mounted folder** — only
  `/workspace/notebook-data` (the Network Volume) survives a stop ✅. Claude will keep
  all installs re-creatable by script and all artifacts inside the mount.
- On restart the machine may get a **new IP and new SSH host keys** ✅ — reconnection
  details may change; expected, not a problem.
- Long trainings (a 2–3 minute dance may run overnight) need the instance **on** the
  whole time; jobs run under tmux so your laptop can sleep/reboot freely.
- The platform has **automatic on/off schedules** for notebooks (added Aug 2025 ✅)
  — useful later for routine training windows.
- **Delete** (vs Stop) is unrecoverable ✅ — only ever delete when Claude confirms
  everything is on the Network Volume or synced to the laptop.

## Known limitations found during research (so you're not surprised)

- **No public API/CLI exists to create or manage notebooks** ✅ — the console is the
  only way, which is why Parts A–E need your hands. (Their only public API is model
  inference, unrelated to us.)
- The **helpdesk knowledge base went behind a login** in mid-2026 ✅; the same content
  lives publicly at `docs.vngcloud.vn` (→ AI Stack → AI Platform), which is what this
  guide is built from.
- The fixed **PyTorch 2.5.1 / CUDA 12.4** image can't be swapped ✅ — if Isaac Lab
  2.1.0 fights that environment, we fall back to **mjlab** (already planned in the
  architecture as the bounded fallback).

## Main sources

- Notebook create/manage/connect (official docs): docs.vngcloud.vn → ai-stack →
  ai-platform → notebook-instance (+ /notebook-instance-management, /network-volume,
  /pricing, /getting-started-with-ai-platform)
- Live signup form: register.greennode.ai/signup
- Prices: greennode.ai/product/gpu-instances, greennode.ai/pricing
- SSH announcement: greennode.ai/blog/greennode-updates-cpu-instances
- Signup/OTP flow: docs.vngcloud.vn → getting-start-with-vng-cloud-account/register
