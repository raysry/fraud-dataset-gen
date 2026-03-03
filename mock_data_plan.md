# Mock Data Generator for Donation Fraud Detection — Plan

## Overview

A Python script that generates realistic donation data for a cloud storage tipping platform, designed to test Louvain-based fraud ring detection. Gray-industry operators control batches of fake accounts that donate to their paying customers' accounts.

## File

`gen_donation_fraud_data.py` (this directory)

## Output Files (same directory as script by default)

- `donations.csv[.gz]` — all donation records with opaque IDs
- `labels.json` — ground truth mapping

## CLI Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `-n` / `--total` | `10_000_000` | Total donation records |
| `--preset` | none | `small`=1M, `medium`=10M, `large`=50M (overrides `-n`) |
| `--fraud-pct` | `1.5` | Target fraud percentage; ring count and sizes auto-scale to hit this |
| `--normal-users` | `500_000` | Normal user pool size |
| `--recipients` | `800` | Popular recipient pool |
| `--seed` | `42` | Random seed for reproducibility |
| `--start-date` | `2025-01-01` | Time window start |
| `--end-date` | `2025-12-31` | Time window end |
| `--output-dir` | `.` (script dir) | Where to write output |
| `--compress` | `false` | Write `donations.csv.gz` instead of plain CSV |
| `--multi-seed` | `1` | Generate N datasets with seeds `seed, seed+1, …, seed+N-1` (output dirs `run_0/`, `run_1/`, …) |
| `--label-noise` | `0` | Percentage of labels to flip (e.g. `0.5` = 0.5% normal↔fraud swaps in `labels.json`) |

## CSV Columns

All IDs are opaque (`u_0000001` format). Only `labels.json` reveals roles.

| Column | Type | Description |
|--------|------|-------------|
| `donation_id` | `d_00000001` | Unique record ID |
| `donor_id` | `u_0000001` | Opaque user ID |
| `recipient_id` | `u_0000001` | Opaque user ID |
| `amount` | float | ¥, 2 decimal places, range [0.10, 2000.00] |
| `timestamp` | ISO 8601 | Second precision, CST |
| `device_type` | str | `mobile` / `pc` / `api` |
| `ip_hash` | `ip_xxxxx` | Hashed IP identifier |
| `session_seconds` | int | Time-on-page before donating |
| `account_age_days` | int | Donor's account age at donation time |
| `account_reg_date` | `YYYY-MM-DD` | Donor's registration date |
| `is_refund` | bool | `true` if this record is a refund reversal |

## Entity Design

**ID allocation** — single opaque `u_` namespace, shuffled so fraud/normal IDs are interleaved.

### Fraud Rings (auto-scaled to hit `--fraud-pct`)

Typical: 8–15 rings, each with 50–300 fakes and 5–20 customers.

Ring count and per-ring sizes are randomly sampled, then scaled so total fraud donations ≈ target percentage of `--total`.

Each ring is assigned a **difficulty tier**:

| Tier | ~Distribution | Characteristics |
|------|---------------|-----------------|
| **Easy** (~30%) | Tight burst timing, identical amounts, all-fresh accounts, single device type |
| **Medium** (~45%) | Some timing spread, jittered amounts, mixed account ages, 1–2 device types |
| **Hard** (~25%) | Organic-like timing, diverse amounts, warm-up cover activity, stealth fakes, mixed devices, delayed activation |

Each ring is assigned a **business model** (fraud type):

| Model | ~Distribution | Amount Profile |
|-------|---------------|----------------|
| **Volume brushing** (~50%) | High frequency, small amounts (¥0.1–¥10), goal is inflating donation count |
| **Chart topping** (~30%) | Medium frequency, large amounts (¥50–¥2000), goal is pushing recipients up leaderboards |
| **Laundering** (~20%) | Low frequency, varied amounts, irregular timing, goal is moving money through the platform |

Business model is independent of difficulty tier — an easy ring can be any type.

### Normal Users (`--normal-users`)

- Donation count per user: power-law (most 1–10, some power users 50–500)
- ~0.5% **suspicious normals**: high-volume donors to few recipients — natural false-positive bait
- **Fan clubs** (~3–5 clubs): 200–2000 normal users who heavily overlap on the same 1–3 top creators, creating tight non-fraud communities that Louvain will detect — tests whether downstream analysis can distinguish real fan communities from fraud rings

### Recipients (`--recipients`)

- Shared pool of popular content creators
- Some are also fraud ring customers (organic donations mix with fraud — realistic overlap)
- ~5% are **cross-donor-recipients**: creators who also donate to other creators

## Behavior Patterns

### Temporal

- **Diurnal cycle**: activity peaks 10am–11pm CST, dips 2–6am
- **Weekday/weekend**: weekends +30% activity
- **Holiday spikes**: Chinese New Year, 618, Double 11, National Day — 2–3× normal volume
- **Fraud operation windows**: each ring picks 3–8 burst windows (2–6 hour blocks on specific days); ~85% of ring's fraud lands in these windows, rest scattered as cover
- **Ring lifecycle**: ~20% of rings go dormant for 1–3 months mid-year, then reactivate with some fresh accounts
- **Rate limit compliance**: max 5 donations per donor per minute (operators pace scripts accordingly)

### Amounts

- **Normal users**: log-normal base, ~40% snapped to culturally significant round numbers (1, 5, 10, 20, 50, 66, 88, 100, 200, 520, 1314), platform limits ¥0.10–¥2000
- **Fraud per ring**: operator picks 2–4 template amounts + jitter ±5–15%; easy rings use fixed values, hard rings sample more diversely
- **Ramp-up**: operators start with small test donations (¥0.1–¥1) for the first 5–10% of a ring's activity, then scale to target amounts

### Account Registration

- **Fake accounts**: registered in batches — within a ring, reg dates cluster in 1–3 tight windows (2–7 day spans)
- **Normal users**: reg dates spread uniformly over a wider historical range (1–1500 days before `--start-date`)
- `account_age_days` is computed per-donation as `donation_date - reg_date`

### Device & IP

- **Fraud rings**: each ring uses 1–2 device types (skew `api` or `pc`); shares a pool of 3–8 proxy IPs (some IP overlap between rings from same VPN provider)
- **Normal users**: device mix ~60% mobile / 30% pc / 10% api; each user has 1–3 IPs from a large pool (~50k unique)
- **Session seconds**: bots 1–5s; normal humans 10–300s (log-normal); hard-tier fakes artificially inflate to 8–60s

### Cross-Ring Contamination

- ~5% of fake accounts shared across 2 rings (account resale)
- ~10% of fraud customers appear in 2+ rings (bought from multiple operators)
- **Incidental cross-ring donations** (~1–3% of fakes): occasional 1–2 small donations to customers of *other* rings or random recipients — simulates operators testing accounts or competing for the same clients; creates weak inter-community bridges that bleed modularity and prevent overly clean Louvain splits

### Recipient Overlap

- Fraud customers also receive organic donations from normal users (otherwise they'd be trivially flagged)
- Organic-to-fraud ratio on customer accounts: ~30–70% organic depending on the customer's real popularity

### Noise & Edge Cases

- **Retry duplicates** (~0.5%): near-identical records — same donor→recipient, same amount, within 1–60s
- **Refunds**: baseline ~1–2% random across all donations; additionally, ~20% of rings employ **strategic refund evasion** — these rings refund 20–30% of their donations (often the large ones, often within minutes), creating graph edges with weak net-amount signal
- **Warm-up** (~30% of fakes): 1–3 small donations to random popular recipients before fraud begins
- **Cover donations** (~20% of fakes): 1–5 donations sprinkled to popular recipients throughout the year
- **Stealth fakes** (~2% of all fakes): mimic normal timing, amounts, and session durations — labeled in `labels.json` as `stealth_fakes`
- **Delayed activation** (hard-tier only, ~10% of hard-tier fakes): accounts behave 100% normally for 2–4 months (organic donations, normal timing/amounts), then begin participating in ring fraud — defeats time-window-based graph construction and rolling-window Louvain

## Labels JSON Structure

```json
{
  "fraud_rings": [
    {
      "ring_id": 0,
      "difficulty": "easy|medium|hard",
      "business_model": "volume_brushing|chart_topping|laundering",
      "refund_strategy": "normal|evasion",
      "fake_accounts": ["u_0003201", ...],
      "customer_accounts": ["u_0000045", ...],
      "shared_fakes_from_other_rings": ["u_0004102"],
      "delayed_activation_fakes": ["u_0003205", ...],
      "operation_windows": [
        ["2025-03-15T02:00:00", "2025-03-15T06:00:00"], ...
      ]
    }
  ],
  "stealth_fakes": ["u_0000088", ...],
  "delayed_activation_fakes": ["u_0000089", ...],
  "suspicious_normals": ["u_0012345", ...],
  "fan_clubs": [
    {"recipients": ["u_0500010"], "members": ["u_0012000", "u_0012001", ...]}
  ],
  "cross_donor_recipients": ["u_0000102", ...],
  "normal_users": ["u_0000001", ...],
  "recipients": ["u_0500001", ...],
  "stats": {
    "total_donations": 10000000,
    "total_refunds": 150000,
    "total_retries": 50000,
    "fraud_donations": 150000,
    "fraud_pct": 1.5,
    "num_rings": 12,
    "num_fake_accounts": 1800,
    "num_customer_accounts": 140,
    "num_normal_users": 500000,
    "num_recipients": 800,
    "difficulty_distribution": {"easy": 4, "medium": 5, "hard": 3},
    "business_model_distribution": {"volume_brushing": 6, "chart_topping": 4, "laundering": 2},
    "rings_with_refund_evasion": 3,
    "num_fan_clubs": 4,
    "label_noise_pct": 0.0
  }
}
```

## Implementation Details

- **Dependencies**: `numpy`, `argparse`, `csv`, `json`, `gzip`, `datetime` — no pandas
- **Memory**: generate and write in chunks (~500k rows); never hold full 10M in memory
- **Progress**: print progress every chunk + ETA
- **Reproducibility**: all randomness via `numpy.random.Generator` seeded from `--seed`
- **Verification**: after writing, print summary stats and validate fraud percentage
- **Quality metrics log**: print per-ring intra-ring edge density (fake→customer edges / possible edges), mean node degree for fakes vs normals, and fan-club community sizes — sanity check that generated data has detectable but non-trivial structure

## Run

```bash
python gen_donation_fraud_data.py                          # 10M default
python gen_donation_fraud_data.py --preset small           # 1M quick test
python gen_donation_fraud_data.py --preset large --compress # 50M, gzipped
python gen_donation_fraud_data.py -n 20000000 --fraud-pct 2.0 --seed 123
python gen_donation_fraud_data.py --multi-seed 10                       # 10 datasets for statistical robustness
python gen_donation_fraud_data.py --label-noise 0.5                     # 0.5% label flips
```
