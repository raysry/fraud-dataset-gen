# Donation Fraud Data Generator

A Python script that generates realistic donation data, designed to test fraud ring detection.

## Overview

This tool creates synthetic donation datasets with embedded fraud rings of varying difficulty levels. Fraud operators control batches of fake accounts that donate to paying customers' accounts, with realistic temporal patterns, amount distributions, and behavioral characteristics.

## Features

- **Configurable dataset size**: from 1M to 50M+ donations
- **Tunable fraud percentage**: auto-scaled ring counts and sizes to hit target fraud rate
- **Multiple fraud types**: volume brushing, chart topping, laundering
- **Difficulty tiers**: easy, medium, and hard rings with progressively sophisticated patterns
- **Realistic behavior**: diurnal cycles, holiday spikes, account warm-up, refund strategies
- **Ground truth labels**: JSON file mapping all entities (fakes, customers, normal users, fans, rings)
- **Memory efficient**: streaming writes in chunks, never loads full dataset into memory
- **Reproducible**: seeded randomness for consistent results

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/gen_donation_fraud.git
cd gen_donation_fraud

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# 10M donations with 1.5% fraud (default)
python gen_donation_fraud_data.py

# Quick test: 1K donations
python gen_donation_fraud_data.py --preset small

# Large dataset: 50M donations, gzip compressed
python gen_donation_fraud_data.py --preset large --compress

# Custom parameters
python gen_donation_fraud_data.py -n 20000000 --fraud-pct 2.0 --seed 123

# Generate 10 datasets for statistical robustness
python gen_donation_fraud_data.py --multi-seed 10

# Add 0.5% label noise to test robustness
python gen_donation_fraud_data.py --label-noise 0.5
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-n, --total` | `10000000` | Total donation records |
| `--preset` | - | `small` (1K), `medium` (10M), `large` (50M) — overrides `-n` |
| `--fraud-pct` | `1.5` | Target fraud percentage |
| `--normal-users` | `500000` | Normal user pool size |
| `--recipients` | `800` | Popular recipient pool |
| `--seed` | `42` | Random seed for reproducibility |
| `--start-date` | `2025-01-01` | Time window start (YYYY-MM-DD) |
| `--end-date` | `2025-12-31` | Time window end (YYYY-MM-DD) |
| `--output-dir` | `.` | Output directory |
| `--compress` | `false` | Write gzip-compressed CSV |
| `--multi-seed` | `1` | Generate N datasets with sequential seeds |
| `--label-noise` | `0` | Flip this % of labels in output |

## Output Files

- `donations.csv` (or `.csv.gz`) — donation records with opaque IDs
- `labels.json` — ground truth: fraud rings, fake accounts, customers, normal users, fan clubs, etc.

### CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| `donation_id` | `d_00000001` | Unique record ID |
| `donor_id` | `u_0000001` | Opaque user ID |
| `recipient_id` | `u_0000001` | Opaque user ID |
| `amount` | float | Donation amount (CNY) |
| `timestamp` | ISO 8601 | Donation time (second precision, CST) |
| `device_type` | str | `mobile`, `pc`, or `api` |
| `ip_hash` | `ip_xxxxx` | Hashed IP identifier |
| `session_seconds` | int | Time-on-page before donating |
| `account_age_days` | int | Donor's account age at donation |
| `account_reg_date` | YYYY-MM-DD | Donor's registration date |
| `is_refund` | bool | True if this record is a refund reversal |

### labels.json Structure

Maps all entities to roles:
- `fraud_rings[]` — ring ID, difficulty, business model, fake accounts, customers, operation windows
- `stealth_fakes[]` — accounts mimicking normal behavior
- `delayed_activation_fakes[]` — accounts that behave normally before joining fraud
- `suspicious_normals[]` — legitimate high-volume donors (false-positive bait)
- `fan_clubs[]` — tight communities of normal users (tests Louvain discrimination)
- `cross_donor_recipients[]` — accounts that both donate and receive
- `normal_users[]` — all normal user IDs
- `recipients[]` — all recipient IDs
- `stats{}` — aggregate statistics

## Data Characteristics

### Fraud Rings (auto-scaled)
- **Count**: 8–15 rings per 10M donations
- **Sizes**: 50–300 fake accounts, 5–20 customer accounts per ring
- **Overlap**: ~5% of fakes shared across rings; ~10% of customers in multiple rings

### Difficulty Distribution
- **Easy** (~30%): tight timing, identical amounts, fresh accounts, single device type
- **Medium** (~45%): timing spread, jittered amounts, mixed ages, 1–2 device types
- **Hard** (~25%): organic-like timing, diverse amounts, warm-up activity, stealth fakes, delayed activation

### Business Models
- **Volume brushing** (~50%): high frequency, small amounts (¥0.1–¥10)
- **Chart topping** (~30%): medium frequency, large amounts (¥50–¥2000)
- **Laundering** (~20%): low frequency, varied amounts, irregular timing

### Realistic Patterns
- **Diurnal**: activity peaks 10am–11pm CST, dips 2–6am
- **Weekends**: +30% activity vs weekdays
- **Holidays**: 2–3× volume on Chinese New Year, 618, Double 11, National Day
- **Temporal spikes**: rings operate in 2–6 hour burst windows on specific days
- **Account warm-up**: ~30% of fakes make 1–3 organic-looking donations before fraud
- **Refunds**: baseline ~1–2% across all donations; ~20% of rings employ evasion tactics
- **Retry duplicates**: ~0.5% near-identical records (same donor→recipient, within 60s)

## Requirements

- Python 3.7+
- numpy

## Performance

- Generates and writes in 500k-row chunks
- Memory usage stays constant regardless of dataset size
- Progress indicators with ETA
- Example times (on modern hardware):
  - 1M donations: ~2–3 seconds
  - 10M donations: ~20–30 seconds
  - 50M donations: ~100–150 seconds

## License

[Choose appropriate license - e.g., MIT, Apache 2.0]

## References

See `mock_data_plan.md` for detailed implementation documentation.
