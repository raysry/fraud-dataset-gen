#!/usr/bin/env python3
"""
Mock Donation Data Generator for Donation Fraud Detection.

Generates realistic donation data for a cloud storage tipping platform,
designed to test Louvain-based fraud ring detection.
"""

import argparse
import csv
import gzip
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np

# ─── Constants ───────────────────────────────────────────────────────────────

CHUNK_SIZE = 500_000

# Culturally significant round amounts (CNY)
ROUND_AMOUNTS = [1, 5, 10, 20, 50, 66, 88, 100, 200, 520, 1314]

# Chinese holidays (month, day, duration_days)
HOLIDAYS = [
    ((1, 28), 7),   # Chinese New Year (~late Jan/early Feb)
    ((6, 18), 1),   # 618
    ((10, 1), 7),   # National Day
    ((11, 11), 1),  # Double 11
]

CSV_COLUMNS = [
    "donation_id", "donor_id", "recipient_id", "amount", "timestamp",
    "device_type", "ip_hash", "session_seconds", "account_age_days",
    "account_reg_date", "is_refund",
]

PRESETS = {"small": 1_000, "medium": 10_000_000, "large": 50_000_000}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def uid(idx: int) -> str:
    return f"u_{idx:07d}"


def did(idx: int) -> str:
    return f"d_{idx:08d}"


def ip_hash(idx: int) -> str:
    return f"ip_{idx:05d}"


def ts_to_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def date_to_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def diurnal_weight(hour: int) -> float:
    """Activity weight by hour of day (CST). Peak 10-23, dip 2-6."""
    weights = [
        0.3, 0.15, 0.08, 0.05, 0.05, 0.05,  # 0-5
        0.1, 0.2, 0.4, 0.7, 1.0, 1.0,         # 6-11
        0.9, 0.85, 0.9, 1.0, 1.0, 1.0,         # 12-17
        1.0, 1.0, 1.0, 0.95, 0.8, 0.6,         # 18-23
    ]
    return weights[hour]


def is_holiday(dt: datetime) -> bool:
    for (m, d), dur in HOLIDAYS:
        start = datetime(dt.year, m, d)
        if start <= dt < start + timedelta(days=dur):
            return True
    return False


def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5


# ─── Entity Generation ──────────────────────────────────────────────────────

class EntityPool:
    """Manages all user IDs and their roles."""

    def __init__(self, rng: np.random.Generator, args):
        self.rng = rng
        self.args = args

        # Determine fraud ring structure
        self.rings = self._design_rings()
        total_fakes = sum(r["num_fakes"] for r in self.rings)
        total_customers = sum(r["num_customers"] for r in self.rings)

        # Allocate IDs: all in one namespace, shuffled
        total_users = args.normal_users + total_fakes + total_customers + args.recipients
        all_ids = list(range(total_users))
        rng.shuffle(all_ids)

        idx = 0
        # Assign recipient IDs
        self.recipient_ids = [all_ids[idx + i] for i in range(args.recipients)]
        idx += args.recipients

        # Assign normal user IDs
        self.normal_ids = [all_ids[idx + i] for i in range(args.normal_users)]
        idx += args.normal_users

        # Assign fraud ring IDs
        for ring in self.rings:
            ring["fake_ids"] = [all_ids[idx + i] for i in range(ring["num_fakes"])]
            idx += ring["num_fakes"]
            ring["customer_ids"] = [all_ids[idx + i] for i in range(ring["num_customers"])]
            idx += ring["num_customers"]

        # Cross-ring contamination: ~5% of fakes shared across 2 rings
        self._apply_cross_ring_fakes()
        # ~10% of customers appear in 2+ rings
        self._apply_cross_ring_customers()

        # Suspicious normals: ~0.5% of normal users
        num_suspicious = max(1, int(args.normal_users * 0.005))
        self.suspicious_normal_ids = list(rng.choice(
            self.normal_ids, size=num_suspicious, replace=False
        ))

        # Fan clubs: 3-5 clubs
        self.fan_clubs = self._create_fan_clubs()

        # Cross-donor-recipients: ~5% of recipients also donate
        num_cross = max(1, int(args.recipients * 0.05))
        self.cross_donor_recipient_ids = list(rng.choice(
            self.recipient_ids, size=num_cross, replace=False
        ))

        # Registration dates
        self.start_date = datetime.fromisoformat(args.start_date)
        self.end_date = datetime.fromisoformat(args.end_date)
        self._assign_reg_dates()

        # IP pools
        self._assign_ips()

    def _design_rings(self) -> list:
        """Auto-scale ring count and sizes to hit --fraud-pct."""
        rng = self.rng
        args = self.args
        target_fraud_donations = int(args.total * args.fraud_pct / 100)

        # Start with 8-15 rings
        num_rings = rng.integers(8, 16)

        # Assign difficulty tiers
        difficulties = []
        for _ in range(num_rings):
            r = rng.random()
            if r < 0.30:
                difficulties.append("easy")
            elif r < 0.75:
                difficulties.append("medium")
            else:
                difficulties.append("hard")

        # Assign business models
        business_models = []
        for _ in range(num_rings):
            r = rng.random()
            if r < 0.50:
                business_models.append("volume_brushing")
            elif r < 0.80:
                business_models.append("chart_topping")
            else:
                business_models.append("laundering")

        # Generate raw ring sizes
        rings = []
        for i in range(num_rings):
            num_fakes = int(rng.integers(50, 301))
            num_customers = int(rng.integers(5, 21))
            # Estimate donations per fake based on business model
            if business_models[i] == "volume_brushing":
                avg_donations_per_fake = rng.integers(30, 100)
            elif business_models[i] == "chart_topping":
                avg_donations_per_fake = rng.integers(10, 40)
            else:  # laundering
                avg_donations_per_fake = rng.integers(5, 20)

            est_donations = int(num_fakes * avg_donations_per_fake)

            # Refund strategy: ~20% of rings use evasion
            refund_strategy = "evasion" if rng.random() < 0.2 else "normal"

            rings.append({
                "ring_id": i,
                "difficulty": difficulties[i],
                "business_model": business_models[i],
                "refund_strategy": refund_strategy,
                "num_fakes": num_fakes,
                "num_customers": num_customers,
                "est_donations": est_donations,
            })

        # Scale to hit target fraud donations
        total_est = sum(r["est_donations"] for r in rings)
        if total_est > 0:
            scale = target_fraud_donations / total_est
            for r in rings:
                r["est_donations"] = max(10, int(r["est_donations"] * scale))

        return rings

    def _apply_cross_ring_fakes(self):
        """~5% of fake accounts shared across 2 rings."""
        if len(self.rings) < 2:
            return
        all_fakes = []
        for r in self.rings:
            all_fakes.extend(r["fake_ids"])

        num_shared = max(1, int(len(all_fakes) * 0.05))
        # Pick fakes to share
        for r in self.rings:
            r["shared_fakes_from_other_rings"] = []

        for _ in range(num_shared):
            src_idx = int(self.rng.integers(0, len(self.rings)))
            dst_idx = int(self.rng.integers(0, len(self.rings)))
            if src_idx == dst_idx:
                continue
            if not self.rings[src_idx]["fake_ids"]:
                continue
            fake = self.rng.choice(self.rings[src_idx]["fake_ids"])
            if fake not in self.rings[dst_idx]["fake_ids"]:
                self.rings[dst_idx]["fake_ids"].append(fake)
                self.rings[dst_idx]["shared_fakes_from_other_rings"].append(fake)

    def _apply_cross_ring_customers(self):
        """~10% of fraud customers appear in 2+ rings."""
        if len(self.rings) < 2:
            return
        all_customers = set()
        for r in self.rings:
            all_customers.update(r["customer_ids"])

        num_shared = max(1, int(len(all_customers) * 0.10))
        all_customers = list(all_customers)

        for _ in range(num_shared):
            src_idx = int(self.rng.integers(0, len(self.rings)))
            dst_idx = int(self.rng.integers(0, len(self.rings)))
            if src_idx == dst_idx:
                continue
            if not self.rings[src_idx]["customer_ids"]:
                continue
            cust = self.rng.choice(self.rings[src_idx]["customer_ids"])
            if cust not in self.rings[dst_idx]["customer_ids"]:
                self.rings[dst_idx]["customer_ids"].append(cust)

    def _create_fan_clubs(self) -> list:
        """3-5 fan clubs of 200-2000 normal users each."""
        num_clubs = int(self.rng.integers(3, 6))
        clubs = []
        available = set(self.normal_ids)

        for _ in range(num_clubs):
            if len(available) < 200:
                break
            size = int(self.rng.integers(200, min(2001, len(available) + 1)))
            members = list(self.rng.choice(list(available), size=size, replace=False))
            # 1-3 top creators they focus on
            num_targets = int(self.rng.integers(1, 4))
            targets = list(self.rng.choice(
                self.recipient_ids, size=min(num_targets, len(self.recipient_ids)),
                replace=False
            ))
            clubs.append({"recipients": targets, "members": members})
            available -= set(members)

        return clubs

    def _assign_reg_dates(self):
        """Assign registration dates to all users."""
        rng = self.rng

        self.reg_dates = {}

        # Normal users: spread over 1-1500 days before start_date
        for uid_id in self.normal_ids:
            days_before = int(rng.integers(1, 1501))
            self.reg_dates[uid_id] = self.start_date - timedelta(days=days_before)

        # Recipients
        for uid_id in self.recipient_ids:
            days_before = int(rng.integers(30, 1501))
            self.reg_dates[uid_id] = self.start_date - timedelta(days=days_before)

        # Fraud ring fakes: registered in batches (1-3 tight windows of 2-7 days)
        for ring in self.rings:
            num_windows = int(rng.integers(1, 4))
            windows = []
            for _ in range(num_windows):
                # Window starts sometime before or during the period
                days_before_start = int(rng.integers(7, 180))
                window_start = self.start_date - timedelta(days=days_before_start)
                window_span = int(rng.integers(2, 8))
                windows.append((window_start, window_span))

            for uid_id in ring["fake_ids"]:
                if uid_id in self.reg_dates:
                    continue  # shared fake already assigned
                w_start, w_span = windows[int(rng.integers(0, len(windows)))]
                offset = int(rng.integers(0, w_span))
                self.reg_dates[uid_id] = w_start + timedelta(days=offset)

            # Customers: normal-ish reg dates
            for uid_id in ring["customer_ids"]:
                if uid_id in self.reg_dates:
                    continue
                days_before = int(rng.integers(30, 1501))
                self.reg_dates[uid_id] = self.start_date - timedelta(days=days_before)

    def _assign_ips(self):
        """Assign IP pools."""
        rng = self.rng

        # Normal users: 1-3 IPs from a large pool (~50k unique)
        normal_ip_pool_size = 50000
        self.normal_user_ips = {}
        for uid_id in self.normal_ids:
            num_ips = int(rng.integers(1, 4))
            ips = [int(rng.integers(0, normal_ip_pool_size)) for _ in range(num_ips)]
            self.normal_user_ips[uid_id] = ips

        # Recipients
        for uid_id in self.recipient_ids:
            num_ips = int(rng.integers(1, 4))
            ips = [int(rng.integers(0, normal_ip_pool_size)) for _ in range(num_ips)]
            self.normal_user_ips[uid_id] = ips

        # Fraud rings: 3-8 proxy IPs per ring, some overlap between rings
        proxy_base = normal_ip_pool_size  # proxy IPs start after normal pool
        shared_proxy_pool = list(range(proxy_base, proxy_base + 30))
        self.ring_ips = {}
        for ring in self.rings:
            num_proxies = int(rng.integers(3, 9))
            # Mix of ring-specific and shared proxies
            ring_specific = [proxy_base + 30 + ring["ring_id"] * 10 + i for i in range(num_proxies - 1)]
            shared = [int(rng.choice(shared_proxy_pool))]
            self.ring_ips[ring["ring_id"]] = ring_specific + shared


# ─── Donation Generator ─────────────────────────────────────────────────────

class DonationGenerator:
    """Generates all donation records."""

    def __init__(self, rng: np.random.Generator, pool: EntityPool, args):
        self.rng = rng
        self.pool = pool
        self.args = args
        self.start_date = datetime.fromisoformat(args.start_date)
        self.end_date = datetime.fromisoformat(args.end_date)
        self.total_days = (self.end_date - self.start_date).days
        self.donation_counter = 0

        # Pre-compute fraud operation windows for each ring
        self._assign_operation_windows()

        # Determine delayed activation fakes per ring
        self._assign_delayed_activation()

        # Determine stealth fakes
        self._assign_stealth_fakes()

        # Pre-compute per-ring amount templates
        self._assign_amount_templates()

        # Pre-compute normal user donation counts (power-law)
        self._assign_normal_donation_counts()

    def _assign_operation_windows(self):
        """Each ring picks 3-8 burst windows (2-6 hour blocks)."""
        rng = self.rng
        for ring in self.pool.rings:
            num_windows = int(rng.integers(3, 9))
            windows = []
            for _ in range(num_windows):
                day_offset = int(rng.integers(0, self.total_days))
                hour_start = int(rng.integers(0, 22))
                duration = int(rng.integers(2, 7))
                dt_start = self.start_date + timedelta(days=day_offset, hours=hour_start)
                dt_end = dt_start + timedelta(hours=duration)
                if dt_end > self.end_date:
                    dt_end = self.end_date
                windows.append((dt_start, dt_end))
            ring["operation_windows"] = windows

            # Ring lifecycle: ~20% go dormant for 1-3 months
            ring["dormant_period"] = None
            if rng.random() < 0.20:
                dormant_start_day = int(rng.integers(60, max(61, self.total_days - 90)))
                dormant_duration = int(rng.integers(30, 91))
                ring["dormant_period"] = (
                    self.start_date + timedelta(days=dormant_start_day),
                    self.start_date + timedelta(days=dormant_start_day + dormant_duration),
                )

    def _assign_delayed_activation(self):
        """Hard-tier only: ~10% of hard-tier fakes have delayed activation."""
        rng = self.rng
        for ring in self.pool.rings:
            ring["delayed_activation_fakes"] = []
            if ring["difficulty"] == "hard":
                num_delayed = max(0, int(len(ring["fake_ids"]) * 0.10))
                if num_delayed > 0:
                    delayed = list(rng.choice(
                        ring["fake_ids"], size=num_delayed, replace=False
                    ))
                    ring["delayed_activation_fakes"] = delayed
                    # Activation delay: 2-4 months
                    ring["delayed_activation_date"] = (
                        self.start_date + timedelta(days=int(rng.integers(60, 121)))
                    )

    def _assign_stealth_fakes(self):
        """~2% of all fakes are stealth fakes."""
        all_fakes = set()
        for ring in self.pool.rings:
            all_fakes.update(ring["fake_ids"])
        all_fakes = list(all_fakes)

        num_stealth = max(0, int(len(all_fakes) * 0.02))
        if num_stealth > 0:
            self.stealth_fakes = list(self.rng.choice(
                all_fakes, size=num_stealth, replace=False
            ))
        else:
            self.stealth_fakes = []

    def _assign_amount_templates(self):
        """Each ring picks 2-4 template amounts + jitter."""
        rng = self.rng
        for ring in self.pool.rings:
            bm = ring["business_model"]
            if bm == "volume_brushing":
                base_range = (0.1, 10.0)
            elif bm == "chart_topping":
                base_range = (50.0, 2000.0)
            else:  # laundering
                base_range = (5.0, 500.0)

            num_templates = int(rng.integers(2, 5))
            templates = []
            for _ in range(num_templates):
                amt = rng.uniform(base_range[0], base_range[1])
                templates.append(round(amt, 2))
            ring["amount_templates"] = templates

            # Jitter range based on difficulty
            if ring["difficulty"] == "easy":
                ring["amount_jitter"] = 0.0  # fixed values
            elif ring["difficulty"] == "medium":
                ring["amount_jitter"] = 0.10  # ±10%
            else:
                ring["amount_jitter"] = 0.15  # ±15%

    def _assign_normal_donation_counts(self):
        """Power-law distribution for normal users. Most 1-10, some 50-500."""
        rng = self.rng
        args = self.args

        # Compute target normal donations
        total_fraud = sum(r["est_donations"] for r in self.pool.rings)
        target_normal = args.total - total_fraud

        # Power-law: shape parameter ~1.5
        raw = rng.pareto(1.5, size=args.normal_users) + 1
        raw = np.clip(raw, 1, 500)

        # Suspicious normals get boosted
        suspicious_set = set(self.pool.suspicious_normal_ids)
        for i, uid_id in enumerate(self.pool.normal_ids):
            if uid_id in suspicious_set:
                raw[i] = rng.uniform(50, 500)

        # Fan club members get boosted for their target recipients
        for club in self.pool.fan_clubs:
            member_set = set(club["members"])
            for i, uid_id in enumerate(self.pool.normal_ids):
                if uid_id in member_set:
                    raw[i] = max(raw[i], rng.uniform(10, 100))

        # Cross-donor-recipients
        cross_set = set(self.pool.cross_donor_recipient_ids)
        self.cross_donor_counts = {}
        for uid_id in self.pool.cross_donor_recipient_ids:
            self.cross_donor_counts[uid_id] = int(rng.integers(5, 50))

        # Scale to hit target
        total_raw = raw.sum() + sum(self.cross_donor_counts.values())
        if total_raw > 0:
            scale = target_normal / total_raw
            raw = np.maximum(1, np.round(raw * scale)).astype(int)
            for k in self.cross_donor_counts:
                self.cross_donor_counts[k] = max(1, int(self.cross_donor_counts[k] * scale))

        self.normal_donation_counts = raw

    def _random_timestamp(self, start: datetime = None, end: datetime = None) -> datetime:
        """Generate a random timestamp with diurnal/weekend/holiday weighting."""
        rng = self.rng
        if start is None:
            start = self.start_date
        if end is None:
            end = self.end_date

        total_secs = int((end - start).total_seconds())
        if total_secs <= 0:
            return start

        # Rejection sampling with diurnal weight
        for _ in range(100):
            offset = int(rng.integers(0, total_secs))
            dt = start + timedelta(seconds=offset)

            weight = diurnal_weight(dt.hour)
            if is_weekend(dt):
                weight *= 1.3
            if is_holiday(dt):
                weight *= 2.5

            if rng.random() < weight:
                return dt

        # Fallback
        offset = int(rng.integers(0, total_secs))
        return start + timedelta(seconds=offset)

    def _fraud_timestamp(self, ring: dict, is_in_window: bool) -> datetime:
        """Generate timestamp for a fraud donation."""
        if is_in_window and ring["operation_windows"]:
            # Pick a random operation window
            w_start, w_end = ring["operation_windows"][
                int(self.rng.integers(0, len(ring["operation_windows"])))
            ]
            return self._random_timestamp(w_start, w_end)
        return self._random_timestamp()

    def _fraud_amount(self, ring: dict, is_rampup: bool) -> float:
        """Generate amount for a fraud donation."""
        if is_rampup:
            return round(self.rng.uniform(0.1, 1.0), 2)

        template = float(self.rng.choice(ring["amount_templates"]))
        jitter = ring["amount_jitter"]
        if jitter > 0:
            template *= (1 + self.rng.uniform(-jitter, jitter))
        return round(np.clip(template, 0.10, 2000.00), 2)

    def _normal_amount(self) -> float:
        """Generate amount for a normal donation."""
        rng = self.rng
        # ~40% snap to round numbers
        if rng.random() < 0.40:
            return float(rng.choice(ROUND_AMOUNTS))
        # Log-normal base
        amt = float(np.exp(rng.normal(2.5, 1.5)))
        return round(np.clip(amt, 0.10, 2000.00), 2)

    def _session_seconds(self, is_fraud: bool, difficulty: str = "easy",
                         is_stealth: bool = False) -> int:
        """Generate session seconds."""
        rng = self.rng
        if is_fraud and not is_stealth:
            if difficulty == "hard":
                return int(np.clip(rng.lognormal(2.5, 0.8), 8, 60))
            return int(rng.integers(1, 6))
        # Normal human
        return int(np.clip(rng.lognormal(3.5, 1.0), 10, 300))

    def _device_type(self, is_fraud: bool, ring: dict = None) -> str:
        rng = self.rng
        if is_fraud and ring is not None:
            # Each ring uses 1-2 device types, skew api/pc
            if not hasattr(ring, "_device_pool_cache"):
                if ring["difficulty"] == "easy":
                    ring["_device_pool_cache"] = [rng.choice(["api", "pc"])]
                else:
                    choices = ["api", "pc", "mobile"]
                    n = 1 if ring["difficulty"] == "medium" else 2
                    ring["_device_pool_cache"] = list(rng.choice(choices, size=n, replace=False))
            return str(rng.choice(ring["_device_pool_cache"]))
        # Normal: ~60% mobile / 30% pc / 10% api
        r = rng.random()
        if r < 0.6:
            return "mobile"
        elif r < 0.9:
            return "pc"
        return "api"

    def _pick_ip(self, donor_id: int, is_fraud: bool, ring: dict = None) -> str:
        rng = self.rng
        if is_fraud and ring is not None:
            ips = self.pool.ring_ips[ring["ring_id"]]
            return ip_hash(int(rng.choice(ips)))
        if donor_id in self.pool.normal_user_ips:
            ips = self.pool.normal_user_ips[donor_id]
            return ip_hash(int(rng.choice(ips)))
        return ip_hash(int(rng.integers(0, 50000)))

    def generate_all(self) -> list:
        """Generate all donations and return as list of row dicts.

        Returns rows sorted by timestamp.
        """
        all_rows = []
        print("Generating fraud ring donations...")
        all_rows.extend(self._generate_fraud_donations())
        print(f"  → {len(all_rows)} fraud donations generated")

        print("Generating normal user donations...")
        normal_rows = self._generate_normal_donations()
        print(f"  → {len(normal_rows)} normal donations generated")
        all_rows.extend(normal_rows)

        print("Generating cross-donor-recipient donations...")
        cross_rows = self._generate_cross_donor_donations()
        print(f"  → {len(cross_rows)} cross-donor-recipient donations generated")
        all_rows.extend(cross_rows)

        print("Applying retry duplicates (~0.5%)...")
        retry_rows = self._generate_retries(all_rows)
        print(f"  → {len(retry_rows)} retry duplicates generated")
        all_rows.extend(retry_rows)

        print("Applying refunds...")
        refund_rows = self._generate_refunds(all_rows)
        print(f"  → {len(refund_rows)} refund records generated")
        all_rows.extend(refund_rows)

        # Sort by timestamp
        print("Sorting by timestamp...")
        all_rows.sort(key=lambda r: r["timestamp"])

        # Assign donation IDs in order
        for i, row in enumerate(all_rows):
            row["donation_id"] = did(i)

        return all_rows

    def _generate_fraud_donations(self) -> list:
        """Generate all fraud ring donations including warm-up and cover."""
        rng = self.rng
        rows = []

        for ring in self.pool.rings:
            ring_rows = []
            est = ring["est_donations"]
            fakes = ring["fake_ids"]
            customers = ring["customer_ids"]
            delayed_set = set(ring.get("delayed_activation_fakes", []))
            stealth_set = set(self.stealth_fakes)

            if not fakes or not customers:
                continue

            # Distribute donations among fakes
            per_fake = max(1, est // len(fakes))

            for fake_id in fakes:
                is_delayed = fake_id in delayed_set
                is_stealth = fake_id in stealth_set
                n_donations = max(1, int(per_fake + rng.integers(-per_fake // 3, per_fake // 3 + 1)))

                # Warm-up: ~30% of fakes get 1-3 small donations to random recipients
                if rng.random() < 0.30:
                    for _ in range(int(rng.integers(1, 4))):
                        target = int(rng.choice(self.pool.recipient_ids))
                        ts = self._random_timestamp(
                            self.start_date,
                            self.start_date + timedelta(days=max(1, self.total_days // 6))
                        )
                        row = self._make_row(
                            fake_id, target, round(rng.uniform(0.1, 5.0), 2),
                            ts, False, ring, is_stealth
                        )
                        ring_rows.append(row)

                # Cover donations: ~20% of fakes get 1-5 scattered donations
                if rng.random() < 0.20:
                    for _ in range(int(rng.integers(1, 6))):
                        target = int(rng.choice(self.pool.recipient_ids))
                        ts = self._random_timestamp()
                        row = self._make_row(
                            fake_id, target, self._normal_amount(),
                            ts, False, ring, is_stealth
                        )
                        ring_rows.append(row)

                # Main fraud donations
                activation_date = self.start_date
                if is_delayed and "delayed_activation_date" in ring:
                    activation_date = ring["delayed_activation_date"]
                    # Before activation: behave normally
                    normal_period = int((activation_date - self.start_date).days)
                    if normal_period > 30:
                        n_normal = int(rng.integers(3, 15))
                        for _ in range(n_normal):
                            target = int(rng.choice(self.pool.recipient_ids))
                            ts = self._random_timestamp(
                                self.start_date, activation_date
                            )
                            row = self._make_row(
                                fake_id, target, self._normal_amount(),
                                ts, False, ring, True  # stealth-like behavior
                            )
                            ring_rows.append(row)

                # Ramp-up: first 5-10% are small test donations
                rampup_count = max(1, int(n_donations * rng.uniform(0.05, 0.10)))

                for j in range(n_donations):
                    is_rampup = j < rampup_count
                    # ~85% in operation windows, 15% scattered
                    in_window = rng.random() < 0.85

                    # Check dormant period
                    ts = self._fraud_timestamp(ring, in_window)
                    if ring["dormant_period"]:
                        d_start, d_end = ring["dormant_period"]
                        if d_start <= ts < d_end:
                            # Skip or shift
                            ts = self._random_timestamp(d_end, self.end_date)

                    if is_delayed and ts < activation_date:
                        ts = self._random_timestamp(activation_date, self.end_date)

                    target = int(rng.choice(customers))
                    amount = self._fraud_amount(ring, is_rampup)

                    if is_stealth:
                        amount = self._normal_amount()

                    row = self._make_row(fake_id, target, amount, ts, True, ring, is_stealth)
                    ring_rows.append(row)

                # Incidental cross-ring donations (~1-3% of fakes)
                if rng.random() < 0.03 and len(self.pool.rings) > 1:
                    other_rings = [r for r in self.pool.rings if r["ring_id"] != ring["ring_id"]]
                    if other_rings:
                        other = rng.choice(other_rings)
                        n_cross = int(rng.integers(1, 3))
                        for _ in range(n_cross):
                            if other["customer_ids"]:
                                target = int(rng.choice(other["customer_ids"]))
                                ts = self._random_timestamp()
                                row = self._make_row(
                                    fake_id, target,
                                    round(rng.uniform(0.1, 5.0), 2),
                                    ts, True, ring, is_stealth
                                )
                                ring_rows.append(row)

            rows.extend(ring_rows)

        return rows

    def _generate_normal_donations(self) -> list:
        """Generate normal user donations."""
        rng = self.rng
        rows = []
        recipients = self.pool.recipient_ids
        suspicious_set = set(self.pool.suspicious_normal_ids)

        # Build fan club membership lookup
        fan_club_map = {}  # uid -> list of recipient ids
        for club in self.pool.fan_clubs:
            for member in club["members"]:
                fan_club_map.setdefault(member, []).extend(club["recipients"])

        total_normal = len(self.pool.normal_ids)
        report_every = max(1, total_normal // 10)

        for i, donor_id in enumerate(self.pool.normal_ids):
            if i > 0 and i % report_every == 0:
                print(f"  Normal users: {i}/{total_normal} ({100*i/total_normal:.0f}%)")

            n_donations = int(self.normal_donation_counts[i])

            # Suspicious normals: high-volume to few recipients
            if donor_id in suspicious_set:
                few_targets = list(rng.choice(
                    recipients, size=min(3, len(recipients)), replace=False
                ))
                for _ in range(n_donations):
                    target = int(rng.choice(few_targets))
                    ts = self._random_timestamp()
                    row = self._make_row(donor_id, target, self._normal_amount(),
                                         ts, False, None, False)
                    rows.append(row)
                continue

            # Fan club members: heavy overlap on specific recipients
            fan_targets = fan_club_map.get(donor_id, [])
            for j in range(n_donations):
                if fan_targets and rng.random() < 0.7:
                    target = int(rng.choice(fan_targets))
                else:
                    target = int(rng.choice(recipients))
                ts = self._random_timestamp()
                row = self._make_row(donor_id, target, self._normal_amount(),
                                     ts, False, None, False)
                rows.append(row)

        return rows

    def _generate_cross_donor_donations(self) -> list:
        """Recipients who also donate to other creators."""
        rng = self.rng
        rows = []
        recipients = self.pool.recipient_ids

        for donor_id in self.pool.cross_donor_recipient_ids:
            n = self.cross_donor_counts.get(donor_id, 5)
            other_recipients = [r for r in recipients if r != donor_id]
            if not other_recipients:
                continue
            for _ in range(n):
                target = int(rng.choice(other_recipients))
                ts = self._random_timestamp()
                row = self._make_row(donor_id, target, self._normal_amount(),
                                     ts, False, None, False)
                rows.append(row)

        return rows

    def _generate_retries(self, existing: list) -> list:
        """~0.5% retry duplicates."""
        rng = self.rng
        n_retries = int(len(existing) * 0.005)
        retries = []

        if n_retries == 0 or not existing:
            return retries

        indices = rng.choice(len(existing), size=n_retries, replace=True)
        for idx in indices:
            orig = existing[idx]
            delay = int(rng.integers(1, 61))
            ts = orig["timestamp"] + timedelta(seconds=delay)
            retry = dict(orig)
            retry["timestamp"] = ts
            retry["donation_id"] = ""  # will be reassigned
            retry["_is_retry"] = True
            retries.append(retry)

        return retries

    def _generate_refunds(self, existing: list) -> list:
        """Baseline ~1-2% random refunds + strategic refund evasion for some rings."""
        rng = self.rng
        refunds = []

        # Baseline random refunds: ~1.5%
        n_baseline = int(len(existing) * 0.015)
        if n_baseline > 0:
            indices = rng.choice(len(existing), size=n_baseline, replace=False)
            for idx in indices:
                orig = existing[idx]
                if orig.get("is_refund"):
                    continue
                delay = int(rng.integers(60, 86400))  # 1 min to 1 day
                ts = orig["timestamp"] + timedelta(seconds=delay)
                refund = dict(orig)
                refund["timestamp"] = ts
                refund["is_refund"] = True
                refund["donation_id"] = ""
                refunds.append(refund)

        # Strategic refund evasion rings: refund 20-30% of their donations
        for ring in self.pool.rings:
            if ring["refund_strategy"] != "evasion":
                continue
            ring_fakes = set(ring["fake_ids"])
            ring_donations = [
                r for r in existing
                if r["_donor_raw"] in ring_fakes and not r.get("is_refund")
            ]
            if not ring_donations:
                continue
            refund_pct = rng.uniform(0.20, 0.30)
            n_refund = int(len(ring_donations) * refund_pct)
            # Prefer large amounts
            ring_donations.sort(key=lambda r: -r["amount"])
            for orig in ring_donations[:n_refund]:
                delay = int(rng.integers(30, 600))  # within minutes
                ts = orig["timestamp"] + timedelta(seconds=delay)
                refund = dict(orig)
                refund["timestamp"] = ts
                refund["is_refund"] = True
                refund["donation_id"] = ""
                refunds.append(refund)

        return refunds

    def _make_row(self, donor_id: int, recipient_id: int, amount: float,
                  timestamp: datetime, is_fraud: bool, ring: dict = None,
                  is_stealth: bool = False) -> dict:
        difficulty = ring["difficulty"] if ring else "easy"
        reg_date = self.pool.reg_dates.get(donor_id, self.start_date - timedelta(days=100))
        account_age = max(0, (timestamp - reg_date).days)

        return {
            "donation_id": "",
            "donor_id": uid(donor_id),
            "recipient_id": uid(recipient_id),
            "amount": amount,
            "timestamp": timestamp,
            "device_type": self._device_type(is_fraud, ring),
            "ip_hash": self._pick_ip(donor_id, is_fraud, ring),
            "session_seconds": self._session_seconds(is_fraud, difficulty, is_stealth),
            "account_age_days": account_age,
            "account_reg_date": date_to_str(reg_date),
            "is_refund": False,
            "_is_fraud": is_fraud,
            "_donor_raw": donor_id,
            "_recipient_raw": recipient_id,
            "_ring_id": ring["ring_id"] if ring else -1,
        }


# ─── Output ──────────────────────────────────────────────────────────────────

def write_csv(rows: list, output_path: str, compress: bool):
    """Write donations CSV in chunks with progress."""
    fname = "donations.csv.gz" if compress else "donations.csv"
    fpath = os.path.join(output_path, fname)
    total = len(rows)

    opener = gzip.open if compress else open
    mode = "wt" if compress else "w"

    print(f"Writing {total:,} rows to {fpath}...")
    t0 = time.time()

    with opener(fpath, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for i in range(0, total, CHUNK_SIZE):
            chunk = rows[i:i + CHUNK_SIZE]
            for row in chunk:
                out_row = dict(row)
                out_row["timestamp"] = ts_to_iso(row["timestamp"])
                out_row["is_refund"] = str(row["is_refund"]).lower()
                writer.writerow(out_row)

            written = min(i + CHUNK_SIZE, total)
            elapsed = time.time() - t0
            eta = (elapsed / written) * (total - written) if written > 0 else 0
            print(f"  {written:,}/{total:,} ({100*written/total:.1f}%) "
                  f"elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    elapsed = time.time() - t0
    print(f"CSV written in {elapsed:.1f}s")
    return fpath


def write_labels(pool: EntityPool, gen: DonationGenerator, rows: list,
                 output_path: str, label_noise: float):
    """Write labels.json with ground truth."""
    fpath = os.path.join(output_path, "labels.json")

    # Count stats
    fraud_donations = sum(1 for r in rows if r.get("_is_fraud"))
    total_refunds = sum(1 for r in rows if r.get("is_refund"))
    total_retries = sum(1 for r in rows if r.get("_is_retry"))

    fraud_rings_json = []
    for ring in pool.rings:
        fraud_rings_json.append({
            "ring_id": ring["ring_id"],
            "difficulty": ring["difficulty"],
            "business_model": ring["business_model"],
            "refund_strategy": ring["refund_strategy"],
            "fake_accounts": [uid(x) for x in ring["fake_ids"]],
            "customer_accounts": [uid(x) for x in ring["customer_ids"]],
            "shared_fakes_from_other_rings": [uid(x) for x in ring.get("shared_fakes_from_other_rings", [])],
            "delayed_activation_fakes": [uid(x) for x in ring.get("delayed_activation_fakes", [])],
            "operation_windows": [
                [ts_to_iso(w[0]), ts_to_iso(w[1])] for w in ring.get("operation_windows", [])
            ],
        })

    # Difficulty distribution
    diff_dist = {}
    bm_dist = {}
    refund_evasion_count = 0
    for ring in pool.rings:
        diff_dist[ring["difficulty"]] = diff_dist.get(ring["difficulty"], 0) + 1
        bm_dist[ring["business_model"]] = bm_dist.get(ring["business_model"], 0) + 1
        if ring["refund_strategy"] == "evasion":
            refund_evasion_count += 1

    all_fake_ids = set()
    all_customer_ids = set()
    for ring in pool.rings:
        all_fake_ids.update(ring["fake_ids"])
        all_customer_ids.update(ring["customer_ids"])

    labels = {
        "fraud_rings": fraud_rings_json,
        "stealth_fakes": [uid(x) for x in gen.stealth_fakes],
        "delayed_activation_fakes": [
            uid(x) for ring in pool.rings for x in ring.get("delayed_activation_fakes", [])
        ],
        "suspicious_normals": [uid(x) for x in pool.suspicious_normal_ids],
        "fan_clubs": [
            {
                "recipients": [uid(x) for x in club["recipients"]],
                "members": [uid(x) for x in club["members"]],
            }
            for club in pool.fan_clubs
        ],
        "cross_donor_recipients": [uid(x) for x in pool.cross_donor_recipient_ids],
        "normal_users": [uid(x) for x in pool.normal_ids],
        "recipients": [uid(x) for x in pool.recipient_ids],
        "stats": {
            "total_donations": len(rows),
            "total_refunds": total_refunds,
            "total_retries": total_retries,
            "fraud_donations": fraud_donations,
            "fraud_pct": round(100 * fraud_donations / len(rows), 2) if rows else 0,
            "num_rings": len(pool.rings),
            "num_fake_accounts": len(all_fake_ids),
            "num_customer_accounts": len(all_customer_ids),
            "num_normal_users": len(pool.normal_ids),
            "num_recipients": len(pool.recipient_ids),
            "difficulty_distribution": diff_dist,
            "business_model_distribution": bm_dist,
            "rings_with_refund_evasion": refund_evasion_count,
            "num_fan_clubs": len(pool.fan_clubs),
            "label_noise_pct": label_noise,
        },
    }

    # Apply label noise
    if label_noise > 0:
        _apply_label_noise(labels, label_noise, gen.rng)

    with open(fpath, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Labels written to {fpath}")
    return labels


def _apply_label_noise(labels: dict, noise_pct: float, rng: np.random.Generator):
    """Flip noise_pct% of labels (swap some normal↔fraud)."""
    normal_set = set(labels["normal_users"])
    fraud_set = set()
    for ring in labels["fraud_rings"]:
        fraud_set.update(ring["fake_accounts"])

    total = len(normal_set) + len(fraud_set)
    n_flips = int(total * noise_pct / 100)

    # Flip some normals to appear in fraud
    normal_list = list(normal_set)
    fraud_list = list(fraud_set)

    n_normal_to_fraud = min(n_flips // 2, len(normal_list))
    n_fraud_to_normal = min(n_flips - n_normal_to_fraud, len(fraud_list))

    if n_normal_to_fraud > 0:
        flipped_normals = list(rng.choice(normal_list, size=n_normal_to_fraud, replace=False))
        # Add to first ring's fakes
        if labels["fraud_rings"]:
            labels["fraud_rings"][0]["fake_accounts"].extend(flipped_normals)

    if n_fraud_to_normal > 0:
        flipped_fraud = list(rng.choice(fraud_list, size=n_fraud_to_normal, replace=False))
        labels["normal_users"].extend(flipped_fraud)


def print_quality_metrics(pool: EntityPool, rows: list):
    """Print per-ring edge density and other quality metrics."""
    print("\n" + "=" * 60)
    print("QUALITY METRICS")
    print("=" * 60)

    # Build co-action counts for fakes and normals
    for ring in pool.rings:
        fake_set = set(ring["fake_ids"])
        customer_set = set(ring["customer_ids"])

        # Count fake→customer edges
        edges = set()
        for r in rows:
            if r["_donor_raw"] in fake_set and r["_recipient_raw"] in customer_set:
                edges.add((r["_donor_raw"], r["_recipient_raw"]))

        possible = len(fake_set) * len(customer_set) if customer_set else 1
        density = len(edges) / possible if possible > 0 else 0

        # Mean degree for fakes in this ring
        fake_degrees = {}
        for r in rows:
            if r["_donor_raw"] in fake_set:
                fake_degrees[r["_donor_raw"]] = fake_degrees.get(r["_donor_raw"], 0) + 1
        mean_fake_degree = (
            sum(fake_degrees.values()) / len(fake_degrees) if fake_degrees else 0
        )

        print(f"\n  Ring {ring['ring_id']} [{ring['difficulty']}/{ring['business_model']}]:")
        print(f"    Fakes: {len(fake_set)}, Customers: {len(customer_set)}")
        print(f"    Fake→Customer edge density: {density:.4f}")
        print(f"    Mean fake donation count: {mean_fake_degree:.1f}")

    # Normal user mean degree
    normal_set = set(pool.normal_ids)
    normal_degrees = {}
    for r in rows:
        if r["_donor_raw"] in normal_set:
            normal_degrees[r["_donor_raw"]] = normal_degrees.get(r["_donor_raw"], 0) + 1
    mean_normal = (
        sum(normal_degrees.values()) / len(normal_degrees) if normal_degrees else 0
    )
    print(f"\n  Normal users mean donation count: {mean_normal:.1f}")

    # Fan club sizes
    for i, club in enumerate(pool.fan_clubs):
        print(f"  Fan club {i}: {len(club['members'])} members → "
              f"{len(club['recipients'])} recipients")

    print("=" * 60)


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate mock donation data for fraud detection testing."
    )
    p.add_argument("-n", "--total", type=int, default=10_000_000,
                   help="Total donation records (default: 10M)")
    p.add_argument("--preset", choices=["small", "medium", "large"],
                   help="Preset sizes: small=1K, medium=10M, large=50M")
    p.add_argument("--fraud-pct", type=float, default=1.5,
                   help="Target fraud percentage (default: 1.5)")
    p.add_argument("--normal-users", type=int, default=500_000,
                   help="Normal user pool size (default: 500K)")
    p.add_argument("--recipients", type=int, default=800,
                   help="Popular recipient pool (default: 800)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--start-date", default="2025-01-01",
                   help="Time window start (default: 2025-01-01)")
    p.add_argument("--end-date", default="2025-12-31",
                   help="Time window end (default: 2025-12-31)")
    p.add_argument("--output-dir", default=".",
                   help="Output directory (default: script directory)")
    p.add_argument("--compress", action="store_true",
                   help="Write .csv.gz instead of plain CSV")
    p.add_argument("--multi-seed", type=int, default=1,
                   help="Generate N datasets with seeds seed..seed+N-1")
    p.add_argument("--label-noise", type=float, default=0,
                   help="Percentage of labels to flip (default: 0)")
    return p.parse_args()


def run_single(args, seed: int, output_dir: str):
    """Run a single dataset generation."""
    print(f"\n{'='*60}")
    print(f"Generating dataset: seed={seed}, total={args.total:,}, "
          f"fraud_pct={args.fraud_pct}%")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    rng = np.random.default_rng(seed)

    print("Building entity pool...")
    pool = EntityPool(rng, args)
    print(f"  Rings: {len(pool.rings)}, "
          f"Normal users: {len(pool.normal_ids):,}, "
          f"Recipients: {len(pool.recipient_ids)}")

    gen = DonationGenerator(rng, pool, args)

    rows = gen.generate_all()

    # Trim or pad to hit target total
    if len(rows) > args.total:
        print(f"Trimming {len(rows):,} → {args.total:,} rows")
        # Keep fraud rows, trim normals
        fraud_rows = [r for r in rows if r.get("_is_fraud") or r.get("is_refund")]
        normal_rows = [r for r in rows if not r.get("_is_fraud") and not r.get("is_refund")]
        need = args.total - len(fraud_rows)
        if need > 0 and normal_rows:
            rng.shuffle(normal_rows)
            rows = fraud_rows + normal_rows[:need]
        else:
            rows = rows[:args.total]
        rows.sort(key=lambda r: r["timestamp"])
        for i, row in enumerate(rows):
            row["donation_id"] = did(i)

    # Write outputs
    csv_path = write_csv(rows, output_dir, args.compress)
    labels = write_labels(pool, gen, rows, output_dir, args.label_noise)

    # Print summary
    stats = labels["stats"]
    elapsed = time.time() - t0
    print(f"\n{'─'*60}")
    print(f"SUMMARY (seed={seed})")
    print(f"{'─'*60}")
    print(f"  Total donations: {stats['total_donations']:,}")
    print(f"  Fraud donations: {stats['fraud_donations']:,} "
          f"({stats['fraud_pct']:.2f}%)")
    print(f"  Total refunds:   {stats['total_refunds']:,}")
    print(f"  Total retries:   {stats['total_retries']:,}")
    print(f"  Rings:           {stats['num_rings']}")
    print(f"  Fake accounts:   {stats['num_fake_accounts']:,}")
    print(f"  Customer accts:  {stats['num_customer_accounts']:,}")
    print(f"  Normal users:    {stats['num_normal_users']:,}")
    print(f"  Recipients:      {stats['num_recipients']}")
    print(f"  Difficulty:      {stats['difficulty_distribution']}")
    print(f"  Business models: {stats['business_model_distribution']}")
    print(f"  Refund evasion:  {stats['rings_with_refund_evasion']} rings")
    print(f"  Fan clubs:       {stats['num_fan_clubs']}")
    print(f"  Label noise:     {stats['label_noise_pct']}%")
    print(f"  Time elapsed:    {elapsed:.1f}s")

    # Validate fraud percentage
    target = args.fraud_pct
    actual = stats["fraud_pct"]
    if abs(actual - target) > target * 0.5:
        print(f"  ⚠ WARNING: fraud_pct={actual:.2f}% deviates significantly "
              f"from target={target:.1f}%")
    else:
        print(f"  ✓ Fraud percentage within expected range")

    print_quality_metrics(pool, rows)


def main():
    args = parse_args()

    if args.preset:
        args.total = PRESETS[args.preset]

    # Scale normal-users and recipients for small datasets
    if args.total < 5_000_000:
        scale = args.total / 10_000_000
        args.normal_users = max(1000, int(args.normal_users * scale))
        args.recipients = max(50, int(args.recipients * scale))

    # Resolve output-dir relative to script location
    if args.output_dir == ".":
        args.output_dir = os.path.dirname(os.path.abspath(__file__))

    if args.multi_seed > 1:
        for i in range(args.multi_seed):
            seed = args.seed + i
            out = os.path.join(args.output_dir, f"run_{i}")
            run_single(args, seed, out)
    else:
        run_single(args, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
