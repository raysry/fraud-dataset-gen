#!/usr/bin/env python3
"""
Unit tests for gen_donation_fraud_data.py
"""

import tempfile
import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime

# Import the module
import gen_donation_fraud_data as gdf


class TestHelpers:
    """Test helper functions."""

    def test_uid(self):
        assert gdf.uid(1) == "u_0000001"
        assert gdf.uid(1000000) == "u_1000000"

    def test_did(self):
        assert gdf.did(1) == "d_00000001"
        assert gdf.did(100000000) == "d_100000000"

    def test_ip_hash(self):
        assert gdf.ip_hash(1) == "ip_00001"
        assert gdf.ip_hash(10000) == "ip_10000"

    def test_ts_to_iso(self):
        dt = datetime(2025, 3, 15, 14, 30, 45)
        assert gdf.ts_to_iso(dt) == "2025-03-15T14:30:45"

    def test_date_to_str(self):
        dt = datetime(2025, 3, 15)
        assert gdf.date_to_str(dt) == "2025-03-15"

    def test_diurnal_weight(self):
        # Peak hours should have weight 1.0
        assert gdf.diurnal_weight(10) == 1.0
        assert gdf.diurnal_weight(23) == 0.6
        # Low hours should be lower
        assert gdf.diurnal_weight(3) < gdf.diurnal_weight(10)

    def test_is_holiday(self):
        # Chinese New Year spans 7 days from Jan 28
        assert gdf.is_holiday(datetime(2025, 1, 28)) is True
        assert gdf.is_holiday(datetime(2025, 2, 3)) is True
        # Not a holiday
        assert gdf.is_holiday(datetime(2025, 3, 15)) is False

    def test_is_weekend(self):
        # March 15, 2025 is Saturday
        assert gdf.is_weekend(datetime(2025, 3, 15)) is True
        # March 17, 2025 is Monday
        assert gdf.is_weekend(datetime(2025, 3, 17)) is False


class TestEntityPool:
    """Test entity pool generation."""

    def test_entity_pool_creation(self):
        rng = np.random.default_rng(42)
        args = type('Args', (), {
            'total': 1_000_000,
            'fraud_pct': 1.5,
            'normal_users': 10_000,
            'recipients': 100,
            'start_date': '2025-01-01',
            'end_date': '2025-12-31',
            'seed': 42,
        })()

        pool = gdf.EntityPool(rng, args)

        # Check basic structure
        assert len(pool.rings) > 0
        assert len(pool.normal_ids) == 10_000
        assert len(pool.recipient_ids) == 100
        assert len(pool.suspicious_normal_ids) > 0
        assert len(pool.fan_clubs) > 0

    def test_ring_difficulty_distribution(self):
        rng = np.random.default_rng(42)
        args = type('Args', (), {
            'total': 1_000_000,
            'fraud_pct': 1.5,
            'normal_users': 10_000,
            'recipients': 100,
            'start_date': '2025-01-01',
            'end_date': '2025-12-31',
            'seed': 42,
        })()

        pool = gdf.EntityPool(rng, args)

        difficulties = [r['difficulty'] for r in pool.rings]
        assert all(d in ['easy', 'medium', 'hard'] for d in difficulties)

    def test_ring_business_models(self):
        rng = np.random.default_rng(42)
        args = type('Args', (), {
            'total': 1_000_000,
            'fraud_pct': 1.5,
            'normal_users': 10_000,
            'recipients': 100,
            'start_date': '2025-01-01',
            'end_date': '2025-12-31',
            'seed': 42,
        })()

        pool = gdf.EntityPool(rng, args)

        models = [r['business_model'] for r in pool.rings]
        assert all(m in ['volume_brushing', 'chart_topping', 'laundering'] for m in models)


class TestDonationGenerator:
    """Test donation generation."""

    def test_donation_generator_creation(self):
        rng = np.random.default_rng(42)
        args = type('Args', (), {
            'total': 100_000,
            'fraud_pct': 1.5,
            'normal_users': 5_000,
            'recipients': 100,
            'start_date': '2025-01-01',
            'end_date': '2025-12-31',
            'seed': 42,
        })()

        pool = gdf.EntityPool(rng, args)
        gen = gdf.DonationGenerator(rng, pool, args)

        # Check that generator has operation windows
        for ring in pool.rings:
            assert 'operation_windows' in ring
            assert len(ring['operation_windows']) > 0

    def test_fraud_amount_generation(self):
        rng = np.random.default_rng(42)
        args = type('Args', (), {
            'total': 100_000,
            'fraud_pct': 1.5,
            'normal_users': 5_000,
            'recipients': 100,
            'start_date': '2025-01-01',
            'end_date': '2025-12-31',
            'seed': 42,
        })()

        pool = gdf.EntityPool(rng, args)
        gen = gdf.DonationGenerator(rng, pool, args)

        # Test fraud amount
        ring = pool.rings[0]
        amt = gen._fraud_amount(ring, is_rampup=True)
        assert 0.10 <= amt <= 1.0

        # Test normal amount
        normal_amt = gen._normal_amount()
        assert 0.10 <= normal_amt <= 2000.0

    def test_session_seconds_generation(self):
        rng = np.random.default_rng(42)
        args = type('Args', (), {
            'total': 100_000,
            'fraud_pct': 1.5,
            'normal_users': 5_000,
            'recipients': 100,
            'start_date': '2025-01-01',
            'end_date': '2025-12-31',
            'seed': 42,
        })()

        pool = gdf.EntityPool(rng, args)
        gen = gdf.DonationGenerator(rng, pool, args)

        # Fraud bot: 1-5 seconds
        fraud_session = gen._session_seconds(is_fraud=True, difficulty='easy')
        assert 1 <= fraud_session <= 5

        # Normal human: 10-300 seconds
        normal_session = gen._session_seconds(is_fraud=False)
        assert 10 <= normal_session <= 300


class TestIntegration:
    """Integration tests."""

    def test_small_dataset_generation(self):
        """Generate a small dataset end-to-end."""
        rng = np.random.default_rng(42)
        args = type('Args', (), {
            'total': 100_000,
            'fraud_pct': 1.5,
            'normal_users': 5_000,
            'recipients': 50,
            'start_date': '2025-01-01',
            'end_date': '2025-12-31',
            'seed': 42,
        })()

        pool = gdf.EntityPool(rng, args)
        gen = gdf.DonationGenerator(rng, pool, args)
        rows = gen.generate_all()

        # Basic validations
        assert len(rows) > 0
        assert all('donation_id' in r for r in rows)
        assert all('donor_id' in r for r in rows)
        assert all('recipient_id' in r for r in rows)
        assert all(0.10 <= r['amount'] <= 2000.0 for r in rows)

    def test_csv_and_labels_output(self):
        """Test CSV and labels JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.default_rng(42)
            args = type('Args', (), {
                'total': 100_000,
                'fraud_pct': 1.5,
                'normal_users': 5_000,
                'recipients': 100,
                'start_date': '2025-01-01',
                'end_date': '2025-12-31',
                'seed': 42,
            })()

            pool = gdf.EntityPool(rng, args)
            gen = gdf.DonationGenerator(rng, pool, args)
            rows = gen.generate_all()

            # Write outputs
            gdf.write_csv(rows, tmpdir, compress=False)
            labels = gdf.write_labels(pool, gen, rows, tmpdir, label_noise=0)

            # Check files exist
            csv_path = Path(tmpdir) / "donations.csv"
            labels_path = Path(tmpdir) / "labels.json"

            assert csv_path.exists()
            assert labels_path.exists()

            # Check labels structure
            assert 'fraud_rings' in labels
            assert 'stealth_fakes' in labels
            assert 'normal_users' in labels
            assert 'stats' in labels
            assert labels['stats']['fraud_pct'] > 0


if __name__ == "__main__":
    # Simple test runner
    print("Running TestHelpers...")
    helpers = TestHelpers()
    for method in dir(helpers):
        if method.startswith("test_"):
            try:
                getattr(helpers, method)()
                print(f"  ✓ {method}")
            except AssertionError as e:
                print(f"  ✗ {method}: {e}")

    print("\nRunning TestEntityPool...")
    entity_tests = TestEntityPool()
    for method in dir(entity_tests):
        if method.startswith("test_"):
            try:
                getattr(entity_tests, method)()
                print(f"  ✓ {method}")
            except Exception as e:
                print(f"  ✗ {method}: {e}")

    print("\nRunning TestDonationGenerator...")
    gen_tests = TestDonationGenerator()
    for method in dir(gen_tests):
        if method.startswith("test_"):
            try:
                getattr(gen_tests, method)()
                print(f"  ✓ {method}")
            except Exception as e:
                print(f"  ✗ {method}: {e}")

    print("\nRunning TestIntegration...")
    int_tests = TestIntegration()
    for method in dir(int_tests):
        if method.startswith("test_"):
            try:
                getattr(int_tests, method)()
                print(f"  ✓ {method}")
            except Exception as e:
                print(f"  ✗ {method}: {e}")

    print("\n✓ All tests completed!")
