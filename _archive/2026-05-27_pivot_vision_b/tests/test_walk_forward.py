# =============================================================================
# WALK-FORWARD VALIDATION TESTS (Sprint 3)
# =============================================================================
"""
Tests for the Walk-Forward Validation system.

Walk-forward validation is CRITICAL for trading systems because:
1. Markets are non-stationary - patterns change over time
2. Standard train/test splits give overly optimistic results
3. Walk-forward simulates real trading: train on past, test on future

Run tests:
    python -m pytest tests/test_walk_forward.py -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules to test
from parallel_training import (
    WalkForwardValidator,
    WALK_FORWARD_CONFIG,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing (1 year of 15-min data)."""
    # 1 year of 15-minute bars = ~35,000 bars
    n_bars = 35040  # 365 days * 24 hours * 4 bars/hour

    dates = pd.date_range(
        start='2023-01-01',
        periods=n_bars,
        freq='15min'
    )

    # Generate synthetic price data (random walk)
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, n_bars)
    close = 1900 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'Date': dates,
        'Open': close * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        'High': close * (1 + np.random.uniform(0, 0.002, n_bars)),
        'Low': close * (1 - np.random.uniform(0, 0.002, n_bars)),
        'Close': close,
        'Volume': np.random.randint(1000, 10000, n_bars)
    })

    return df


@pytest.fixture
def small_df():
    """Create a smaller DataFrame for quick tests."""
    n_bars = 12000  # ~4 months

    dates = pd.date_range(
        start='2023-01-01',
        periods=n_bars,
        freq='15min'
    )

    np.random.seed(123)
    close = 2000 + np.cumsum(np.random.normal(0, 1, n_bars))

    df = pd.DataFrame({
        'Date': dates,
        'Open': close,
        'High': close + 5,
        'Low': close - 5,
        'Close': close,
        'Volume': 5000
    })

    return df


@pytest.fixture
def custom_config():
    """Create a custom walk-forward config for testing."""
    return {
        'train_window_bars': 4000,
        'validation_window_bars': 1000,
        'test_window_bars': 500,
        'step_size_bars': 500,
        'purge_gap_bars': 50,
        'min_folds': 2,
        'max_folds': 10,
        'strategy': 'rolling',
        'early_stop_degradation_threshold': 0.3,
    }


# =============================================================================
# TEST: FOLD GENERATION
# =============================================================================

class TestFoldGeneration:
    """Tests for walk-forward fold generation."""

    def test_rolling_strategy_generates_folds(self, sample_df, custom_config):
        """Test that rolling strategy generates the correct number of folds."""
        custom_config['strategy'] = 'rolling'
        validator = WalkForwardValidator(sample_df, custom_config)

        folds = validator.generate_folds()

        # Should generate multiple folds
        assert len(folds) >= custom_config['min_folds'], \
            f"Expected at least {custom_config['min_folds']} folds, got {len(folds)}"

        # Should not exceed max folds
        assert len(folds) <= custom_config['max_folds'], \
            f"Expected at most {custom_config['max_folds']} folds, got {len(folds)}"

    def test_expanding_strategy_generates_folds(self, sample_df, custom_config):
        """Test that expanding strategy generates folds with growing train sets."""
        custom_config['strategy'] = 'expanding'
        validator = WalkForwardValidator(sample_df, custom_config)

        folds = validator.generate_folds()

        # Should generate multiple folds
        assert len(folds) >= 2, f"Expected at least 2 folds, got {len(folds)}"

        # Each subsequent fold should have a larger training set
        if len(folds) >= 2:
            for i in range(1, len(folds)):
                train_size_prev = folds[i-1]['train_end'] - folds[i-1]['train_start']
                train_size_curr = folds[i]['train_end'] - folds[i]['train_start']
                assert train_size_curr >= train_size_prev, \
                    f"Expanding strategy: fold {i} train size ({train_size_curr}) " \
                    f"should be >= fold {i-1} ({train_size_prev})"

    def test_anchored_strategy_generates_folds(self, sample_df, custom_config):
        """Test that anchored strategy generates folds with fixed start."""
        custom_config['strategy'] = 'anchored'
        validator = WalkForwardValidator(sample_df, custom_config)

        folds = validator.generate_folds()

        # All folds should start from index 0
        for fold in folds:
            assert fold['train_start'] == 0, \
                f"Anchored strategy: train_start should be 0, got {fold['train_start']}"

    def test_fold_ids_are_sequential(self, sample_df, custom_config):
        """Test that fold IDs are sequential starting from 0."""
        validator = WalkForwardValidator(sample_df, custom_config)
        folds = validator.generate_folds()

        for i, fold in enumerate(folds):
            assert fold['fold_id'] == i, \
                f"Expected fold_id {i}, got {fold['fold_id']}"


# =============================================================================
# TEST: PURGE GAP ENFORCEMENT
# =============================================================================

class TestPurgeGap:
    """Tests for purge gap enforcement (preventing look-ahead bias)."""

    def test_purge_gap_between_train_and_val(self, sample_df, custom_config):
        """Test that there's a purge gap between train and validation sets."""
        purge_gap = custom_config['purge_gap_bars']
        validator = WalkForwardValidator(sample_df, custom_config)
        folds = validator.generate_folds()

        for fold in folds:
            gap = fold['val_start'] - fold['train_end']
            assert gap >= purge_gap, \
                f"Fold {fold['fold_id']}: Gap between train and val ({gap}) " \
                f"should be >= purge_gap ({purge_gap})"

    def test_purge_gap_between_val_and_test(self, sample_df, custom_config):
        """Test that there's a purge gap between validation and test sets."""
        purge_gap = custom_config['purge_gap_bars']
        validator = WalkForwardValidator(sample_df, custom_config)
        folds = validator.generate_folds()

        for fold in folds:
            gap = fold['test_start'] - fold['val_end']
            assert gap >= purge_gap, \
                f"Fold {fold['fold_id']}: Gap between val and test ({gap}) " \
                f"should be >= purge_gap ({purge_gap})"

    def test_no_overlap_between_sets(self, sample_df, custom_config):
        """Test that train, val, and test sets don't overlap."""
        validator = WalkForwardValidator(sample_df, custom_config)
        folds = validator.generate_folds()

        for fold in folds:
            # Train should end before val starts
            assert fold['train_end'] < fold['val_start'], \
                f"Fold {fold['fold_id']}: train_end ({fold['train_end']}) " \
                f"should be < val_start ({fold['val_start']})"

            # Val should end before test starts
            assert fold['val_end'] < fold['test_start'], \
                f"Fold {fold['fold_id']}: val_end ({fold['val_end']}) " \
                f"should be < test_start ({fold['test_start']})"

            # Test should end within data bounds
            assert fold['test_end'] <= len(sample_df), \
                f"Fold {fold['fold_id']}: test_end ({fold['test_end']}) " \
                f"should be <= data length ({len(sample_df)})"


# =============================================================================
# TEST: DATA EXTRACTION
# =============================================================================

class TestDataExtraction:
    """Tests for extracting fold data."""

    def test_get_fold_data_returns_correct_sizes(self, sample_df, custom_config):
        """Test that extracted data has correct sizes."""
        validator = WalkForwardValidator(sample_df, custom_config)
        folds = validator.generate_folds()

        for fold in folds:
            df_train, df_val, df_test = validator.get_fold_data(fold)

            expected_train_size = fold['train_end'] - fold['train_start']
            expected_val_size = fold['val_end'] - fold['val_start']
            expected_test_size = fold['test_end'] - fold['test_start']

            assert len(df_train) == expected_train_size, \
                f"Fold {fold['fold_id']}: Expected train size {expected_train_size}, got {len(df_train)}"
            assert len(df_val) == expected_val_size, \
                f"Fold {fold['fold_id']}: Expected val size {expected_val_size}, got {len(df_val)}"
            assert len(df_test) == expected_test_size, \
                f"Fold {fold['fold_id']}: Expected test size {expected_test_size}, got {len(df_test)}"

    def test_fold_data_is_chronological(self, sample_df, custom_config):
        """Test that fold data maintains chronological order."""
        validator = WalkForwardValidator(sample_df, custom_config)
        folds = validator.generate_folds()

        if len(folds) > 0:
            fold = folds[0]
            df_train, df_val, df_test = validator.get_fold_data(fold)

            # Last train timestamp should be before first val timestamp
            train_last_idx = fold['train_end'] - 1
            val_first_idx = fold['val_start']

            assert train_last_idx < val_first_idx, \
                "Train data should end before validation data starts"

    def test_fold_data_is_independent_copy(self, sample_df, custom_config):
        """Test that extracted data is a copy (no data leakage through references)."""
        validator = WalkForwardValidator(sample_df, custom_config)
        folds = validator.generate_folds()

        if len(folds) > 0:
            fold = folds[0]
            df_train, df_val, df_test = validator.get_fold_data(fold)

            # Modify the copy
            original_value = df_train.iloc[0]['Close']
            df_train.iloc[0, df_train.columns.get_loc('Close')] = -9999

            # Original should be unchanged
            assert sample_df.iloc[fold['train_start']]['Close'] == original_value, \
                "Extracted data should be a copy, not a view"


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_insufficient_data_for_min_folds(self):
        """Test behavior when data is insufficient for minimum folds."""
        # Create very small DataFrame
        small_df = pd.DataFrame({
            'Close': [100] * 1000,
            'Open': [100] * 1000,
            'High': [101] * 1000,
            'Low': [99] * 1000,
            'Volume': [1000] * 1000
        })

        config = {
            'train_window_bars': 500,
            'validation_window_bars': 200,
            'test_window_bars': 200,
            'step_size_bars': 200,
            'purge_gap_bars': 50,
            'min_folds': 5,  # Requires more data than we have
            'max_folds': 10,
            'strategy': 'rolling',
            'early_stop_degradation_threshold': 0.3,
        }

        validator = WalkForwardValidator(small_df, config)
        folds = validator.generate_folds()

        # Should still work but generate fewer folds
        assert len(folds) < 5, "Should generate fewer than min_folds when data is insufficient"

    def test_empty_dataframe_handling(self):
        """Test that empty DataFrame is handled gracefully."""
        empty_df = pd.DataFrame()

        validator = WalkForwardValidator(empty_df, WALK_FORWARD_CONFIG)
        folds = validator.generate_folds()

        assert len(folds) == 0, "Empty DataFrame should produce 0 folds"

    def test_very_large_purge_gap(self, small_df):
        """Test behavior with purge gap larger than available data."""
        config = {
            'train_window_bars': 4000,
            'validation_window_bars': 1000,
            'test_window_bars': 500,
            'step_size_bars': 500,
            'purge_gap_bars': 5000,  # Larger than reasonable
            'min_folds': 1,
            'max_folds': 10,
            'strategy': 'rolling',
            'early_stop_degradation_threshold': 0.3,
        }

        validator = WalkForwardValidator(small_df, config)
        folds = validator.generate_folds()

        # Should handle gracefully (possibly 0 folds)
        assert isinstance(folds, list), "Should return a list even with bad config"


# =============================================================================
# TEST: CONFIGURATION VALIDATION
# =============================================================================

class TestConfigValidation:
    """Tests for walk-forward configuration validation."""

    def test_default_config_is_valid(self):
        """Test that default WALK_FORWARD_CONFIG is valid."""
        config = WALK_FORWARD_CONFIG

        assert config['train_window_bars'] > 0
        assert config['validation_window_bars'] > 0
        assert config['test_window_bars'] > 0
        assert config['step_size_bars'] > 0
        assert config['purge_gap_bars'] >= 0
        assert config['min_folds'] >= 1
        assert config['max_folds'] >= config['min_folds']
        assert config['strategy'] in ['rolling', 'expanding', 'anchored']
        assert 0 < config['early_stop_degradation_threshold'] < 1

    def test_train_window_larger_than_val_test(self):
        """Test that train window is typically larger than val/test."""
        config = WALK_FORWARD_CONFIG

        # Train should be larger for meaningful learning
        assert config['train_window_bars'] > config['validation_window_bars'], \
            "Train window should be larger than validation window"
        assert config['train_window_bars'] > config['test_window_bars'], \
            "Train window should be larger than test window"


# =============================================================================
# TEST: RESULTS AGGREGATION
# =============================================================================

class TestResultsAggregation:
    """Tests for aggregating results across folds."""

    def test_robust_stats_calculation(self):
        """Test that robust statistics are calculated correctly."""
        # Simulate fold results
        fold_results = [
            {'test_sharpe': 1.5, 'test_return': 0.10},
            {'test_sharpe': 1.8, 'test_return': 0.12},
            {'test_sharpe': 1.2, 'test_return': 0.08},
            {'test_sharpe': 10.0, 'test_return': 0.50},  # Outlier
        ]

        sharpes = [r['test_sharpe'] for r in fold_results]

        median_sharpe = np.median(sharpes)
        mean_sharpe = np.mean(sharpes)

        # Median should be more robust to outlier
        assert median_sharpe < mean_sharpe, \
            "Median should be less affected by outlier than mean"

        # Median should be around 1.5-1.8 range
        assert 1.0 < median_sharpe < 2.0, \
            f"Median Sharpe ({median_sharpe}) should be in reasonable range"

    def test_stability_score_calculation(self):
        """Test that stability score penalizes inconsistent performance."""
        # Stable results (low variance)
        stable_sharpes = [1.5, 1.6, 1.4, 1.5]
        stable_cv = np.std(stable_sharpes) / np.mean(stable_sharpes)
        stable_score = max(0, 1 - stable_cv)

        # Unstable results (high variance)
        unstable_sharpes = [0.5, 2.5, 1.0, 2.0]
        unstable_cv = np.std(unstable_sharpes) / np.mean(unstable_sharpes)
        unstable_score = max(0, 1 - unstable_cv)

        assert stable_score > unstable_score, \
            "Stable performance should have higher stability score"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
