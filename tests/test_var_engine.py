# =============================================================================
# Tests for Sprint 1: VaR Engine Integration
# =============================================================================
import sys
import os
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.performance.vectorized_risk import VectorizedRiskCalculator
from src.risk.var_engine import VaREngine, VaRResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calc():
    return VectorizedRiskCalculator()


@pytest.fixture
def normal_returns():
    """Normally distributed returns (mean=0, std=0.01)."""
    np.random.seed(42)
    return np.random.normal(0, 0.01, 1000)


@pytest.fixture
def fat_tail_returns():
    """
    Leptokurtic (fat-tailed) returns simulating Gold.
    Student-t with df=4 produces excess kurtosis > 3.
    """
    np.random.seed(42)
    from scipy.stats import t as t_dist
    return t_dist.rvs(df=4, loc=0, scale=0.01, size=1000)


@pytest.fixture
def engine():
    return VaREngine(confidence=0.95, window=252, method='cornish_fisher')


# ===========================================================================
# 1. VectorizedRiskCalculator — Individual Method Tests
# ===========================================================================

class TestVaRMethods:

    def test_historical_var_positive(self, calc, normal_returns):
        var = calc.var_historical(normal_returns, 0.95)
        assert var > 0, "Historical VaR should be positive"

    def test_parametric_var_normal(self, calc, normal_returns):
        """For a normal dist, parametric VaR_95 ~ 1.645 * sigma."""
        var = calc.var_parametric(normal_returns, 0.95)
        sigma = np.std(normal_returns, ddof=1)
        expected = 1.645 * sigma
        assert abs(var - expected) / expected < 0.15, (
            f"Parametric VaR {var:.6f} should be close to 1.645*sigma={expected:.6f}"
        )

    def test_monte_carlo_var_positive(self, calc, normal_returns):
        var = calc.var_monte_carlo(normal_returns, 0.95, simulations=5000)
        assert var > 0, "Monte Carlo VaR should be positive"

    def test_cvar_exceeds_var(self, calc, normal_returns):
        """CVaR (Expected Shortfall) must always be >= VaR."""
        var = calc.var_historical(normal_returns, 0.95)
        cvar = calc.cvar(normal_returns, 0.95)
        assert cvar >= var, f"CVaR ({cvar}) must >= VaR ({var})"

    def test_cornish_fisher_var_positive(self, calc, normal_returns):
        var = calc.var_cornish_fisher(normal_returns, 0.95)
        assert var > 0, "Cornish-Fisher VaR should be positive"

    def test_cornish_fisher_higher_for_fat_tails(self, calc, fat_tail_returns):
        """
        Cornish-Fisher should produce higher VaR than parametric
        when the distribution has excess kurtosis (fat tails).
        """
        var_cf = calc.var_cornish_fisher(fat_tail_returns, 0.95)
        var_param = calc.var_parametric(fat_tail_returns, 0.95)
        assert var_cf > var_param, (
            f"Cornish-Fisher VaR ({var_cf:.6f}) should exceed parametric "
            f"VaR ({var_param:.6f}) for fat-tailed returns"
        )

    def test_all_methods_agree_on_normal(self, calc, normal_returns):
        """All 5 methods should agree within 30% on normal distributions."""
        results = {
            'historical': calc.var_historical(normal_returns, 0.95),
            'parametric': calc.var_parametric(normal_returns, 0.95),
            'monte_carlo': calc.var_monte_carlo(normal_returns, 0.95, simulations=10000),
            'cornish_fisher': calc.var_cornish_fisher(normal_returns, 0.95),
        }
        values = list(results.values())
        mean_var = np.mean(values)
        for name, val in results.items():
            deviation = abs(val - mean_var) / mean_var
            assert deviation < 0.30, (
                f"{name} VaR ({val:.6f}) deviates {deviation:.0%} from mean ({mean_var:.6f})"
            )

    def test_empty_returns(self, calc):
        assert calc.var_historical(np.array([]), 0.95) == 0.0
        assert calc.var_parametric(np.array([]), 0.95) == 0.0
        assert calc.cvar(np.array([]), 0.95) == 0.0
        assert calc.var_cornish_fisher(np.array([0.01]), 0.95) == 0.0  # < 10 obs

    def test_var_99_exceeds_var_95(self, calc, normal_returns):
        """99% VaR should be larger than 95% VaR."""
        var_95 = calc.var_historical(normal_returns, 0.95)
        var_99 = calc.var_historical(normal_returns, 0.99)
        assert var_99 > var_95


# ===========================================================================
# 2. VaREngine — Wrapper Tests
# ===========================================================================

class TestVaREngine:

    def test_engine_creation(self, engine):
        assert engine.confidence == 0.95
        assert engine.method == 'cornish_fisher'
        assert engine.buffer_size == 0
        assert not engine.is_ready

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown VaR method"):
            VaREngine(method='invalid_method')

    def test_invalid_confidence(self):
        with pytest.raises(ValueError, match="Confidence"):
            VaREngine(confidence=1.5)

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="Window"):
            VaREngine(window=5)

    def test_not_ready_until_min_observations(self, engine):
        for i in range(29):
            engine.update(np.random.normal(0, 0.01))
        assert not engine.is_ready

        engine.update(np.random.normal(0, 0.01))
        assert engine.is_ready

    def test_compute_returns_zeros_when_not_ready(self, engine):
        engine.update(0.01)
        result = engine.compute()
        assert result.var_95 == 0.0
        assert result.var_99 == 0.0

    def test_compute_with_data(self, engine, normal_returns):
        engine.update_batch(normal_returns)
        result = engine.compute()

        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.cvar_95 >= result.var_95
        assert result.method == 'cornish_fisher'
        assert result.window_size == 252  # capped by deque maxlen
        assert result.computation_time_ms > 0

    def test_rolling_window_caps(self, engine):
        """Buffer should not exceed window size."""
        np.random.seed(42)
        for _ in range(500):
            engine.update(np.random.normal(0, 0.01))
        assert engine.buffer_size == 252

    def test_update_batch(self, engine, normal_returns):
        engine.update_batch(normal_returns[:100])
        assert engine.buffer_size == 100
        assert engine.is_ready

    def test_compute_all_methods(self, engine, normal_returns):
        engine.update_batch(normal_returns)
        results = engine.compute_all_methods()

        assert set(results.keys()) == set(VaREngine.METHODS)
        for m, result in results.items():
            assert result.var_95 > 0, f"{m} VaR should be positive"
            assert result.method == m

    def test_method_override(self, engine, normal_returns):
        engine.update_batch(normal_returns)
        result = engine.compute(method='historical')
        assert result.method == 'historical'

    def test_last_result_cached(self, engine, normal_returns):
        assert engine.last_result is None
        engine.update_batch(normal_returns)
        result = engine.compute()
        assert engine.last_result is result

    def test_reset(self, engine, normal_returns):
        engine.update_batch(normal_returns)
        engine.compute()
        engine.reset()
        assert engine.buffer_size == 0
        assert engine.last_result is None
        assert not engine.is_ready

    def test_to_dict(self, engine, normal_returns):
        engine.update_batch(normal_returns)
        result = engine.compute()
        d = result.to_dict()
        assert 'var_95' in d
        assert 'var_99' in d
        assert 'cvar_95' in d
        assert 'method' in d


# ===========================================================================
# 3. Integration — Kill Switch VaR Breach
# ===========================================================================

class TestKillSwitchVaRIntegration:

    def test_var_breach_triggers_halt(self):
        """Kill switch should trigger on VaR breach."""
        from src.agents.kill_switch import KillSwitch, KillSwitchConfig, HaltLevel, HaltReason

        config = KillSwitchConfig(max_var_pct=0.02)
        ks = KillSwitch(config=config, enable_persistence=False)

        # Normal update — no breach
        level = ks.update(equity=100000, var_pct=0.015)
        assert level.value < HaltLevel.NEW_ONLY.value

        # VaR breach — should trigger
        level = ks.update(equity=100000, var_pct=0.025)
        assert level.value >= HaltLevel.NEW_ONLY.value
        assert ks.halt_reason == HaltReason.VAR_BREACH

    def test_no_var_no_breach(self):
        """If var_pct is not provided, VaR breaker should not trip."""
        from src.agents.kill_switch import KillSwitch, KillSwitchConfig, HaltLevel

        config = KillSwitchConfig(max_var_pct=0.02)
        ks = KillSwitch(config=config, enable_persistence=False)

        level = ks.update(equity=100000)
        assert level == HaltLevel.NONE


# ===========================================================================
# 4. Risk Manager — print() Removal Verification
# ===========================================================================

class TestRiskManagerNoPrint:

    def test_no_print_calls_in_risk_manager(self):
        """risk_manager.py must have zero print() calls."""
        import ast
        risk_manager_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'environment', 'risk_manager.py'
        )
        with open(risk_manager_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        print_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == 'print':
                    print_calls.append(node.lineno)

        assert len(print_calls) == 0, (
            f"Found print() calls at lines: {print_calls}. "
            "Use logger.critical/warning/info instead."
        )
