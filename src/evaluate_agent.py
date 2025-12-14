import pandas as pd
import numpy as np
import gymnasium as gym
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Tuple, Dict
import os
from rich.console import Console
from rich.table import Table

# --- Import Config and Custom Environment ---
import config
from src.environment.environment import TradingEnv

# --- Logging Setup ---
logging.basicConfig(level=config.LOGGING_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

console = Console()

# =============================================================================
# Helper Functions for KPIs
# =============================================================================

def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Calculates Maximum Drawdown (MDD)."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    return np.max(drawdown)

def calculate_annualized_return(daily_returns: np.ndarray) -> float:
    """Calculates Annualized Return."""
    trading_days = config.TRADING_DAYS_YEAR
    return np.mean(daily_returns) * trading_days

def calculate_sharpe_ratio(daily_returns: np.ndarray) -> float:
    """Calculates annualized Sharpe Ratio."""
    risk_free_rate = config.RISK_FREE_RATE
    trading_days = config.TRADING_DAYS_YEAR
    if np.std(daily_returns) == 0:
        return 0.0
    annualized_return = calculate_annualized_return(daily_returns)
    annualized_volatility = np.std(daily_returns) * np.sqrt(trading_days)
    return (annualized_return - risk_free_rate) / annualized_volatility

def calculate_sortino_ratio(daily_returns: np.ndarray) -> float:
    """Calculates Sortino Ratio (downside risk only)."""
    risk_free_rate = config.RISK_FREE_RATE
    trading_days = config.TRADING_DAYS_YEAR
    target_returns = daily_returns - (risk_free_rate / trading_days)
    downside_returns = target_returns[target_returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    annualized_return = calculate_annualized_return(daily_returns)
    downside_volatility = np.std(downside_returns) * np.sqrt(trading_days)
    return (annualized_return - risk_free_rate) / downside_volatility

# =============================================================================
# Environment Wrapper
# =============================================================================

def create_trading_environment(df: pd.DataFrame) -> gym.Env:
    """Creates an instance of the custom TradingEnv."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column.")
    env = TradingEnv(df=df.copy())
    return env

# =============================================================================
# Enhanced Evaluation Function (for multi-bot setup)
# =============================================================================

def evaluate_agent_advanced(agent: PPO, df_test: pd.DataFrame, bot_name: str = "Bot_1", save_results: bool = True) -> Dict[str, float]:
    """
    Evaluates a trained PPO agent on unseen data and computes all key metrics.
    Designed for multi-bot training setups (parallel evaluations).
    """
    console.rule(f"[bold cyan]📊 Evaluation Started for {bot_name}[/bold cyan]")
    env_test = create_trading_environment(df_test)
    obs, info = env_test.reset()
    done = False
    portfolio_values = [float(info["net_worth"])]

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        portfolio_values.append(float(info["net_worth"]))
        done = terminated or truncated

    portfolio_values = np.array(portfolio_values, dtype=float)
    if len(portfolio_values) < 2 or portfolio_values[-1] == 0:
        logging.warning(f"[{bot_name}] Evaluation failed: insufficient data.")
        return {}

    # --- Calculate KPIs ---
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    sortino_ratio = calculate_sortino_ratio(daily_returns)
    max_drawdown = calculate_max_drawdown(portfolio_values)
    annualized_return = calculate_annualized_return(daily_returns)
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    results = {
        "Bot": bot_name,
        "Cumulative Return": cumulative_return,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Max Drawdown": max_drawdown,
        "Annualized Return": annualized_return,
    }

    # --- Console Display ---
    table = Table(title=f"{bot_name} Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for key, val in results.items():
        if key != "Bot":
            val_str = f"{val:.2%}" if "Return" in key or "Drawdown" in key else f"{val:.2f}"
            table.add_row(key, val_str)

    console.print(table)

    # --- Save Results ---
    if save_results:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        results_path = os.path.join(config.RESULTS_DIR, "evaluation_summary.csv")

        df_new = pd.DataFrame([results])
        if os.path.exists(results_path):
            df_old = pd.read_csv(results_path)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new.to_csv(results_path, index=False)
        logging.info(f"[{bot_name}] Results saved to {results_path}")

    console.rule(f"[bold green]✅ Evaluation Complete for {bot_name}[/bold green]")
    return results

# =============================================================================
# Manual Test Section (Optional)
# =============================================================================

if __name__ == "__main__":
    # --- Mock test data ---
    data_points = 500
    prices = 100 + np.cumsum(np.random.randn(data_points) * 0.1)
    df = pd.DataFrame({
        "Date": pd.date_range(start="2024-01-01", periods=data_points, freq="15min"),
        "Close": prices,
        "Open": prices + np.random.uniform(-0.1, 0.1, data_points),
        "High": prices + np.random.uniform(0.1, 0.2, data_points),
        "Low": prices - np.random.uniform(0.1, 0.2, data_points),
        "Volume": np.random.randint(100, 1000, data_points),
    })

    # Example usage
    try:
        console.print("\n[bold yellow]--- MOCK EVALUATION ---[/bold yellow]")
        console.print("⚠️ Please load a trained PPO model to run a real evaluation.")
        # Example:
        # model = PPO.load("models/model_BEST_production.zip")
        # df_test = df.iloc[400:].copy()
        # evaluate_agent_advanced(model, df_test, bot_name="Bot_1")
    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")
