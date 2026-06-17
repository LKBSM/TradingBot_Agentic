import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import os
import logging
from typing import Tuple, List, Dict, Optional, Any
import sys
from enum import Enum, auto

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from stable_baselines3.common.callbacks import BaseCallback
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.console import Console
from rich.table import Table

# Configuration du logging
logging.basicConfig(level=config.LOGGING_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS - Calcul des Métriques
# ============================================================================

def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Calcule le Maximum Drawdown."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    return np.max(drawdown)


def calculate_sharpe_ratio(daily_returns: np.ndarray) -> float:
    """Calcule le Sharpe Ratio annualisé."""
    risk_free_rate = config.RISK_FREE_RATE
    trading_days = config.TRADING_DAYS_YEAR
    if daily_returns.std() == 0:
        return 0.0
    annualized_return = np.mean(daily_returns) * trading_days
    annualized_volatility = np.std(daily_returns) * np.sqrt(trading_days)
    return (annualized_return - risk_free_rate) / annualized_volatility


def calculate_annualized_return(daily_returns: np.ndarray) -> float:
    """Calcule le retour annualisé."""
    return np.mean(daily_returns) * config.TRADING_DAYS_YEAR


def calculate_sortino_ratio(daily_returns: np.ndarray) -> float:
    """Calcule le Sortino Ratio (focus sur downside risk)."""
    risk_free_rate = config.RISK_FREE_RATE
    trading_days = config.TRADING_DAYS_YEAR
    target_returns = daily_returns - (risk_free_rate / trading_days)
    downside_returns = target_returns[target_returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    annualized_return = calculate_annualized_return(daily_returns)
    downside_volatility = np.std(downside_returns) * np.sqrt(trading_days)
    return (annualized_return - risk_free_rate) / downside_volatility


# ============================================================================
# CALLBACK: Rich Progress Bar
# ============================================================================

class RichProgressBarCallback(BaseCallback):
    """Barre de progression colorée pour l'entraînement."""

    def __init__(self, total_timesteps: int, verbose: int = 0, **kwargs):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress = None
        self.task_id = None

    def _on_training_start(self) -> None:
        console = Console()
        self.progress = Progress(
            SpinnerColumn(style="bold green"),
            TextColumn("[bold cyan]Training Progress:"),
            BarColumn(bar_width=None, style="bright_magenta", complete_style="bright_green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True
        )
        self.progress.start()
        self.task_id = self.progress.add_task("Training", total=self.total_timesteps)

    def _on_step(self) -> bool:
        self.progress.update(self.task_id, completed=self.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        if self.model is not None and hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            # Afficher stats détaillées tous les 10 rollouts
            if len(self.model.ep_info_buffer) >= 10:
                recent_rewards = [ep['r'] for ep in list(self.model.ep_info_buffer)[-10:]]
                avg_reward = np.mean(recent_rewards)
                self.progress.update(
                    self.task_id, 
                    description=f"🤖 Training | Avg Reward (10 ep): {avg_reward:.2f}"
                )
        
        # Log tous les 50k steps
        if self.num_timesteps % 50000 == 0 and self.num_timesteps > 0:
            logger.info(f"Checkpoint: {self.num_timesteps:,} steps | Progress: {(self.num_timesteps/self.total_timesteps)*100:.1f}%")
        
        if self.model is not None and hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
            last_ep_reward = self.model.ep_info_buffer[-1]['r']
            self.progress.update(self.task_id, description=f"🤖 Last Reward: {last_ep_reward:.2f}")

    def _on_training_end(self) -> None:
        self.progress.stop()


# ============================================================================
# CALLBACK: Early Stopping
# ============================================================================

class EarlyStoppingCallback(BaseCallback):
    """Arrête l'entraînement si le Sharpe Ratio stagne."""

    def __init__(self, eval_env, eval_freq: int = 5000, patience: int = 3,
                 min_delta: float = 0.05, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_delta = min_delta

        self.best_sharpe = -np.inf
        self.wait = 0
        self.stopped_step = 0
        self.best_model_path = None

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        sharpe = self._evaluate_sharpe()

        if self.verbose > 0:
            logger.info(f"Step {self.n_calls}: Validation Sharpe = {sharpe:.2f} (Best: {self.best_sharpe:.2f})")

        if sharpe > self.best_sharpe + self.min_delta:
            self.best_sharpe = sharpe
            self.wait = 0
            self.best_model_path = os.path.join(config.MODEL_DIR, f"best_model_sharpe_{sharpe:.2f}.zip")
            self.model.save(self.best_model_path)
            if self.verbose > 0:
                logger.info(f"Nouveau meilleur modele sauvegarde: {self.best_model_path}")
        else:
            self.wait += 1
            if self.verbose > 0:
                logger.warning(f"Pas d'amelioration ({self.wait}/{self.patience})")

        if self.wait >= self.patience:
            self.stopped_step = self.n_calls
            if self.verbose > 0:
                logger.warning(f"Early Stopping a step {self.stopped_step} | Meilleur Sharpe: {self.best_sharpe:.2f}")
            return False

        return True

    def _evaluate_sharpe(self) -> float:
        """Calcule le Sharpe Ratio sur l'env de validation."""
        obs, info = self.eval_env.reset()
        done = False
        portfolio_values = [float(info['net_worth'])]

        while not done:
            # v6: MaskablePPO requires action_masks for correct prediction
            predict_kwargs = {'deterministic': True}
            if hasattr(self.eval_env, 'action_masks'):
                predict_kwargs['action_masks'] = self.eval_env.action_masks()
            action, _ = self.model.predict(obs, **predict_kwargs)
            obs, reward, done, truncated, info = self.eval_env.step(int(action))
            portfolio_values.append(float(info['net_worth']))
            done = done or truncated

        portfolio_values = np.array(portfolio_values, dtype=float)
        if len(portfolio_values) < 2:
            return -np.inf

        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        return calculate_sharpe_ratio(daily_returns)


# ============================================================================
# CLASSE PRINCIPALE: AgentTrainer
# ============================================================================

class AgentTrainer:
    """
    Gestionnaire d'entraînement pour l'agent PPO.

    Méthodes disponibles:
    - train_offline(): Entraînement initial
    - continue_training(): Reprendre un modèle existant
    - fine_tune_online(): Fine-tuning avec nouvelles données
    - train_multiple_runs(): Entraînements multiples avec seeds différentes
    """

    def __init__(self, df_historical: pd.DataFrame, train_ratio: float = 0.8):
        from src.environment.environment import TradingEnv

        self.model_dir = config.MODEL_DIR
        self.df_historical = df_historical
        self.train_ratio = train_ratio

        # Calculate training split index to prevent data leakage in scaler
        self.train_split_idx = int(len(df_historical) * train_ratio)

        # Create training environment with scaler fit ONLY on training portion
        # This prevents data leakage - scaler doesn't see validation/test data
        self.env_train = TradingEnv(
            df=df_historical,
            enable_logging=True,
            scaler_fit_end_idx=self.train_split_idx
        )

        os.makedirs(self.model_dir, exist_ok=True)
        self.agent = None

    # ========================================================================
    # MÉTHODE 1: Entraînement Initial (Offline)
    # ========================================================================

    def train_offline(self, total_timesteps: int, use_early_stopping: bool = True,
                      seed: int = None) -> PPO:
        """
        Entraînement offline avec hyperparamètres du config et early stopping.

        Args:
            total_timesteps: Nombre de steps d'entraînement
            use_early_stopping: Activer l'arrêt anticipé si stagnation
            seed: Seed pour reproductibilité (optionnel)

        Returns:
            PPO: Agent entraîné
        """

        logger.info(f"Starting Offline Training ({total_timesteps:,} timesteps)...")

        # Définir la seed si fournie
        if seed is not None:
            np.random.seed(seed)
            import torch
            torch.manual_seed(seed)
            logger.info(f"Seed definie: {seed}")

        # Créer l'agent PPO avec hyperparamètres du config
        self.agent = PPO(
            'MlpPolicy',
            self.env_train,
            verbose=0,
            seed=seed,
            tensorboard_log=os.path.join(self.model_dir, "tensorboard_logs"),
            **config.MODEL_HYPERPARAMETERS
        )

        callbacks = []

        # Ajouter la barre de progression
        callbacks.append(RichProgressBarCallback(total_timesteps=total_timesteps))

        # Ajouter early stopping si activé
        if use_early_stopping:
            df_val = self.df_historical.iloc[self.train_split_idx:].copy()

            from src.environment.environment import TradingEnv
            # CRITICAL: Use pre_fitted_scaler from training env to prevent data leakage
            env_val = TradingEnv(
                df=df_val,
                pre_fitted_scaler=self.env_train.scaler
            )

            callbacks.append(EarlyStoppingCallback(
                eval_env=env_val,
                eval_freq=50000,  # Less frequent eval
                patience=50,  # Maximum patience for Railway training
                min_delta=0.05,
                verbose=1
            ))

        # Lancer l'entraînement
        self.agent.learn(total_timesteps=total_timesteps, callback=callbacks)

        # Sauvegarder le modèle final
        model_path = os.path.join(self.model_dir, f"model_offline_final.zip")
        self.agent.save(model_path)
        logger.info(f"Training complete. Model saved to: {model_path}")

        return self.agent

    # ========================================================================
    # MÉTHODE 2: Reprendre l'Entraînement d'un Modèle Existant
    # ========================================================================

    def continue_training(self, model_path: str, additional_timesteps: int,
                          reset_timesteps: bool = False) -> PPO:
        """
        Reprend l'entraînement d'un modèle déjà sauvegardé.

        Args:
            model_path: Chemin vers le modèle .zip à charger
            additional_timesteps: Nombre de steps supplémentaires
            reset_timesteps: Si True, remet le compteur de timesteps à 0

        Returns:
            PPO: Agent avec entraînement continué
        """

        # Vérifier que le modèle existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Modèle introuvable: {model_path}")

        logger.info(f"Chargement du modele: {model_path}")

        # Charger le modèle existant
        self.agent = PPO.load(model_path, env=self.env_train)

        logger.info(f"Continuation de l'entrainement ({additional_timesteps:,} steps supplementaires)...")

        # Créer callback
        callbacks = [RichProgressBarCallback(total_timesteps=additional_timesteps)]

        # Reprendre l'entraînement
        self.agent.learn(
            total_timesteps=additional_timesteps,
            reset_num_timesteps=reset_timesteps,
            callback=callbacks
        )

        # Sauvegarder le modèle continué
        new_model_path = model_path.replace('.zip', '_continued.zip')
        self.agent.save(new_model_path)
        logger.info(f"Modele continue sauvegarde: {new_model_path}")

        return self.agent

    # ========================================================================
    # MÉTHODE 3: Fine-Tuning avec Nouvelles Données (Online Learning)
    # ========================================================================

    def fine_tune_online(self, df_new_data: pd.DataFrame,
                         base_model_path: str = None,
                         fine_tune_timesteps: int = None,
                         use_original_scaler: bool = True) -> PPO:
        """
        Fine-tune un modèle pré-entraîné avec de nouvelles données.
        Utile pour adapter le bot à de nouvelles conditions de marché.

        Args:
            df_new_data: Nouvelles données (ex: données récentes de 2025)
            base_model_path: Chemin du modèle de base (si None, cherche le dernier)
            fine_tune_timesteps: Steps de fine-tuning (défaut: config.TOTAL_TIMESTEPS_ONLINE)
            use_original_scaler: If True, use scaler from original training (recommended)

        Returns:
            PPO: Agent fine-tuné
        """

        if fine_tune_timesteps is None:
            fine_tune_timesteps = getattr(config, 'TOTAL_TIMESTEPS_ONLINE', 10000)

        # Charger le modèle de base
        if base_model_path is None:
            base_model_path = os.path.join(self.model_dir, "model_offline_final.zip")

        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"❌ Modèle de base introuvable: {base_model_path}")

        logger.info(f"Chargement du modele de base: {base_model_path}")
        self.agent = PPO.load(base_model_path)

        # Créer un nouvel environnement avec les nouvelles données
        from src.environment.environment import TradingEnv

        if use_original_scaler and self.env_train is not None and self.env_train.scaler is not None:
            # Use original scaler to maintain consistent feature scaling
            env_new = TradingEnv(
                df=df_new_data,
                enable_logging=True,
                pre_fitted_scaler=self.env_train.scaler
            )
            logger.info("Using original training scaler for fine-tuning (no data leakage)")
        else:
            # Fit new scaler on all new data (acceptable for completely new regime)
            logger.warning("Fitting new scaler on fine-tune data - ensure this is intentional")
            env_new = TradingEnv(df=df_new_data, enable_logging=True)

        # Changer l'environnement de l'agent
        self.agent.set_env(env_new)

        logger.info(f"Fine-tuning sur nouvelles donnees ({fine_tune_timesteps:,} steps) | Nouvelles donnees: {len(df_new_data)} bars")

        # Fine-tuning
        callbacks = [RichProgressBarCallback(total_timesteps=fine_tune_timesteps)]
        self.agent.learn(
            total_timesteps=fine_tune_timesteps,
            reset_num_timesteps=False,
            callback=callbacks
        )

        # Sauvegarder le modèle fine-tuné
        fine_tuned_path = os.path.join(self.model_dir, "model_fine_tuned.zip")
        self.agent.save(fine_tuned_path)
        logger.info(f"Modele fine-tune sauvegarde: {fine_tuned_path}")

        return self.agent

    # ========================================================================
    # MÉTHODE 4: Entraînements Multiples avec Seeds Différentes (ROBUSTESSE)
    # ========================================================================

    def train_multiple_runs(self, n_runs: int = 5,
                            timesteps_per_run: int = None,
                            cumulative: bool = True) -> List[Dict]:
        """
        Lance plusieurs entraînements successifs avec des seeds différentes.

        DEUX MODES:
        1. Cumulative (défaut): Chaque run continue le précédent
           → Apprentissage progressif, modèle de plus en plus robuste

        2. Independent: Chaque run repart de zéro avec une seed différente
           → Pour tester la variance et trouver le meilleur modèle

        Args:
            n_runs: Nombre d'entraînements à lancer
            timesteps_per_run: Steps par run (défaut: config.TRAINING_TIMESTEPS / n_runs)
            cumulative: Si True, chaque run continue le précédent

        Returns:
            List[Dict]: Résultats de chaque run avec métriques
        """

        if timesteps_per_run is None:
            timesteps_per_run = config.TRAINING_TIMESTEPS // n_runs

        logger.info(
            f"Lancement de {n_runs} Runs d'Entrainement | "
            f"Mode: {'Cumulatif (chaque run continue le precedent)' if cumulative else 'Independent (chaque run repart de zero)'} | "
            f"Timesteps par run: {timesteps_per_run:,} | "
            f"Total timesteps: {timesteps_per_run * n_runs:,}"
        )

        results = []
        base_seed = getattr(config, 'RANDOM_SEED', 42)

        # Préparer données de test (uses consistent split from __init__)
        df_test = self.df_historical.iloc[self.train_split_idx:].copy()

        for run in range(1, n_runs + 1):
            logger.info(f"{'=' * 70}\nRUN {run}/{n_runs}\n{'=' * 70}")

            # Seed différente pour chaque run
            current_seed = base_seed + run

            # MODE CUMULATIF: Charger le modèle du run précédent
            if cumulative and run > 1:
                previous_model_path = os.path.join(
                    self.model_dir,
                    f"model_run_{run - 1}_seed_{base_seed + run - 1}.zip"
                )

                logger.info(f"Chargement du modele du run precedent...")
                self.agent = PPO.load(previous_model_path, env=self.env_train)

                # Changer la seed pour l'exploration
                self.agent.seed = current_seed

                # Continuer l'entraînement
                callbacks = [RichProgressBarCallback(total_timesteps=timesteps_per_run)]
                self.agent.learn(
                    total_timesteps=timesteps_per_run,
                    reset_num_timesteps=False,
                    callback=callbacks
                )

            # MODE INDÉPENDANT ou PREMIER RUN: Nouveau modèle
            else:
                self.agent = self.train_offline(
                    total_timesteps=timesteps_per_run,
                    use_early_stopping=False,  # Désactiver pour comparaison équitable
                    seed=current_seed
                )

            # Sauvegarder le modèle de ce run
            run_model_path = os.path.join(
                self.model_dir,
                f"model_run_{run}_seed_{current_seed}.zip"
            )
            self.agent.save(run_model_path)

            # Évaluer le modèle sur les données de test
            logger.info(f"Evaluation du Run {run}...")
            # Pass the training scaler to prevent data leakage in evaluation
            metrics = evaluate_agent(self.agent, df_test, pre_fitted_scaler=self.env_train.scaler)

            # Stocker les résultats
            run_results = {
                'run': run,
                'seed': current_seed,
                'model_path': run_model_path,
                'cumulative_return': metrics[0],
                'sharpe_ratio': metrics[1],
                'sortino_ratio': metrics[2],
                'calmar_ratio': metrics[3],
                'max_drawdown': metrics[4],
                'timesteps_trained': timesteps_per_run * (run if cumulative else 1)
            }
            results.append(run_results)

            logger.info(f"Run {run} termine: Sharpe Ratio: {metrics[1]:.2f} | Max Drawdown: {metrics[4]:.2%}")

        # ====================================================================
        # RAPPORT FINAL
        # ====================================================================

        logger.info(f"{'=' * 70}\nRAPPORT FINAL DES {n_runs} RUNS\n{'=' * 70}")

        df_results = pd.DataFrame(results)

        table_lines = [f"Comparaison des {n_runs} Runs:"]
        table_lines.append(f"{'Run':>4} | {'Seed':>6} | {'Sharpe':>7} | {'Sortino':>7} | {'MDD':>8} | {'Return':>8}")
        table_lines.append("-" * 55)
        for _, row in df_results.iterrows():
            table_lines.append(
                f"{int(row['run']):>4} | {int(row['seed']):>6} | {row['sharpe_ratio']:>7.2f} | "
                f"{row['sortino_ratio']:>7.2f} | {row['max_drawdown']:>7.2%} | {row['cumulative_return']:>7.2%}"
            )
        logger.info("\n".join(table_lines))

        logger.info(
            f"STATISTIQUES GLOBALES: "
            f"Sharpe Moyen: {df_results['sharpe_ratio'].mean():.2f} (+/-{df_results['sharpe_ratio'].std():.2f}) | "
            f"Meilleur Sharpe: {df_results['sharpe_ratio'].max():.2f} (Run {df_results['sharpe_ratio'].idxmax() + 1}) | "
            f"Pire Sharpe: {df_results['sharpe_ratio'].min():.2f} (Run {df_results['sharpe_ratio'].idxmin() + 1})"
        )

        # Identifier le meilleur modèle
        best_run_idx = df_results['sharpe_ratio'].idxmax()
        best_model_path = df_results.loc[best_run_idx, 'model_path']

        logger.info(
            f"MEILLEUR MODELE: Run: {df_results.loc[best_run_idx, 'run']} | "
            f"Sharpe: {df_results.loc[best_run_idx, 'sharpe_ratio']:.2f} | "
            f"Chemin: {best_model_path}"
        )

        # Sauvegarder le meilleur modèle séparément
        best_model_production_path = os.path.join(self.model_dir, "model_BEST_production.zip")
        import shutil
        shutil.copy(best_model_path, best_model_production_path)
        logger.info(f"Meilleur modele copie vers: {best_model_production_path}")

        # Sauvegarder les résultats en CSV
        results_csv_path = os.path.join(config.RESULTS_DIR, 'multiple_runs_results.csv')
        df_results.to_csv(results_csv_path, index=False)
        logger.info(f"Resultats sauvegardes: {results_csv_path}")

        return results

    # ========================================================================
    # MÉTHODE 5: Entraînement Sophistiqué (NOUVEAU - MAXIMUM PERFORMANCE)
    # ========================================================================

    def train_sophisticated(
        self,
        strategy: str = "full_pipeline",
        total_timesteps: Optional[int] = None,
        use_curriculum: bool = True,
        use_ensemble: bool = True,
        use_meta_learning: bool = True,
        n_ensemble_models: int = 5,
        seed: Optional[int] = None
    ) -> Tuple[PPO, Dict[str, Any]]:
        """
        Entraînement sophistiqué utilisant le nouveau système avancé.

        Ce système combine:
        1. CURRICULUM LEARNING: Apprentissage progressif en 4 phases
           - BASE: Apprend les patterns de marché de base
           - ENRICHED: Intègre les signaux des agents comme observations
           - SOFT: Penalties douces pour actions rejetées
           - PRODUCTION: Contraintes dures identiques à la production

        2. ENSEMBLE TRAINING: Diversité des modèles pour robustesse
           - Hyperparamètres diversifiés
           - Objectifs diversifiés (Sharpe vs Sortino vs Calmar)
           - Combinaison pondérée par performance

        3. META-LEARNING: Adaptation rapide aux régimes de marché
           - MAML-inspired pour initialisation optimale
           - Adaptation online aux changements de régime
           - Détection automatique des régimes

        AVANTAGE CRITIQUE: Résout le problème de domain shift!
        L'environnement d'entraînement est identique à la production.

        Args:
            strategy: Stratégie d'entraînement
                - "curriculum_only": Seulement curriculum learning
                - "ensemble_only": Seulement ensemble training
                - "meta_only": Seulement meta-learning
                - "curriculum_ensemble": Curriculum + Ensemble
                - "curriculum_meta": Curriculum + Meta
                - "full_pipeline": Tous les composants (recommandé)
            total_timesteps: Total timesteps (défaut: config.TRAINING_TIMESTEPS)
            use_curriculum: Activer curriculum learning
            use_ensemble: Activer ensemble training
            use_meta_learning: Activer meta-learning
            n_ensemble_models: Nombre de modèles dans l'ensemble
            seed: Seed pour reproductibilité

        Returns:
            Tuple[PPO, Dict]: (meilleur modèle, résumé d'entraînement)
        """
        from src.training.sophisticated_trainer import (
            SophisticatedTrainer,
            SophisticatedTrainerConfig,
            TrainingStrategy
        )
        from src.training.ensemble_trainer import EnsembleConfig
        from src.training.curriculum_trainer import CurriculumConfig
        from src.training.meta_learner import MetaLearnerConfig

        if total_timesteps is None:
            total_timesteps = config.TOTAL_TIMESTEPS_PER_BOT

        logger.info(f"Starting Sophisticated Training ({total_timesteps:,} timesteps) | Strategy: {strategy}")

        # Map string strategy to enum
        strategy_map = {
            "curriculum_only": TrainingStrategy.CURRICULUM_ONLY,
            "ensemble_only": TrainingStrategy.ENSEMBLE_ONLY,
            "meta_only": TrainingStrategy.META_ONLY,
            "curriculum_ensemble": TrainingStrategy.CURRICULUM_ENSEMBLE,
            "curriculum_meta": TrainingStrategy.CURRICULUM_META,
            "full_pipeline": TrainingStrategy.FULL_PIPELINE
        }
        training_strategy = strategy_map.get(strategy.lower(), TrainingStrategy.FULL_PIPELINE)

        # Calculate allocations based on enabled components
        n_components = sum([use_curriculum, use_ensemble, use_meta_learning])
        if n_components == 0:
            logger.warning("No components enabled, defaulting to curriculum_only")
            training_strategy = TrainingStrategy.CURRICULUM_ONLY
            curriculum_fraction = 1.0
            ensemble_fraction = 0.0
            meta_fraction = 0.0
        else:
            curriculum_fraction = 0.4 if use_curriculum else 0.0
            ensemble_fraction = 0.35 if use_ensemble else 0.0
            meta_fraction = 0.25 if use_meta_learning else 0.0

            # Normalize
            total = curriculum_fraction + ensemble_fraction + meta_fraction
            curriculum_fraction /= total
            ensemble_fraction /= total
            meta_fraction /= total

        # Create configuration
        trainer_config = SophisticatedTrainerConfig(
            strategy=training_strategy,
            total_timesteps=total_timesteps,
            curriculum_fraction=curriculum_fraction,
            ensemble_fraction=ensemble_fraction,
            meta_fraction=meta_fraction,
            base_hyperparams=config.MODEL_HYPERPARAMETERS,
            base_save_dir=os.path.join(self.model_dir, 'sophisticated'),
            tensorboard_log_dir=os.path.join(self.model_dir, 'sophisticated', 'logs'),
            verbose=1
        )

        # Update ensemble config if needed
        if use_ensemble:
            trainer_config.ensemble_config.n_models = n_ensemble_models

        # Split data into train/val/test
        total_len = len(self.df_historical)
        train_end = int(total_len * config.TRAIN_RATIO)
        val_end = int(total_len * (config.TRAIN_RATIO + config.VAL_RATIO))

        df_train = self.df_historical.iloc[:train_end].copy()
        df_val = self.df_historical.iloc[train_end:val_end].copy()
        df_test = self.df_historical.iloc[val_end:].copy()

        logger.info(f"Data splits: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

        # Create trainer
        trainer = SophisticatedTrainer(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            config=trainer_config
        )

        # Run training
        results = trainer.train(seed=seed)

        # Load best model
        best_model = PPO.load(results.best_model_path, env=self.env_train)

        # Store the agent
        self.agent = best_model

        # Create summary
        summary = {
            'strategy': results.strategy,
            'total_timesteps': results.total_timesteps,
            'training_duration_hours': results.training_duration_seconds / 3600,
            'final_sharpe': results.final_sharpe,
            'final_win_rate': results.final_win_rate,
            'final_max_drawdown': results.final_max_drawdown,
            'final_cumulative_return': results.final_cumulative_return,
            'best_model_path': results.best_model_path,
            'has_ensemble': results.has_ensemble,
            'has_meta_adapter': results.has_meta_adapter,
            'curriculum_results': results.curriculum_results,
            'ensemble_results': results.ensemble_results,
            'meta_results': results.meta_results
        }

        logger.info(
            f"Sophisticated Training Complete! "
            f"Final Sharpe: {results.final_sharpe:.2f} | "
            f"Final Win Rate: {results.final_win_rate:.1%} | "
            f"Best Model: {results.best_model_path}"
        )

        return best_model, summary


# ============================================================================
# FONCTION D'ÉVALUATION
# ============================================================================

def evaluate_agent(agent: PPO, df_test: pd.DataFrame,
                   pre_fitted_scaler=None) -> Tuple[float, float, float, float, float]:
    """
    Évalue l'agent et retourne les métriques clés.

    Args:
        agent: The trained PPO agent
        df_test: Test data DataFrame
        pre_fitted_scaler: Scaler fitted on training data (CRITICAL to prevent data leakage)

    Returns:
        tuple: (cumulative_return, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown)
    """
    from src.environment.environment import TradingEnv

    df_test = df_test.reset_index(drop=True)

    if pre_fitted_scaler is not None:
        # Use training scaler to prevent data leakage
        env_test = TradingEnv(df=df_test, pre_fitted_scaler=pre_fitted_scaler)
    else:
        # Legacy behavior with warning
        logger.warning("evaluate_agent: No pre_fitted_scaler provided - potential data leakage!")
        env_test = TradingEnv(df=df_test)

    obs, info = env_test.reset()
    done = False
    portfolio_values = [float(info['net_worth'])]

    while not done:
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env_test.step(int(action))
        portfolio_values.append(float(info['net_worth']))
        done = done or truncated

    portfolio_values = np.array(portfolio_values, dtype=float)

    if len(portfolio_values) < 2 or portfolio_values[-1] == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    cumulative_return = float((portfolio_values[-1] / portfolio_values[0]) - 1)
    max_drawdown = float(calculate_max_drawdown(portfolio_values))
    sharpe_ratio = float(calculate_sharpe_ratio(daily_returns))
    sortino_ratio = float(calculate_sortino_ratio(daily_returns))
    annualized_return = float(calculate_annualized_return(daily_returns))
    calmar_ratio = float(annualized_return / max_drawdown if max_drawdown > 0 else 0.0)

    return cumulative_return, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown


# ============================================================================
# MAIN - EXEMPLES D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    # Mock data pour test
    data_points = 10000
    prices = 100 + np.cumsum(np.random.randn(data_points) * 0.1)
    df_data = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=data_points, freq='15min'),
        'Open': prices + np.random.uniform(-0.1, 0.1, data_points),
        'High': prices + np.random.uniform(0.1, 0.2, data_points),
        'Low': prices - np.random.uniform(0.1, 0.2, data_points),
        'Close': prices,
        'Volume': np.random.randint(100, 1000, data_points)
    }).set_index('Date')

    df_train = df_data.iloc[:8000].copy()
    df_test = df_data.iloc[8000:].copy()

    # ========================================================================
    # EXEMPLE 1: Entraînement Simple
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXEMPLE 1: Entraînement Simple")
    print("=" * 70)

    trainer = AgentTrainer(df_historical=df_train)
    agent = trainer.train_offline(total_timesteps=50000, use_early_stopping=True)

    # ========================================================================
    # EXEMPLE 2: Multiple Runs (RECOMMANDÉ pour Production)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXEMPLE 2: Multiple Runs (5 entraînements)")
    print("=" * 70)

    trainer = AgentTrainer(df_historical=df_train)
    results = trainer.train_multiple_runs(
        n_runs=5,
        timesteps_per_run=20000,
        cumulative=True  # Apprentissage cumulatif
    )

    # ========================================================================
    # EXEMPLE 3: Entraînement Sophistiqué (NOUVEAU - MAXIMUM PERFORMANCE)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXEMPLE 3: Entraînement Sophistiqué (Full Pipeline)")
    print("=" * 70)
    print("Ce système combine curriculum learning, ensemble training, et meta-learning")
    print("pour maximiser la performance sur les marchés financiers.")
    print("=" * 70)

    # Uncomment to run sophisticated training:
    # trainer = AgentTrainer(df_historical=df_train)
    # agent, summary = trainer.train_sophisticated(
    #     strategy="full_pipeline",
    #     total_timesteps=500000,  # Reduced for demo
    #     use_curriculum=True,
    #     use_ensemble=True,
    #     use_meta_learning=True,
    #     n_ensemble_models=3,  # Reduced for demo
    #     seed=42
    # )
    # print(f"Training Summary: {summary}")




