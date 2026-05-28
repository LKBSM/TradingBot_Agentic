# =============================================================================
# EWC REGULARIZATION — Elastic Weight Consolidation (Sprint 14)
# =============================================================================
# Prevents catastrophic forgetting during curriculum phase transitions.
#
# At the end of each phase:
#   1. Compute diagonal Fisher Information Matrix from recent rollouts
#   2. Store current weight snapshot as "anchor" (theta*)
#
# During training in the next phase:
#   3. Add EWC penalty: lambda/2 * sum(F_i * (theta_i - theta*_i)^2)
#
# This protects important weights learned in previous phases while allowing
# the network to adapt to new phase conditions (e.g., agent signals in
# ENRICHED phase without destroying market pattern weights from BASE).
#
# Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in
# neural networks", PNAS 2017.
#
# Usage:
#   ewc = EWCCallback(ewc_lambda=1000.0, fisher_samples=2048)
#   model.learn(callback=[curriculum_cb, ewc])
#   # At phase transition, call: ewc.snapshot(model)
# =============================================================================

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class EWCCallback(BaseCallback):
    """
    Elastic Weight Consolidation callback for SB3 PPO.

    Computes a diagonal Fisher Information approximation from the policy
    gradient and applies a quadratic penalty that anchors weights to their
    values at the end of the previous curriculum phase.

    Args:
        ewc_lambda: Regularization strength. Higher = stronger anchor.
            Typical range: 100-10000. Start with 1000.
        fisher_samples: Number of observations to sample for Fisher computation.
        verbose: Logging verbosity (0=silent, 1=info).
    """

    def __init__(
        self,
        ewc_lambda: float = 1000.0,
        fisher_samples: int = 2048,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples

        # Stored at each phase transition
        self._fisher_diag: Optional[Dict[str, torch.Tensor]] = None
        self._anchor_params: Optional[Dict[str, torch.Tensor]] = None
        self._phase_snapshots: List[int] = []  # timesteps when snapshots were taken
        self._ewc_active = False

    def _on_training_start(self) -> None:
        """Called once at the start of training."""
        pass

    def _on_step(self) -> bool:
        """Called after each env step. Returns True to continue training."""
        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout (n_steps collected).
        This is where we inject the EWC penalty into the policy loss.
        """
        if not self._ewc_active or self._fisher_diag is None:
            return

        # Add EWC penalty as additional loss via the optimizer
        self._apply_ewc_penalty()

    def snapshot(self, model=None) -> None:
        """
        Take a snapshot of the current policy weights and compute Fisher.

        Call this at each curriculum phase transition.

        Args:
            model: SB3 PPO model. If None, uses self.model from callback.
        """
        model = model or self.model
        if model is None:
            logger.warning("EWC snapshot called but no model available")
            return

        policy = model.policy

        # 1. Store current parameter values as anchor
        self._anchor_params = {
            name: param.data.clone()
            for name, param in policy.named_parameters()
            if param.requires_grad
        }

        # 2. Compute diagonal Fisher Information Matrix
        self._fisher_diag = self._compute_fisher(policy, model)

        self._ewc_active = True
        self._phase_snapshots.append(self.num_timesteps if hasattr(self, 'num_timesteps') else 0)

        n_params = sum(p.numel() for p in self._anchor_params.values())
        logger.info(
            f"EWC snapshot taken: {n_params:,} parameters anchored, "
            f"Fisher computed from {self.fisher_samples} samples"
        )

    def _compute_fisher(self, policy: nn.Module, model) -> Dict[str, torch.Tensor]:
        """
        Compute diagonal Fisher Information Matrix approximation.

        Uses the empirical Fisher: F_i = E[(d log pi(a|s) / d theta_i)^2]
        Approximated by sampling observations from the rollout buffer.
        """
        fisher = {
            name: torch.zeros_like(param)
            for name, param in policy.named_parameters()
            if param.requires_grad
        }

        # Sample observations from the rollout buffer
        rollout_buffer = model.rollout_buffer
        if rollout_buffer is None or rollout_buffer.pos == 0:
            logger.warning("EWC: Empty rollout buffer, skipping Fisher computation")
            return fisher

        # Get observations from buffer
        buffer_size = rollout_buffer.buffer_size * rollout_buffer.n_envs
        n_samples = min(self.fisher_samples, buffer_size)

        if n_samples == 0:
            return fisher

        # Sample random indices
        indices = np.random.choice(buffer_size, size=n_samples, replace=False)

        policy.train()
        for idx in indices:
            env_idx = idx % rollout_buffer.n_envs
            step_idx = idx // rollout_buffer.n_envs

            if step_idx >= rollout_buffer.buffer_size:
                continue

            obs = torch.as_tensor(
                rollout_buffer.observations[step_idx, env_idx],
                dtype=torch.float32,
                device=policy.device,
            ).unsqueeze(0)

            action = torch.as_tensor(
                rollout_buffer.actions[step_idx, env_idx],
                dtype=torch.long,
                device=policy.device,
            ).unsqueeze(0)

            # Forward pass to get action distribution
            policy.zero_grad()
            distribution = policy.get_distribution(obs)
            log_prob = distribution.log_prob(action)

            # Backward pass
            log_prob.backward()

            # Accumulate squared gradients
            for name, param in policy.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

        # Average over samples
        for name in fisher:
            fisher[name] /= n_samples

        return fisher

    def _apply_ewc_penalty(self) -> None:
        """
        Apply EWC penalty to the policy parameters via gradient modification.

        Adds lambda * F_i * (theta_i - theta*_i) to the gradient of each parameter.
        This is equivalent to adding lambda/2 * F_i * (theta_i - theta*_i)^2 to the loss.
        """
        if self._fisher_diag is None or self._anchor_params is None:
            return

        policy = self.model.policy
        ewc_loss = 0.0

        for name, param in policy.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self._fisher_diag or name not in self._anchor_params:
                continue

            fisher = self._fisher_diag[name]
            anchor = self._anchor_params[name]

            # Gradient of EWC penalty: lambda * F_i * (theta_i - theta*_i)
            penalty_grad = self.ewc_lambda * fisher * (param.data - anchor)

            if param.grad is not None:
                param.grad.data += penalty_grad
            else:
                param.grad = penalty_grad.clone()

            ewc_loss += (fisher * (param.data - anchor).pow(2)).sum().item()

        ewc_loss *= self.ewc_lambda / 2.0

        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            logger.info(f"EWC penalty at step {self.num_timesteps}: {ewc_loss:.4f}")

    @property
    def is_active(self) -> bool:
        """Whether EWC regularization is currently active."""
        return self._ewc_active

    @property
    def n_snapshots(self) -> int:
        """Number of phase snapshots taken."""
        return len(self._phase_snapshots)

    def get_summary(self) -> dict:
        """Get summary of EWC state."""
        return {
            'active': self._ewc_active,
            'lambda': self.ewc_lambda,
            'fisher_samples': self.fisher_samples,
            'n_snapshots': len(self._phase_snapshots),
            'snapshot_timesteps': self._phase_snapshots.copy(),
            'n_anchored_params': len(self._anchor_params) if self._anchor_params else 0,
        }
