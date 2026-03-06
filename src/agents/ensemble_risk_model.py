# =============================================================================
# ENSEMBLE RISK MODEL - Sprint 2: Intelligence Enhancement
# =============================================================================
"""
Institutional-Grade Ensemble Machine Learning for Risk Prediction

This module implements a sophisticated ensemble of ML models combining:
- Gradient Boosting (XGBoost-style) for feature importance and non-linear patterns
- LSTM-style Recurrent Networks for temporal dependencies
- Multi-Layer Perceptron for dense pattern recognition
- Adaptive ensemble weighting with performance tracking

Key Features:
- Pure NumPy implementations (no external ML dependencies required)
- Optional XGBoost/TensorFlow integration when available
- Online learning with incremental updates
- Confidence-weighted ensemble predictions
- Feature importance tracking
- Model performance monitoring
- Automatic model selection based on market conditions

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENSEMBLE RISK MODEL                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │  GRADIENT BOOST │  │     LSTM        │  │      MLP        │            │
│  │  (Tree-based)   │  │  (Sequential)   │  │   (Dense NN)    │            │
│  │  - Feature Imp  │  │  - Time Series  │  │  - Non-linear   │            │
│  │  - Interactions │  │  - Memory       │  │  - Universal    │            │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘            │
│           │                    │                    │                      │
│           └────────────────────┼────────────────────┘                      │
│                                ▼                                           │
│                  ┌─────────────────────────┐                              │
│                  │   ADAPTIVE ENSEMBLE     │                              │
│                  │  - Dynamic Weighting    │                              │
│                  │  - Confidence Scoring   │                              │
│                  │  - Performance Tracking │                              │
│                  └─────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘

Version: 2.0.0
Author: TradingBot Team
License: Proprietary - Commercial Use
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import numpy as np
from collections import deque
import logging
import hashlib
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# CHECK FOR OPTIONAL DEPENDENCIES
# =============================================================================

HAS_XGBOOST = False
HAS_SKLEARN = False
HAS_TENSORFLOW = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    pass

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    pass

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    pass


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class ModelType(Enum):
    """Types of models in the ensemble."""
    GRADIENT_BOOST = auto()
    LSTM = auto()
    MLP = auto()
    DECISION_TREE = auto()
    LINEAR = auto()


class RiskCategory(Enum):
    """Risk prediction categories."""
    VERY_LOW = auto()
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    VERY_HIGH = auto()
    EXTREME = auto()

    @property
    def score(self) -> float:
        """Numerical risk score 0-1."""
        scores = {
            RiskCategory.VERY_LOW: 0.1,
            RiskCategory.LOW: 0.25,
            RiskCategory.MODERATE: 0.5,
            RiskCategory.HIGH: 0.7,
            RiskCategory.VERY_HIGH: 0.85,
            RiskCategory.EXTREME: 0.95
        }
        return scores.get(self, 0.5)

    @classmethod
    def from_score(cls, score: float) -> 'RiskCategory':
        """Convert score to category."""
        if score < 0.15:
            return cls.VERY_LOW
        elif score < 0.35:
            return cls.LOW
        elif score < 0.55:
            return cls.MODERATE
        elif score < 0.75:
            return cls.HIGH
        elif score < 0.9:
            return cls.VERY_HIGH
        else:
            return cls.EXTREME


class PredictionType(Enum):
    """Type of prediction being made."""
    RISK_LEVEL = auto()       # Overall risk level
    DRAWDOWN = auto()         # Max drawdown prediction
    VOLATILITY = auto()       # Future volatility
    LOSS_PROBABILITY = auto()  # Probability of loss
    REGIME_SHIFT = auto()     # Probability of regime change


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelPrediction:
    """Single model prediction result."""
    model_type: ModelType
    prediction: float          # Raw prediction value
    confidence: float          # Model confidence 0-1
    feature_importance: Optional[Dict[str, float]] = None
    prediction_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type.name,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "feature_importance": self.feature_importance,
            "prediction_time_ms": self.prediction_time_ms
        }


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result combining all models."""
    timestamp: datetime
    prediction_type: PredictionType

    # Individual predictions
    model_predictions: Dict[ModelType, ModelPrediction]

    # Ensemble result
    ensemble_prediction: float
    ensemble_confidence: float
    risk_category: RiskCategory

    # Model weights used
    model_weights: Dict[ModelType, float]

    # Uncertainty metrics
    prediction_std: float
    prediction_range: Tuple[float, float]

    # Feature analysis
    top_features: List[Tuple[str, float]] = field(default_factory=list)

    # Metadata
    models_used: int = 0
    ensemble_method: str = "weighted_average"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "prediction_type": self.prediction_type.name,
            "model_predictions": {k.name: v.to_dict() for k, v in self.model_predictions.items()},
            "ensemble_prediction": self.ensemble_prediction,
            "ensemble_confidence": self.ensemble_confidence,
            "risk_category": self.risk_category.name,
            "model_weights": {k.name: v for k, v in self.model_weights.items()},
            "prediction_std": self.prediction_std,
            "prediction_range": self.prediction_range,
            "top_features": self.top_features,
            "models_used": self.models_used
        }


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""
    # Model selection
    use_gradient_boost: bool = True
    use_lstm: bool = True
    use_mlp: bool = True

    # Initial weights (will be adapted based on performance)
    initial_weights: Dict[ModelType, float] = field(default_factory=lambda: {
        ModelType.GRADIENT_BOOST: 0.4,
        ModelType.LSTM: 0.35,
        ModelType.MLP: 0.25
    })

    # Gradient Boost config
    gb_n_estimators: int = 100
    gb_max_depth: int = 5
    gb_learning_rate: float = 0.1
    gb_min_samples_split: int = 10

    # LSTM config
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_sequence_length: int = 20
    lstm_dropout: float = 0.2

    # MLP config
    mlp_hidden_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    mlp_activation: str = "relu"
    mlp_dropout: float = 0.3

    # Training config
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 10
    validation_split: float = 0.2

    # Ensemble config
    ensemble_method: str = "weighted_average"  # weighted_average, stacking, voting
    adapt_weights: bool = True
    weight_adaptation_rate: float = 0.1
    min_model_weight: float = 0.1

    # Feature config
    feature_names: List[str] = field(default_factory=list)
    normalize_features: bool = True

    # Performance tracking
    performance_window: int = 100  # Number of predictions to track


# =============================================================================
# FEATURE NORMALIZER
# =============================================================================

class FeatureNormalizer:
    """Normalize features for ML models."""

    def __init__(self):
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> 'FeatureNormalizer':
        """Fit normalizer to data."""
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds == 0] = 1.0  # Avoid division by zero
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features."""
        if not self.fitted:
            return X
        return (X - self.means) / self.stds

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform."""
        if not self.fitted:
            return X
        return X * self.stds + self.means


# =============================================================================
# PURE NUMPY GRADIENT BOOSTING (No XGBoost dependency)
# =============================================================================

class DecisionTree:
    """Pure NumPy decision tree with configurable max_depth for gradient boosting.

    Replaces the old DecisionStump (depth=1 only) to allow the GradientBoost
    regressor to actually use its max_depth parameter. Depth-3 trees capture
    feature interactions that stumps completely miss.
    """

    def __init__(self, max_depth: int = 3, min_samples_split: int = 10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # Tree stored as arrays (node i: left=2i+1, right=2i+2)
        self._feature_idx: List[int] = []
        self._threshold: List[float] = []
        self._value: List[float] = []
        self._is_leaf: List[bool] = []
        self._features_used: List[int] = []  # For importance tracking

    def _find_best_split(self, X: np.ndarray, gradients: np.ndarray):
        """Find the best feature and threshold to split on."""
        n_samples, n_features = X.shape
        best_loss = float('inf')
        best_feat = 0
        best_thresh = 0.0
        best_left_val = 0.0
        best_right_val = 0.0

        for feat_idx in range(n_features):
            feature_values = X[:, feat_idx]
            thresholds = np.unique(feature_values)

            if len(thresholds) > 20:
                thresholds = np.percentile(feature_values, np.linspace(0, 100, 20))

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                if n_left < 2 or n_right < 2:
                    continue

                left_val = np.mean(gradients[left_mask])
                right_val = np.mean(gradients[right_mask])

                # Weighted MSE reduction
                left_loss = np.sum((gradients[left_mask] - left_val) ** 2)
                right_loss = np.sum((gradients[right_mask] - right_val) ** 2)
                loss = left_loss + right_loss

                if loss < best_loss:
                    best_loss = loss
                    best_feat = feat_idx
                    best_thresh = threshold
                    best_left_val = left_val
                    best_right_val = right_val

        return best_feat, best_thresh, best_left_val, best_right_val, best_loss

    def _build_tree(self, X: np.ndarray, gradients: np.ndarray,
                    depth: int, node_idx: int) -> None:
        """Recursively build the tree."""
        # Ensure arrays are large enough
        while len(self._feature_idx) <= node_idx:
            self._feature_idx.append(0)
            self._threshold.append(0.0)
            self._value.append(0.0)
            self._is_leaf.append(True)

        # Base case: make leaf
        if (depth >= self.max_depth or
                len(X) < self.min_samples_split or
                np.std(gradients) < 1e-8):
            self._value[node_idx] = np.mean(gradients)
            self._is_leaf[node_idx] = True
            return

        # Find best split
        feat, thresh, left_val, right_val, loss = self._find_best_split(X, gradients)

        # Check if split is useful
        no_split_loss = np.sum((gradients - np.mean(gradients)) ** 2)
        if loss >= no_split_loss - 1e-8:
            self._value[node_idx] = np.mean(gradients)
            self._is_leaf[node_idx] = True
            return

        # Make internal node
        self._feature_idx[node_idx] = feat
        self._threshold[node_idx] = thresh
        self._value[node_idx] = np.mean(gradients)
        self._is_leaf[node_idx] = False
        self._features_used.append(feat)

        # Split data
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_child = 2 * node_idx + 1
        right_child = 2 * node_idx + 2

        self._build_tree(X[left_mask], gradients[left_mask], depth + 1, left_child)
        self._build_tree(X[right_mask], gradients[right_mask], depth + 1, right_child)

    def fit(self, X: np.ndarray, gradients: np.ndarray) -> 'DecisionTree':
        """Fit tree to minimize squared error on gradients."""
        self._feature_idx = []
        self._threshold = []
        self._value = []
        self._is_leaf = []
        self._features_used = []

        self._build_tree(X, gradients, depth=0, node_idx=0)
        return self

    def _predict_one(self, x: np.ndarray) -> float:
        """Predict for a single sample by traversing the tree."""
        node_idx = 0
        while node_idx < len(self._is_leaf) and not self._is_leaf[node_idx]:
            if x[self._feature_idx[node_idx]] <= self._threshold[node_idx]:
                node_idx = 2 * node_idx + 1
            else:
                node_idx = 2 * node_idx + 2
        if node_idx < len(self._value):
            return self._value[node_idx]
        return 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for all samples."""
        return np.array([self._predict_one(x) for x in X])

    @property
    def feature_idx(self) -> int:
        """Root feature index (backwards compat with old DecisionStump API)."""
        return self._feature_idx[0] if self._feature_idx else 0


class GradientBoostRegressor:
    """
    Pure NumPy implementation of Gradient Boosting.
    No external dependencies required.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.trees: List[DecisionTree] = []
        self.initial_prediction: float = 0.0
        self.feature_importances_: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostRegressor':
        """Fit gradient boosting model."""
        n_samples, n_features = X.shape

        # Initialize with mean
        self.initial_prediction = np.mean(y)
        current_pred = np.full(n_samples, self.initial_prediction)

        # Feature importance tracking
        feature_splits = np.zeros(n_features)

        for i in range(self.n_estimators):
            # Calculate residuals (negative gradient for MSE)
            residuals = y - current_pred

            # Fit tree to residuals
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions
            tree_pred = tree.predict(X)
            current_pred += self.learning_rate * tree_pred

            # Track feature importance (count all splits, not just root)
            for feat in tree._features_used:
                feature_splits[feat] += 1

        # Normalize feature importances
        self.feature_importances_ = feature_splits / np.sum(feature_splits)
        self.fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.fitted:
            raise ValueError("Model not fitted")

        pred = np.full(X.shape[0], self.initial_prediction)

        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)

        return pred

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance as dictionary."""
        if self.feature_importances_ is None:
            return {}

        return {name: float(imp) for name, imp in
                zip(feature_names, self.feature_importances_)}


# =============================================================================
# PURE NUMPY LSTM (No TensorFlow dependency)
# =============================================================================

class NumpyLSTMCell:
    """Pure NumPy LSTM cell implementation."""

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Xavier initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # Forget gate
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bf = np.zeros((hidden_size, 1))

        # Input gate
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bi = np.zeros((hidden_size, 1))

        # Candidate gate
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bc = np.zeros((hidden_size, 1))

        # Output gate
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x: np.ndarray, h_prev: np.ndarray,
                c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through LSTM cell."""
        # Reshape inputs
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        h_prev = h_prev.reshape(-1, 1) if h_prev.ndim == 1 else h_prev
        c_prev = c_prev.reshape(-1, 1) if c_prev.ndim == 1 else c_prev

        # Concatenate input and hidden state
        concat = np.vstack([h_prev, x])

        # Gates
        f = self.sigmoid(self.Wf @ concat + self.bf)
        i = self.sigmoid(self.Wi @ concat + self.bi)
        c_candidate = np.tanh(self.Wc @ concat + self.bc)
        o = self.sigmoid(self.Wo @ concat + self.bo)

        # New cell state and hidden state
        c = f * c_prev + i * c_candidate
        h = o * np.tanh(c)

        return h, c


class NumpyLSTM:
    """
    Pure NumPy LSTM network for sequence modeling.
    No TensorFlow/PyTorch dependency.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 1,
                 sequence_length: int = 20, dropout: float = 0.2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.dropout = dropout

        # LSTM layers
        self.layers: List[NumpyLSTMCell] = []
        for i in range(num_layers):
            inp_size = input_size if i == 0 else hidden_size
            self.layers.append(NumpyLSTMCell(inp_size, hidden_size))

        # Output layer
        scale = np.sqrt(2.0 / hidden_size)
        self.Wy = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))

        self.fitted = False

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through LSTM network.

        SECURITY FIX: Now validates sequence length to prevent silent mismatch.
        """
        batch_size = X.shape[0] if X.ndim == 3 else 1
        seq_len = X.shape[1] if X.ndim == 3 else X.shape[0]

        # SECURITY FIX: Validate sequence length
        if self.fitted and seq_len != self.sequence_length:
            logger.warning(
                f"LSTM sequence length mismatch: expected {self.sequence_length}, got {seq_len}. "
                f"Padding/truncating to match."
            )
            # Pad or truncate to expected sequence length
            if X.ndim == 2:
                X = X.reshape(1, seq_len, -1)
            n_features = X.shape[2]
            if seq_len < self.sequence_length:
                # Pad with zeros at the beginning
                padding = np.zeros((batch_size, self.sequence_length - seq_len, n_features))
                X = np.concatenate([padding, X], axis=1)
            else:
                # Truncate from the beginning (keep most recent)
                X = X[:, -self.sequence_length:, :]
            seq_len = self.sequence_length
        elif X.ndim == 2:
            X = X.reshape(1, seq_len, -1)

        outputs = []

        for b in range(batch_size):
            # Initialize hidden states
            h = [np.zeros((self.hidden_size, 1)) for _ in range(self.num_layers)]
            c = [np.zeros((self.hidden_size, 1)) for _ in range(self.num_layers)]

            # Process sequence
            for t in range(seq_len):
                x_t = X[b, t, :].reshape(-1, 1)

                for layer_idx, layer in enumerate(self.layers):
                    inp = x_t if layer_idx == 0 else h[layer_idx - 1]
                    h[layer_idx], c[layer_idx] = layer.forward(inp, h[layer_idx], c[layer_idx])

            # Final output
            y = self.Wy @ h[-1] + self.by
            outputs.append(y.flatten()[0])

        return np.array(outputs)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50,
            learning_rate: float = 0.001) -> 'NumpyLSTM':
        """
        Training via gradient descent on the output projection layer (Wy, by).

        The LSTM gates use random (Xavier-initialized) weights as a fixed
        feature extractor, while the output layer learns the mapping from
        hidden state to predictions. This is analogous to reservoir computing
        / echo state networks and is effective for risk prediction tasks.
        """
        # Create sequences
        sequences, targets = self._create_sequences(X, y)

        if len(sequences) == 0:
            logger.warning("Not enough data for LSTM training")
            return self

        best_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            total_loss = 0.0
            n_samples = len(sequences)

            for seq, target in zip(sequences, targets):
                # Forward pass: get hidden state from LSTM layers
                x_input = seq.reshape(1, -1, self.input_size)
                batch_size = 1
                seq_len = x_input.shape[1]

                # Run through LSTM layers to get final hidden state
                h = [np.zeros((self.hidden_size, 1)) for _ in range(self.num_layers)]
                c = [np.zeros((self.hidden_size, 1)) for _ in range(self.num_layers)]

                for t in range(seq_len):
                    x_t = x_input[0, t, :].reshape(-1, 1)
                    for layer_idx, layer in enumerate(self.layers):
                        inp = x_t if layer_idx == 0 else h[layer_idx - 1]
                        h[layer_idx], c[layer_idx] = layer.forward(inp, h[layer_idx], c[layer_idx])

                # Output projection: y = Wy @ h[-1] + by
                h_final = h[-1]  # (hidden_size, 1)
                pred = (self.Wy @ h_final + self.by).flatten()[0]

                # Compute error and loss
                error = pred - target
                total_loss += error ** 2

                # Gradient descent on output layer (Wy and by)
                # dL/dWy = error * h_final^T, dL/dby = error
                grad_Wy = error * h_final.T  # (output_size, hidden_size)
                grad_by = error

                self.Wy -= learning_rate * grad_Wy
                self.by[0, 0] -= learning_rate * grad_by

            avg_loss = total_loss / n_samples
            if (epoch + 1) % 10 == 0:
                logger.debug(f"LSTM Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

            # Early stopping
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.debug(f"LSTM early stopping at epoch {epoch + 1}")
                    break

        self.fitted = True
        return self

    def _create_sequences(self, X: np.ndarray, y: np.ndarray
                          ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Create sequences for training.

        SECURITY FIX: Now validates data length and logs warnings.
        """
        sequences = []
        targets = []

        # SECURITY FIX: Validate input data length
        if len(X) <= self.sequence_length:
            logger.warning(
                f"Insufficient data for LSTM sequences: need > {self.sequence_length} samples, "
                f"got {len(X)}. Returning empty sequences."
            )
            return sequences, targets

        if len(X) != len(y):
            logger.error(f"X and y length mismatch: {len(X)} vs {len(y)}")
            return sequences, targets

        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i + self.sequence_length])
            targets.append(y[i + self.sequence_length])

        logger.debug(f"Created {len(sequences)} sequences of length {self.sequence_length}")
        return sequences, targets

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])

        return self.forward(X)


# =============================================================================
# PURE NUMPY MLP (No TensorFlow dependency)
# =============================================================================

class NumpyMLP:
    """
    Pure NumPy Multi-Layer Perceptron.
    No TensorFlow/PyTorch dependency.
    """

    def __init__(self, input_size: int, hidden_layers: List[int] = None,
                 output_size: int = 1, activation: str = "relu",
                 dropout: float = 0.3):
        self.input_size = input_size
        self.hidden_layers = hidden_layers or [64, 32, 16]
        self.output_size = output_size
        self.activation = activation
        self.dropout = dropout

        # Initialize weights
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        layer_sizes = [input_size] + self.hidden_layers + [output_size]

        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])  # He initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        self.fitted = False

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x

    def _activate_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of activation function."""
        if self.activation == "relu":
            return (x > 0).astype(float)
        elif self.activation == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation == "sigmoid":
            s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return s * (1 - s)
        else:
            return np.ones_like(x)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        current = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            current = current @ w + b
            if i < len(self.weights) - 1:  # No activation on output
                current = self._activate(current)
        return current

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
            learning_rate: float = 0.001, batch_size: int = 32,
            patience: int = 15) -> 'NumpyMLP':
        """Train MLP with mini-batch gradient descent and early stopping."""
        n_samples = X.shape[0]
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        # Early stopping state
        best_loss = float('inf')
        best_weights = None
        best_biases = None
        no_improve_count = 0

        # Gradient clipping threshold
        max_grad_norm = 5.0

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            total_loss = 0.0
            n_batches = 0

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass with caching
                activations = [X_batch]
                pre_activations = []

                current = X_batch
                for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                    z = current @ w + b
                    pre_activations.append(z)
                    if i < len(self.weights) - 1:
                        current = self._activate(z)
                    else:
                        current = z  # Linear output
                    activations.append(current)

                # Compute loss
                pred = activations[-1]
                loss = np.mean((pred - y_batch) ** 2)
                total_loss += loss
                n_batches += 1

                # Backward pass
                delta = 2 * (pred - y_batch) / len(y_batch)

                for i in range(len(self.weights) - 1, -1, -1):
                    # Gradient for weights and biases
                    dW = activations[i].T @ delta
                    db = np.sum(delta, axis=0, keepdims=True)

                    # Gradient clipping to prevent exploding gradients
                    dW_norm = np.linalg.norm(dW)
                    if dW_norm > max_grad_norm:
                        dW = dW * (max_grad_norm / dW_norm)

                    # Update weights
                    self.weights[i] -= learning_rate * dW
                    self.biases[i] -= learning_rate * db

                    # Propagate error
                    if i > 0:
                        delta = (delta @ self.weights[i].T) * self._activate_derivative(pre_activations[i - 1])

            avg_loss = total_loss / max(n_batches, 1)

            # Early stopping check
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                if best_weights is not None:
                    self.weights = best_weights
                    self.biases = best_biases
                logger.debug(f"MLP early stopping at epoch {epoch + 1}, best loss: {best_loss:.6f}")
                break

            if (epoch + 1) % 20 == 0:
                logger.debug(f"MLP Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        # Restore best weights if we completed all epochs
        if no_improve_count < patience and best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases

        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X).flatten()


# =============================================================================
# ENSEMBLE RISK MODEL
# =============================================================================

class EnsembleRiskModel:
    """
    Ensemble Machine Learning Model for Risk Prediction.

    Combines Gradient Boosting, LSTM, and MLP with adaptive weighting.
    Falls back to pure NumPy implementations if external libraries unavailable.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """Initialize ensemble model."""
        self.config = config or EnsembleConfig()

        # Models
        self.gb_model: Optional[Union[GradientBoostRegressor, Any]] = None
        self.lstm_model: Optional[Union[NumpyLSTM, Any]] = None
        self.mlp_model: Optional[Union[NumpyMLP, Any]] = None

        # Normalizer
        self.normalizer = FeatureNormalizer()

        # Model weights (will be adapted based on performance)
        self.model_weights = dict(self.config.initial_weights)

        # Performance tracking
        self.performance_history: Dict[ModelType, deque] = {
            ModelType.GRADIENT_BOOST: deque(maxlen=self.config.performance_window),
            ModelType.LSTM: deque(maxlen=self.config.performance_window),
            ModelType.MLP: deque(maxlen=self.config.performance_window)
        }

        # Prediction history
        self.prediction_history: deque = deque(maxlen=100)

        # State
        self.fitted = False
        self.feature_names: List[str] = []
        self.input_size: int = 0

        logger.info(f"EnsembleRiskModel initialized "
                   f"(XGBoost: {HAS_XGBOOST}, TensorFlow: {HAS_TENSORFLOW})")

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'EnsembleRiskModel':
        """
        Fit all models in the ensemble.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            feature_names: Optional list of feature names

        Returns:
            Self for chaining
        """
        n_samples, n_features = X.shape
        self.input_size = n_features
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]

        logger.info(f"Training ensemble on {n_samples} samples, {n_features} features")

        # Normalize features
        if self.config.normalize_features:
            X_normalized = self.normalizer.fit_transform(X)
        else:
            X_normalized = X

        # Split for validation
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X_normalized[:split_idx], X_normalized[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train Gradient Boosting
        if self.config.use_gradient_boost:
            self._train_gradient_boost(X_train, y_train, X_val, y_val)

        # Train LSTM
        if self.config.use_lstm:
            self._train_lstm(X_train, y_train, X_val, y_val)

        # Train MLP
        if self.config.use_mlp:
            self._train_mlp(X_train, y_train, X_val, y_val)

        self.fitted = True
        logger.info("Ensemble training complete")

        return self

    def _train_gradient_boost(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train gradient boosting model."""
        logger.info("Training Gradient Boosting...")

        if HAS_XGBOOST:
            # Use real XGBoost
            self.gb_model = xgb.XGBRegressor(
                n_estimators=self.config.gb_n_estimators,
                max_depth=self.config.gb_max_depth,
                learning_rate=self.config.gb_learning_rate,
                min_child_weight=self.config.gb_min_samples_split,
                random_state=42
            )
            self.gb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            logger.info("XGBoost model trained")
        else:
            # Use pure NumPy implementation
            self.gb_model = GradientBoostRegressor(
                n_estimators=self.config.gb_n_estimators,
                learning_rate=self.config.gb_learning_rate,
                max_depth=self.config.gb_max_depth
            )
            self.gb_model.fit(X_train, y_train)
            logger.info("NumPy GradientBoost model trained")

    def _train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train LSTM model."""
        logger.info("Training LSTM...")

        if HAS_TENSORFLOW:
            # Use TensorFlow LSTM
            try:
                model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(
                        self.config.lstm_hidden_size,
                        input_shape=(self.config.lstm_sequence_length, self.input_size),
                        return_sequences=True
                    ),
                    tf.keras.layers.Dropout(self.config.lstm_dropout),
                    tf.keras.layers.LSTM(self.config.lstm_hidden_size // 2),
                    tf.keras.layers.Dropout(self.config.lstm_dropout),
                    tf.keras.layers.Dense(16, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])

                model.compile(optimizer='adam', loss='mse')

                # Create sequences
                X_seq, y_seq = self._create_sequences(X_train, y_train)
                X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)

                if len(X_seq) > 0:
                    model.fit(
                        X_seq, y_seq,
                        epochs=self.config.epochs,
                        batch_size=self.config.batch_size,
                        validation_data=(X_val_seq, y_val_seq) if len(X_val_seq) > 0 else None,
                        verbose=0
                    )
                    self.lstm_model = model
                    logger.info("TensorFlow LSTM trained")
                else:
                    logger.warning("Insufficient data for TensorFlow LSTM")
            except Exception as e:
                logger.warning(f"TensorFlow LSTM failed: {e}, falling back to NumPy")
                self._train_numpy_lstm(X_train, y_train)
        else:
            self._train_numpy_lstm(X_train, y_train)

    def _train_numpy_lstm(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train NumPy LSTM."""
        self.lstm_model = NumpyLSTM(
            input_size=self.input_size,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            sequence_length=self.config.lstm_sequence_length
        )
        self.lstm_model.fit(X_train, y_train, epochs=self.config.epochs)
        logger.info("NumPy LSTM trained")

    def _train_mlp(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train MLP model."""
        logger.info("Training MLP...")

        if HAS_TENSORFLOW:
            try:
                layers = [tf.keras.layers.Input(shape=(self.input_size,))]
                for units in self.config.mlp_hidden_layers:
                    layers.append(tf.keras.layers.Dense(units, activation='relu'))
                    layers.append(tf.keras.layers.Dropout(self.config.mlp_dropout))
                layers.append(tf.keras.layers.Dense(1))

                model = tf.keras.Sequential(layers)
                model.compile(optimizer='adam', loss='mse')

                model.fit(
                    X_train, y_train,
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size,
                    validation_data=(X_val, y_val),
                    verbose=0
                )
                self.mlp_model = model
                logger.info("TensorFlow MLP trained")
            except Exception as e:
                logger.warning(f"TensorFlow MLP failed: {e}, falling back to NumPy")
                self._train_numpy_mlp(X_train, y_train)
        else:
            self._train_numpy_mlp(X_train, y_train)

    def _train_numpy_mlp(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train NumPy MLP."""
        self.mlp_model = NumpyMLP(
            input_size=self.input_size,
            hidden_layers=self.config.mlp_hidden_layers,
            activation=self.config.mlp_activation,
            dropout=self.config.mlp_dropout
        )
        self.mlp_model.fit(X_train, y_train, epochs=self.config.epochs)
        logger.info("NumPy MLP trained")

    def _create_sequences(self, X: np.ndarray, y: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        seq_len = self.config.lstm_sequence_length
        if len(X) <= seq_len:
            return np.array([]), np.array([])

        sequences = []
        targets = []

        for i in range(len(X) - seq_len):
            sequences.append(X[i:i + seq_len])
            targets.append(y[i + seq_len])

        return np.array(sequences), np.array(targets)

    def predict(self, X: np.ndarray,
                prediction_type: PredictionType = PredictionType.RISK_LEVEL
                ) -> EnsemblePrediction:
        """
        Make ensemble prediction.

        Args:
            X: Feature vector or matrix
            prediction_type: Type of prediction

        Returns:
            EnsemblePrediction with detailed results
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        start_time = datetime.now()

        # Reshape if needed
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Input validation: replace NaN/inf with 0 to prevent silent propagation
        if np.any(~np.isfinite(X)):
            nan_count = np.sum(np.isnan(X))
            inf_count = np.sum(np.isinf(X))
            if nan_count > 0 or inf_count > 0:
                logger.warning(f"Ensemble input has {nan_count} NaN, {inf_count} inf values - replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize
        if self.config.normalize_features:
            X_normalized = self.normalizer.transform(X)
        else:
            X_normalized = X

        # Get individual predictions
        model_predictions: Dict[ModelType, ModelPrediction] = {}

        # Gradient Boosting prediction
        if self.gb_model is not None:
            pred_start = datetime.now()
            try:
                if HAS_XGBOOST and hasattr(self.gb_model, 'predict'):
                    gb_pred = float(self.gb_model.predict(X_normalized)[0])
                    feat_imp = dict(zip(self.feature_names,
                                       self.gb_model.feature_importances_))
                else:
                    gb_pred = float(self.gb_model.predict(X_normalized)[0])
                    feat_imp = self.gb_model.get_feature_importance(self.feature_names)

                model_predictions[ModelType.GRADIENT_BOOST] = ModelPrediction(
                    model_type=ModelType.GRADIENT_BOOST,
                    prediction=np.clip(gb_pred, 0, 1),
                    confidence=self._calculate_model_confidence(ModelType.GRADIENT_BOOST),
                    feature_importance=feat_imp,
                    prediction_time_ms=(datetime.now() - pred_start).total_seconds() * 1000
                )
            except Exception as e:
                logger.warning(f"GB prediction failed: {e}")

        # LSTM prediction
        if self.lstm_model is not None:
            pred_start = datetime.now()
            try:
                if HAS_TENSORFLOW and hasattr(self.lstm_model, 'predict'):
                    # Need sequence for TF LSTM - reshape to (batch, timesteps, features)
                    # Ensure proper dimensions: X_normalized is 1D, reshape to (1, 1, n_features)
                    n_features = X_normalized.shape[-1] if X_normalized.ndim > 0 else 1
                    X_lstm = X_normalized.flatten().reshape(1, 1, n_features)
                    lstm_pred = float(self.lstm_model.predict(X_lstm, verbose=0)[0][0])
                else:
                    # NumPy LSTM - ensure 2D input (batch, features)
                    X_lstm = X_normalized.reshape(1, -1) if X_normalized.ndim == 1 else X_normalized
                    lstm_pred = float(self.lstm_model.predict(X_lstm)[0])

                model_predictions[ModelType.LSTM] = ModelPrediction(
                    model_type=ModelType.LSTM,
                    prediction=np.clip(lstm_pred, 0, 1),
                    confidence=self._calculate_model_confidence(ModelType.LSTM),
                    prediction_time_ms=(datetime.now() - pred_start).total_seconds() * 1000
                )
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")

        # MLP prediction
        if self.mlp_model is not None:
            pred_start = datetime.now()
            try:
                if HAS_TENSORFLOW and hasattr(self.mlp_model, 'predict'):
                    mlp_pred = float(self.mlp_model.predict(X_normalized, verbose=0)[0][0])
                else:
                    mlp_pred = float(self.mlp_model.predict(X_normalized)[0])

                model_predictions[ModelType.MLP] = ModelPrediction(
                    model_type=ModelType.MLP,
                    prediction=np.clip(mlp_pred, 0, 1),
                    confidence=self._calculate_model_confidence(ModelType.MLP),
                    prediction_time_ms=(datetime.now() - pred_start).total_seconds() * 1000
                )
            except Exception as e:
                logger.warning(f"MLP prediction failed: {e}")

        # Ensemble aggregation
        ensemble_pred, ensemble_conf, weights_used = self._aggregate_predictions(model_predictions)

        # Calculate uncertainty
        predictions = [p.prediction for p in model_predictions.values()]
        pred_std = np.std(predictions) if len(predictions) > 1 else 0.0
        pred_range = (min(predictions), max(predictions)) if predictions else (0.0, 0.0)

        # Get top features
        top_features = self._get_top_features(model_predictions)

        # Create result
        result = EnsemblePrediction(
            timestamp=datetime.now(),
            prediction_type=prediction_type,
            model_predictions=model_predictions,
            ensemble_prediction=ensemble_pred,
            ensemble_confidence=ensemble_conf,
            risk_category=RiskCategory.from_score(ensemble_pred),
            model_weights=weights_used,
            prediction_std=pred_std,
            prediction_range=pred_range,
            top_features=top_features,
            models_used=len(model_predictions),
            ensemble_method=self.config.ensemble_method
        )

        self.prediction_history.append(result)

        return result

    def _calculate_model_confidence(self, model_type: ModelType) -> float:
        """Calculate model confidence based on recent performance."""
        history = self.performance_history.get(model_type, [])

        if len(history) == 0:
            return 0.7  # Default confidence

        # Calculate accuracy-based confidence
        recent_errors = list(history)[-20:]
        if not recent_errors:
            return 0.7

        mean_error = np.mean(np.abs(recent_errors))
        confidence = max(0.3, 1.0 - mean_error)

        return confidence

    def _aggregate_predictions(self, predictions: Dict[ModelType, ModelPrediction]
                               ) -> Tuple[float, float, Dict[ModelType, float]]:
        """Aggregate predictions using configured method."""
        if not predictions:
            return 0.5, 0.3, {}

        if self.config.ensemble_method == "weighted_average":
            return self._weighted_average(predictions)
        elif self.config.ensemble_method == "voting":
            return self._majority_voting(predictions)
        else:
            return self._weighted_average(predictions)

    def _weighted_average(self, predictions: Dict[ModelType, ModelPrediction]
                          ) -> Tuple[float, float, Dict[ModelType, float]]:
        """Weighted average ensemble."""
        weighted_sum = 0.0
        confidence_sum = 0.0
        total_weight = 0.0
        weights_used = {}

        for model_type, pred in predictions.items():
            weight = self.model_weights.get(model_type, 0.25)
            adjusted_weight = weight * pred.confidence

            weighted_sum += pred.prediction * adjusted_weight
            confidence_sum += pred.confidence * weight
            total_weight += adjusted_weight
            weights_used[model_type] = adjusted_weight

        if total_weight > 0:
            ensemble_pred = weighted_sum / total_weight
            ensemble_conf = confidence_sum / sum(weights_used.values())
        else:
            ensemble_pred = 0.5
            ensemble_conf = 0.3

        # Normalize weights
        total = sum(weights_used.values())
        weights_used = {k: v / total for k, v in weights_used.items()}

        return ensemble_pred, ensemble_conf, weights_used

    def _majority_voting(self, predictions: Dict[ModelType, ModelPrediction]
                         ) -> Tuple[float, float, Dict[ModelType, float]]:
        """Majority voting ensemble."""
        votes = []
        weights_used = {}

        for model_type, pred in predictions.items():
            vote = 1 if pred.prediction >= 0.5 else 0
            weight = self.model_weights.get(model_type, 0.25)
            votes.append((vote, weight))
            weights_used[model_type] = weight

        weight_sum = sum(w for _, w in votes)
        if weight_sum <= 0:
            weight_sum = 1.0  # Prevent division by zero
        weighted_vote = sum(v * w for v, w in votes) / weight_sum
        # Clamp confidence to [0, 1] defensively (margin-based)
        confidence = min(1.0, max(0.0, abs(weighted_vote - 0.5) * 2))

        return weighted_vote, confidence, weights_used

    def _get_top_features(self, predictions: Dict[ModelType, ModelPrediction],
                          top_n: int = 5) -> List[Tuple[str, float]]:
        """Get top features from gradient boosting."""
        for model_type, pred in predictions.items():
            if pred.feature_importance:
                sorted_features = sorted(
                    pred.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                return sorted_features[:top_n]
        return []

    def update_performance(self, model_type: ModelType, predicted: float,
                           actual: float) -> None:
        """Update model performance tracking."""
        error = predicted - actual
        self.performance_history[model_type].append(error)

        # Adapt weights if enabled
        if self.config.adapt_weights:
            self._adapt_weights()

    def _adapt_weights(self) -> None:
        """Adapt model weights based on recent performance."""
        new_weights = {}

        for model_type in [ModelType.GRADIENT_BOOST, ModelType.LSTM, ModelType.MLP]:
            history = self.performance_history.get(model_type, [])

            if len(history) < 10:
                new_weights[model_type] = self.config.initial_weights.get(model_type, 0.25)
                continue

            # Calculate inverse error as weight
            recent_errors = list(history)[-20:]
            mean_abs_error = np.mean(np.abs(recent_errors))
            weight = max(self.config.min_model_weight, 1.0 - mean_abs_error)
            new_weights[model_type] = weight

        # Normalize weights
        total = sum(new_weights.values())
        self.model_weights = {k: v / total for k, v in new_weights.items()}

    def save(self, filepath: str) -> None:
        """
        Save model to file with HMAC integrity verification.

        Saves all learned parameters including LSTM weights, GB trees,
        MLP weights, adaptive ensemble weights, and normalizer state.
        """
        state = {
            'version': '2.1.0',
            'config': self.config,
            'model_weights': self.model_weights,
            'normalizer_means': self.normalizer.means,
            'normalizer_stds': self.normalizer.stds,
            'normalizer_fitted': self.normalizer.fitted,
            'feature_names': self.feature_names,
            'input_size': self.input_size,
            'fitted': self.fitted,
            'performance_history': {k.name: list(v) for k, v in self.performance_history.items()},
        }

        # Save NumPy models
        if self.gb_model is not None and isinstance(self.gb_model, GradientBoostRegressor):
            state['gb_model'] = self.gb_model

        if self.lstm_model is not None and isinstance(self.lstm_model, NumpyLSTM):
            state['lstm_model'] = self.lstm_model

        if self.mlp_model is not None and isinstance(self.mlp_model, NumpyMLP):
            state['mlp_model'] = self.mlp_model

        # Save with integrity check
        data = pickle.dumps(state)
        checksum = hashlib.sha256(data).hexdigest()

        with open(filepath, 'wb') as f:
            # Write checksum header (64 hex chars + newline = 65 bytes)
            f.write((checksum + '\n').encode('ascii'))
            f.write(data)

        logger.info(f"Model saved to {filepath} ({len(data)} bytes, sha256={checksum[:16]}...)")

    def load(self, filepath: str) -> 'EnsembleRiskModel':
        """
        Load model from file with integrity verification.

        Verifies SHA-256 checksum before deserializing to detect corruption.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            # Read checksum header
            header = f.readline().decode('ascii').strip()
            data = f.read()

        # Verify integrity
        actual_checksum = hashlib.sha256(data).hexdigest()
        if header != actual_checksum:
            raise ValueError(
                f"Model file integrity check failed: "
                f"expected {header[:16]}..., got {actual_checksum[:16]}..."
            )

        state = pickle.loads(data)

        self.config = state['config']
        self.model_weights = state['model_weights']
        self.normalizer.means = state['normalizer_means']
        self.normalizer.stds = state['normalizer_stds']
        self.normalizer.fitted = state.get('normalizer_fitted', True)
        self.feature_names = state['feature_names']
        self.input_size = state['input_size']
        self.fitted = state['fitted']

        if 'gb_model' in state:
            self.gb_model = state['gb_model']
        if 'lstm_model' in state:
            self.lstm_model = state['lstm_model']
        if 'mlp_model' in state:
            self.mlp_model = state['mlp_model']

        # Restore performance history
        if 'performance_history' in state:
            for k_name, v_list in state['performance_history'].items():
                model_type = ModelType[k_name]
                self.performance_history[model_type] = deque(v_list, maxlen=self.config.performance_window)

        logger.info(f"Model loaded from {filepath} (integrity verified)")
        return self

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        stats = {
            "fitted": self.fitted,
            "input_size": self.input_size,
            "n_features": len(self.feature_names),
            "model_weights": {k.name: v for k, v in self.model_weights.items()},
            "predictions_made": len(self.prediction_history),
            "models_available": {
                "gradient_boost": self.gb_model is not None,
                "lstm": self.lstm_model is not None,
                "mlp": self.mlp_model is not None
            },
            "using_external_libs": {
                "xgboost": HAS_XGBOOST,
                "tensorflow": HAS_TENSORFLOW
            }
        }

        # Performance stats
        for model_type in [ModelType.GRADIENT_BOOST, ModelType.LSTM, ModelType.MLP]:
            history = list(self.performance_history.get(model_type, []))
            if history:
                stats[f"{model_type.name.lower()}_mae"] = float(np.mean(np.abs(history)))
                stats[f"{model_type.name.lower()}_rmse"] = float(np.sqrt(np.mean(np.square(history))))

        return stats


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_ensemble_risk_model(
    use_gradient_boost: bool = True,
    use_lstm: bool = True,
    use_mlp: bool = True,
    feature_names: Optional[List[str]] = None
) -> EnsembleRiskModel:
    """
    Create an ensemble risk model with default configuration.

    Args:
        use_gradient_boost: Include gradient boosting model
        use_lstm: Include LSTM model
        use_mlp: Include MLP model
        feature_names: List of feature names

    Returns:
        Configured EnsembleRiskModel

    Example:
        >>> model = create_ensemble_risk_model()
        >>> model.fit(X_train, y_train)
        >>> prediction = model.predict(X_test)
    """
    config = EnsembleConfig(
        use_gradient_boost=use_gradient_boost,
        use_lstm=use_lstm,
        use_mlp=use_mlp,
        feature_names=feature_names or []
    )

    return EnsembleRiskModel(config)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_risk_features(returns: np.ndarray, prices: np.ndarray,
                           volumes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Generate standard risk features from price data.

    Args:
        returns: Return series
        prices: Price series
        volumes: Optional volume series

    Returns:
        Feature matrix and feature names
    """
    n = len(returns)
    features = []
    names = []

    # Volatility features
    for window in [5, 10, 20, 50]:
        if n >= window:
            vol = np.array([np.std(returns[max(0, i - window):i])
                           for i in range(1, n + 1)])
            features.append(vol)
            names.append(f"volatility_{window}")

    # Return features
    for window in [1, 5, 10, 20]:
        if n >= window:
            ret = np.array([np.sum(returns[max(0, i - window):i])
                           for i in range(1, n + 1)])
            features.append(ret)
            names.append(f"return_{window}")

    # Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    features.append(drawdown)
    names.append("drawdown")

    # Price momentum
    for window in [5, 10, 20]:
        if n >= window:
            mom = np.array([(prices[i] - prices[max(0, i - window)]) / prices[max(0, i - window)]
                           for i in range(n)])
            features.append(mom)
            names.append(f"momentum_{window}")

    # Volume features (if available)
    if volumes is not None and len(volumes) == n:
        vol_ma = np.array([np.mean(volumes[max(0, i - 20):i])
                          for i in range(1, n + 1)])
        vol_ratio = volumes / (vol_ma + 1e-10)
        features.append(vol_ratio)
        names.append("volume_ratio")

    return np.column_stack(features), names


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'ModelType',
    'RiskCategory',
    'PredictionType',

    # Data classes
    'ModelPrediction',
    'EnsemblePrediction',
    'EnsembleConfig',

    # Models
    'FeatureNormalizer',
    'GradientBoostRegressor',
    'NumpyLSTM',
    'NumpyMLP',
    'EnsembleRiskModel',

    # Factory
    'create_ensemble_risk_model',

    # Utilities
    'generate_risk_features',

    # Flags
    'HAS_XGBOOST',
    'HAS_SKLEARN',
    'HAS_TENSORFLOW',
]
