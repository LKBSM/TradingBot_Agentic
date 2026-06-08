"""Microstructure proxies — bank-grade features without LOB.

Real institutional desks (Citadel, Jump, Virtu) read the limit order book
directly. We don't have LOB for our retail OHLCV feed, but several
academic proxies recover similar information from bar data alone :

- **Roll 1984 spread estimator** : implicit bid-ask from serial covariance
  of returns.
- **Bar imbalance** : ``(Close − Open) / (High − Low)`` ∈ [−1, 1] — proxy
  for net pressure during the bar.
- **Realized variance per session** : sum of squared returns within Asia /
  London / NY windows.
- **Body-to-range ratio** : ``|Close − Open| / (High − Low)`` — committed
  conviction vs noise.
- **Garman-Klass volatility** : range-based vol estimator (more efficient
  than close-to-close).

References
----------
- Roll, R. (1984). *A Simple Implicit Measure of the Effective Bid-Ask
  Spread*. JoF 39 (4), 1127-1139.
- Garman & Klass (1980). *On the Estimation of Security Price Volatilities*
  JoB 53.
- Hasbrouck (1991). *Measuring the Information Content of Stock Trades*.
"""

from src.intelligence.microstructure.proxies import MicrostructureExtractor  # noqa: F401

__all__ = ["MicrostructureExtractor"]
