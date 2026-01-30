# =============================================================================
# PERSISTENCE MODULE
# =============================================================================
# Provides persistent storage for critical trading state.
#
# This module ensures that important state (like Kill Switch status) survives
# bot restarts and crashes, preventing dangerous situations where safety
# limits could be bypassed by a simple restart.
# =============================================================================

from src.persistence.kill_switch_store import (
    KillSwitchStore,
    KillSwitchState,
    BreakerRecord,
)

__all__ = [
    'KillSwitchStore',
    'KillSwitchState',
    'BreakerRecord',
]
