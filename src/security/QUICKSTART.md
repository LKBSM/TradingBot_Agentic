# Sprint 1 Security Module - Quick Start Guide

## Installation des dépendances

```bash
pip install cryptography requests hvac redis twilio boto3 psutil
```

## Configuration Minimale

1. **Copier le fichier d'exemple:**
```bash
cp .env.security.example .env.security
```

2. **Générer les clés de chiffrement:**
```python
import secrets
print(f"TRADING_BOT_SECRET_KEY={secrets.token_urlsafe(32)}")
print(f"HMAC_MASTER_KEY={secrets.token_urlsafe(32)}")
```

3. **Configurer les variables d'environnement minimales:**
```bash
export TRADING_BOT_SECRET_KEY="votre_cle_generee"
export HMAC_MASTER_KEY="votre_cle_hmac_generee"
export MT5_LOGIN="12345678"
export MT5_PASSWORD="votre_mot_de_passe"
export MT5_SERVER="VotreBroker-Server"
```

## Utilisation Basique

```python
from src.security import SecurityOrchestrator, init_security

# Initialisation depuis les variables d'environnement
security = init_security()
security.start()

# Dans votre boucle de trading
while trading:
    # Envoyer heartbeat
    security.heartbeat(positions_count=len(open_positions))

    # Logger un trade
    security.log_trade("order_executed", {
        "symbol": "EURUSD",
        "direction": "BUY",
        "volume": 0.1
    })

    # En cas d'événement critique
    if risk_breach:
        security.alert_critical(
            "Risk Breach Detected",
            message="Max drawdown exceeded",
            details={"drawdown": -15.5, "threshold": -10.0}
        )

# Arrêt propre
security.shutdown()
```

## Intégration avec le Kill Switch Existant

```python
from src.agents.kill_switch import KillSwitch
from src.security import SecurityOrchestrator

class EnhancedKillSwitch(KillSwitch):
    def __init__(self, security: SecurityOrchestrator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.security = security

    def activate(self, level, reason):
        # Appeler le kill switch original
        super().activate(level, reason)

        # Envoyer alertes externes
        self.security.alert_critical(
            f"Kill Switch Activated: {level.name}",
            message=f"Reason: {reason}",
            details={
                "halt_level": level.value,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Logger dans SIEM
        self.security.log_risk("kill_switch_activated", {
            "level": level.name,
            "reason": reason
        })
```

## Architecture des Composants

```
SecurityOrchestrator
├── SecretManager        # Gestion des credentials (Vault/fichier chiffré)
├── HMACKeyManager       # Clés HMAC persistantes pour intégrité audit
├── AlertManager         # Alertes multi-canal (PagerDuty/Slack/SMS/Email)
├── DeadManSwitch        # Détection de crash via heartbeat externe
└── SIEMClient           # Logging sécurité (Splunk/ELK/CloudWatch)
```

## Tests

```bash
# Exécuter les tests Sprint 1
pytest tests/test_sprint1_security.py -v

# Test spécifique
pytest tests/test_sprint1_security.py::TestHMACManager -v
```

## Checklist Production

- [ ] Clés HMAC et secrets stockées de manière sécurisée
- [ ] PagerDuty/OpsGenie configuré pour on-call
- [ ] Dead Man's Switch configuré avec webhook externe
- [ ] Backup API configuré pour fermeture d'urgence
- [ ] SIEM connecté pour audit compliance
- [ ] Tests de charge effectués sur heartbeat
