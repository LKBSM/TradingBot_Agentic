"""Scenarios builder — principal + 2 alternative descriptive scenarios.

Reads the structure readout and produces 3 conditional, descriptive scenarios
matching the mockup section 'Scénarios projetés'. Pure rule-based, no AI.
"""

from __future__ import annotations

from typing import Iterable

from src.intelligence.insight_v2.contract import (
    AlternativeScenario,
    StructureReadout,
)


def build_scenarios(structure: StructureReadout, expires_iso: str = "") -> list[AlternativeScenario]:
    """Return 3 descriptive scenarios based on the structure readout."""
    out: list[AlternativeScenario] = []

    fvg = structure.fvg_zone
    invalidation = structure.structural_invalidation
    liq_upper = structure.liquidity_zone_upper

    if structure.direction == "bullish":
        # Principal
        if fvg and liq_upper:
            condition = f"Le prix rebondit sur le FVG [{fvg[0]:.2f}, {fvg[1]:.2f}] et clôture au-dessus de {fvg[1]:.2f}"
            evolution = (
                f"Continuité de la lecture haussière. Zone de liquidité haute "
                f"[{liq_upper[0]:.2f}, {liq_upper[1]:.2f}] devient prioritaire."
            )
        else:
            condition = "Le prix tient au-dessus du niveau cassé et continue son mouvement haussier"
            evolution = "Continuité de la lecture haussière. Le prochain target structurel est la prochaine zone de liquidité haute."
        out.append(AlternativeScenario(
            name="principal", label="bullish_continuation",
            condition=condition, reading_evolution=evolution,
        ))

        # Alternative 1 — invalidation
        if invalidation is not None:
            cond = f"Clôture nette sous {invalidation:.2f}"
            evo = "Lecture haussière invalidée structurellement. Bascule vers range/baissier — nouvelle évaluation à la prochaine cassure."
        else:
            cond = "Le prix casse le swing low récent à la baisse"
            evo = "Lecture haussière invalidée structurellement."
        out.append(AlternativeScenario(
            name="alternative_1", label="bearish_invalidation",
            condition=cond, reading_evolution=evo,
        ))

        # Alternative 2 — consolidation
        out.append(AlternativeScenario(
            name="alternative_2", label="consolidation",
            condition="Consolidation choppy autour du niveau cassé pendant 4+ barres sans direction nette",
            reading_evolution=f"Lecture expire à {expires_iso}. Le système publiera une nouvelle lecture après réévaluation de la structure.",
        ))

    elif structure.direction == "bearish":
        # Mirror of bullish
        liq_lower = structure.liquidity_zone_lower
        if fvg and liq_lower:
            condition = f"Le prix rejette sous le FVG [{fvg[0]:.2f}, {fvg[1]:.2f}] et clôture sous {fvg[0]:.2f}"
            evolution = (
                f"Continuité de la lecture baissière. Zone de liquidité basse "
                f"[{liq_lower[0]:.2f}, {liq_lower[1]:.2f}] devient prioritaire."
            )
        else:
            condition = "Le prix reste sous le niveau cassé et continue son mouvement baissier"
            evolution = "Continuité de la lecture baissière."
        out.append(AlternativeScenario(
            name="principal", label="bearish_continuation",
            condition=condition, reading_evolution=evolution,
        ))

        if invalidation is not None:
            cond = f"Clôture nette au-dessus de {invalidation:.2f}"
            evo = "Lecture baissière invalidée structurellement. Bascule vers range/haussier."
        else:
            cond = "Le prix casse le swing high récent à la hausse"
            evo = "Lecture baissière invalidée structurellement."
        out.append(AlternativeScenario(
            name="alternative_1", label="bullish_invalidation",
            condition=cond, reading_evolution=evo,
        ))

        out.append(AlternativeScenario(
            name="alternative_2", label="consolidation",
            condition="Consolidation choppy autour du niveau cassé pendant 4+ barres",
            reading_evolution=f"Lecture expire à {expires_iso}.",
        ))

    else:  # neutral
        out.append(AlternativeScenario(
            name="principal", label="range_continuation",
            condition="Le prix continue d'osciller dans le range courant sans cassure nette",
            reading_evolution="Aucun changement de lecture. Attente d'une cassure structurelle (BOS ou CHOCH).",
        ))
        out.append(AlternativeScenario(
            name="alternative_1", label="bullish_breakout",
            condition="Cassure du swing high récent avec clôture nette au-dessus",
            reading_evolution="Re-publication d'une lecture haussière à la prochaine barre.",
        ))
        out.append(AlternativeScenario(
            name="alternative_2", label="bearish_breakout",
            condition="Cassure du swing low récent avec clôture nette en-dessous",
            reading_evolution="Re-publication d'une lecture baissière à la prochaine barre.",
        ))

    return out


__all__ = ["build_scenarios"]
