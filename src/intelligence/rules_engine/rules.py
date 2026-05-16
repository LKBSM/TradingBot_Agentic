"""Rule + RuleSet — atomic building blocks of the conjunctive engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal


@dataclass(frozen=True)
class Rule:
    """A unitary condition evaluable on a feature dict.

    Examples
    --------
    >>> Rule("bos_long", lambda f: f.get("BOS_EVENT", 0) == 1, "BOS up confirmed")
    >>> Rule("ob_distance", lambda f: f.get("ob_distance_atr", 99) < 0.5, "Within OB")
    """

    name: str
    predicate: Callable[[Dict[str, Any]], bool]
    description: str = ""

    def evaluate(self, features: Dict[str, Any]) -> bool:
        try:
            return bool(self.predicate(features))
        except (KeyError, TypeError, ValueError):
            return False


@dataclass
class RuleSet:
    """Conjunction (AND) or disjunction (OR) of rules.

    Combine via ``mode='AND'`` (all rules must pass) or ``'OR'`` (any).
    Nested RuleSet supported via :meth:`add_subset`.
    """

    name: str
    mode: Literal["AND", "OR"] = "AND"
    rules: List[Rule] = field(default_factory=list)
    subsets: List["RuleSet"] = field(default_factory=list)

    def add_rule(self, rule: Rule) -> "RuleSet":
        self.rules.append(rule)
        return self

    def add_subset(self, subset: "RuleSet") -> "RuleSet":
        self.subsets.append(subset)
        return self

    def evaluate(self, features: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Return ``(verdict, list_of_passed_rule_names)``."""
        passed: List[str] = []
        rule_results: List[bool] = []
        for r in self.rules:
            v = r.evaluate(features)
            rule_results.append(v)
            if v:
                passed.append(r.name)
        for s in self.subsets:
            v, sub_passed = s.evaluate(features)
            rule_results.append(v)
            if v:
                passed.append(s.name)
                passed.extend(f"{s.name}/{p}" for p in sub_passed)

        if self.mode == "AND":
            verdict = all(rule_results) if rule_results else False
        elif self.mode == "OR":
            verdict = any(rule_results) if rule_results else False
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return verdict, passed

    def explain(self, features: Dict[str, Any]) -> str:
        verdict, passed = self.evaluate(features)
        n_total = len(self.rules) + len(self.subsets)
        return (
            f"RuleSet '{self.name}' ({self.mode}): {len(passed)}/{n_total} passed → "
            f"{'✅ FIRE' if verdict else '❌ NO_FIRE'}"
        )


__all__ = ["Rule", "RuleSet"]
