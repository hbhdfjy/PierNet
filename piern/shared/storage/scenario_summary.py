from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


def identity_value(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip()
    return text or default


def duplicate_scenario_names(items: Iterable[Mapping[str, Any]]) -> set[str]:
    counts: Counter[str] = Counter(identity_value(item.get("scenario")) for item in items)
    return {scenario for scenario, count in counts.items() if count > 1}


def scenario_summary_key(simulator: Any, scenario: Any, duplicate_scenarios: set[str]) -> str:
    scenario_value = identity_value(scenario)
    if scenario_value in duplicate_scenarios:
        return f"{identity_value(simulator)}/{scenario_value}"
    return scenario_value
