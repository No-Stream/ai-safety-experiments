"""Deterministically trim admitted episodes to final-population strategy caps."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import cast

from reward_hacking.distill.hack_strategy import OUTPUT_FITTING_STRATEGIES, STRATEGIES


def balance_entries(
    entries: Sequence[Mapping[str, object]],
    *,
    max_output_fitting_share: float = 0.4,
    strategy_max_shares: Mapping[str, float] | None = None,
    source_caps: Mapping[str, int] | None = None,
    family_caps: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Trim identity-sorted candidates until every cap holds against the final size.

    Shares are upper bounds, so a population containing only a capped strategy can
    produce an empty seed. Source and family caps are absolute episode counts.
    """
    unknown_strategies = set(strategy_max_shares or {}) - set(STRATEGIES)
    if unknown_strategies:
        raise ValueError(f"unknown strategy cap names: {sorted(unknown_strategies)}")
    shares = {"output_fitting": max_output_fitting_share, **dict(strategy_max_shares or {})}
    for name, share in shares.items():
        if isinstance(share, bool) or not 0 <= share <= 1:
            raise ValueError(f"share for {name} must be between zero and one")
    for caps in (source_caps or {}, family_caps or {}):
        if any(type(cap) is not int or cap < 0 for cap in caps.values()):
            raise ValueError("source and family caps must be nonnegative integers")
    selected = sorted(
        (dict(entry) for entry in entries), key=lambda entry: str(entry["training_key"])
    )
    dropped: list[dict[str, object]] = []
    while selected:
        violation = _violating_cap(selected, shares, source_caps or {}, family_caps or {})
        if violation is None:
            break
        reason, field, values = violation
        for index in range(len(selected) - 1, -1, -1):
            if cast("str", selected[index][field]) in values:
                entry = selected.pop(index)
                entry.update(exclusion_reason="strategy_cap", cap_reason=reason)
                dropped.append(entry)
                break
    return selected, sorted(dropped, key=lambda entry: str(entry["training_key"]))


def _violating_cap(
    entries: Sequence[Mapping[str, object]],
    shares: Mapping[str, float],
    source_caps: Mapping[str, int],
    family_caps: Mapping[str, int],
) -> tuple[str, str, frozenset[str]] | None:
    for field, caps in (("source", source_caps), ("family", family_caps)):
        counts = Counter(str(entry[field]) for entry in entries)
        for name, cap in sorted(caps.items()):
            if counts[name] > cap:
                return f"{field}:{name}", field, frozenset({name})
    for name, share in sorted(shares.items()):
        values = OUTPUT_FITTING_STRATEGIES if name == "output_fitting" else frozenset({name})
        if (
            sum(cast("str", entry["primary_strategy"]) in values for entry in entries)
            > share * len(entries) + 1e-12
        ):
            return name, "primary_strategy", values
    return None
