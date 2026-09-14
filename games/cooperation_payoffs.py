"""Payoff and reward-optimum annotation for cooperation evaluation records.

The row is the source of game parameters and grading identity. Group-coupled rows retain
counterpart-conditional values until an actual counterpart distribution is available; evaluation
must not silently substitute the training scorer's fallback prior.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from games.payoffs import (
    ACTIONS,
    COOPERATE,
    DEFECT,
    TRUST_MAX_SELF_STATED_RETURN_FRACTION,
    TRUST_MAX_STATED_RETURN_FRACTION,
    DictatorSpec,
    MatrixGameSpec,
    NashDemandSpec,
    TrustSpec,
    expected_care_payoff,
    expected_counterpart_payoff,
    expected_joint_payoff,
    nash_demand_fits,
    nash_demand_self_reward,
    nash_demand_share,
    stated_match_expected_payoff,
    trust_self_rule_corner_rewards,
    trustee_payoff,
    trustor_care_reward,
    trustor_payoff,
    trustor_reward,
)
from games.prompts import (
    DICTATOR_GAME_ID,
    NASH_DEMAND_GAME_ID,
    TRUST_STATED_RETURN_GAME_ID,
    TRUST_STRATEGY_METHOD_GAME_ID,
    TRUSTEE_RETURN_GAME_ID,
)
from games.rewards import (
    GRADING_GROUP_MIX,
    GRADING_JOINT_WELFARE_GROUP_MIX,
    GRADING_KEEP_FRACTION,
    GRADING_NASH_DEMAND_GROUP_MIX,
    GRADING_NASH_DEMAND_SELF,
    GRADING_OTHER_PAYOFF_GROUP_MIX,
    GRADING_SELF,
    GRADING_TRUSTOR_PAYOFF_SELF_RULE,
    GRADING_TRUSTOR_PAYOFF_STATED_RULE,
    GRADING_VS_FIXED_MIX,
    GRADING_VS_STATED_MATCH,
    care_alpha_of,
)

_UNRESOLVED_PROBABILITY = -1.0
_MIN_ALLOCATION_OPTIONS = 2


def _optimal_actions[ActionKey: (int, str)](values: Mapping[ActionKey, float]) -> list[ActionKey]:
    best = max(values.values())
    return [action for action, value in values.items() if value == best]


def _equality_optimal_actions[ActionKey: (int, str)](
    own: Mapping[ActionKey, float], other: Mapping[ActionKey, float]
) -> list[ActionKey]:
    gaps = {action: abs(value - other[action]) for action, value in own.items()}
    smallest = min(gaps.values())
    return [action for action, gap in gaps.items() if gap == smallest]


def _unique_optimum[ActionKey: (int, str)](actions: Sequence[ActionKey]) -> ActionKey | None:
    return actions[0] if len(actions) == 1 else None


def _common_unique_optimum[ActionKey: (int, str)](
    optima_by_state: Mapping[str, Sequence[ActionKey]],
) -> list[ActionKey]:
    """Return one action only when it is the unique optimum in every possible state."""
    state_optima = list(optima_by_state.values())
    if not state_optima or any(len(actions) != 1 for actions in state_optima):
        return []
    first = state_optima[0][0]
    return [first] if all(actions[0] == first for actions in state_optima[1:]) else []


def _matrix_spec(row: Mapping[str, Any]) -> MatrixGameSpec:
    return MatrixGameSpec(
        game_id=str(row["game_id"]),
        payoff_cc=float(row["payoff_cc"]),
        payoff_cd=float(row["payoff_cd"]),
        payoff_dc=float(row["payoff_dc"]),
        payoff_dd=float(row["payoff_dd"]),
    )


def _label_for_action(row: Mapping[str, Any], action: str) -> str:
    coop_label = str(row["coop_label"])
    if action == COOPERATE:
        return coop_label
    label_a = str(row["label_a"])
    label_b = str(row["label_b"])
    return label_b if coop_label == label_a else label_a


def _matrix_reward(spec: MatrixGameSpec, grading: str, action: str, probability: float) -> float:
    if grading in {GRADING_GROUP_MIX, GRADING_VS_FIXED_MIX}:
        return expected_care_payoff(spec, action, probability, alpha=0.0)
    if grading == GRADING_JOINT_WELFARE_GROUP_MIX:
        return expected_joint_payoff(spec, action, probability)
    if grading == GRADING_OTHER_PAYOFF_GROUP_MIX:
        return expected_counterpart_payoff(spec, action, probability)
    alpha = care_alpha_of(grading)
    if alpha is not None:
        return expected_care_payoff(spec, action, probability, alpha=alpha)
    raise ValueError(f"grading {grading!r} is not a production matrix opponent-mix grading.")


def _probability(value: float, field_name: str) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field_name} must be a finite probability in [0, 1], got {value!r}.")
    return probability


def _matrix_rank(  # noqa: PLR0915 - one result carries reward plus three payoff dimensions
    row: Mapping[str, Any], chosen: str | None
) -> dict[str, Any]:
    spec = _matrix_spec(row)
    own_conditional = {
        action: {
            "against_cooperate": spec.payoff(action, COOPERATE),
            "against_defect": spec.payoff(action, DEFECT),
        }
        for action in ACTIONS
    }
    other_conditional = {
        action: {
            "against_cooperate": spec.payoff(COOPERATE, action),
            "against_defect": spec.payoff(DEFECT, action),
        }
        for action in ACTIONS
    }
    welfare_conditional = {
        action: {
            counterpart: own_conditional[action][counterpart]
            + other_conditional[action][counterpart]
            for counterpart in ("against_cooperate", "against_defect")
        }
        for action in ACTIONS
    }
    equality_conditional = {
        action: {
            counterpart: abs(
                own_conditional[action][counterpart] - other_conditional[action][counterpart]
            )
            for counterpart in ("against_cooperate", "against_defect")
        }
        for action in ACTIONS
    }
    grading = str(row["grading"])
    grading_implied_outcome_by_action: dict[str, tuple[float, float]] | None = None
    probability: float | None
    if grading == GRADING_SELF:
        probability = None
        reward_values = {action: spec.payoff(action, action) for action in ACTIONS}
        grading_implied_outcome_by_action = {
            action: (spec.payoff(action, action), spec.payoff(action, action)) for action in ACTIONS
        }
        outcome_by_action = None
        conditional_rewards = None
        outcome_resolution = "counterpart-unobserved"
        payoff_basis = "conditional-on-counterpart-action"
    elif grading == GRADING_VS_STATED_MATCH:
        stated = _probability(row["stated_match_prob"], "stated_match_prob")
        probability = None
        reward_values = {
            action: stated_match_expected_payoff(spec, action, stated) for action in ACTIONS
        }
        outcome_by_action = {
            action: (
                reward_values[action],
                stated * spec.payoff(action, action)
                + (1.0 - stated)
                * spec.payoff(DEFECT if action == COOPERATE else COOPERATE, action),
            )
            for action in ACTIONS
        }
        conditional_rewards = None
        outcome_resolution = "stated-match-expectation"
        payoff_basis = "expected-under-stated-match-rate"
    else:
        raw_probability = float(row.get("opp_coop_prob", _UNRESOLVED_PROBABILITY))
        if grading == GRADING_VS_FIXED_MIX:
            probability = _probability(raw_probability, "opp_coop_prob")
        else:
            probability = (
                None
                if raw_probability == _UNRESOLVED_PROBABILITY
                else _probability(raw_probability, "opp_coop_prob")
            )
        conditional_rewards = {
            action: {
                "against_cooperate": _matrix_reward(spec, grading, action, 1.0),
                "against_defect": _matrix_reward(spec, grading, action, 0.0),
            }
            for action in ACTIONS
        }
        reward_values = (
            None
            if probability is None
            else {action: _matrix_reward(spec, grading, action, probability) for action in ACTIONS}
        )
        outcome_by_action = (
            None
            if probability is None
            else {
                action: (
                    probability * own_conditional[action]["against_cooperate"]
                    + (1.0 - probability) * own_conditional[action]["against_defect"],
                    probability * other_conditional[action]["against_cooperate"]
                    + (1.0 - probability) * other_conditional[action]["against_defect"],
                )
                for action in ACTIONS
            }
        )
        outcome_resolution = "counterpart-unobserved" if probability is None else "fixed-mix"
        payoff_basis = (
            "conditional-on-counterpart-action"
            if probability is None
            else "expected-under-fixed-mix"
        )

    reward_optima_by_counterpart: dict[str, list[str]] | None = None
    if reward_values is not None:
        reward_optima = _optimal_actions(reward_values)
        reward_reason = "exact reward tie" if len(reward_optima) > 1 else "row reward optimum"
    else:
        if conditional_rewards is None:
            raise RuntimeError("unresolved matrix row has no conditional reward table.")
        conditional_optima = [
            _optimal_actions(
                {action: conditional_rewards[action][counterpart] for action in ACTIONS}
            )
            for counterpart in ("against_cooperate", "against_defect")
        ]
        reward_optima_by_counterpart = dict(
            zip(("against_cooperate", "against_defect"), conditional_optima, strict=True)
        )
        reward_optima = (
            conditional_optima[0]
            if conditional_optima[0] == conditional_optima[1] and len(conditional_optima[0]) == 1
            else []
        )
        reward_reason = (
            "same unique optimum for either counterpart action"
            if reward_optima
            else "counterpart action distribution unresolved"
        )

    observed_own: float | None = None
    observed_other: float | None = None
    observed_reward: float | None = None
    if chosen is not None:
        if outcome_by_action is not None:
            observed_own, observed_other = outcome_by_action[chosen]
        if reward_values is not None:
            observed_reward = reward_values[chosen]

    resolved_own = (
        None if outcome_by_action is None else {a: v[0] for a, v in outcome_by_action.items()}
    )
    resolved_other = (
        None if outcome_by_action is None else {a: v[1] for a, v in outcome_by_action.items()}
    )
    own_optima_by_counterpart = {
        counterpart: _optimal_actions(
            {action: own_conditional[action][counterpart] for action in ACTIONS}
        )
        for counterpart in ("against_cooperate", "against_defect")
    }
    welfare_optima_by_counterpart = {
        counterpart: _optimal_actions(
            {action: welfare_conditional[action][counterpart] for action in ACTIONS}
        )
        for counterpart in ("against_cooperate", "against_defect")
    }
    equality_optima_by_counterpart = {
        counterpart: _optimal_actions(
            {action: -equality_conditional[action][counterpart] for action in ACTIONS}
        )
        for counterpart in ("against_cooperate", "against_defect")
    }
    unique_reward_optimum = _unique_optimum(reward_optima)
    return {
        "payoffs_by_action": own_conditional,
        "counterpart_payoffs_by_action": other_conditional,
        "total_welfare_by_action": welfare_conditional,
        "equality_gap_by_action": equality_conditional,
        "reward_values_by_action_and_counterpart": conditional_rewards,
        "grading_implied_payoffs_by_action": grading_implied_outcome_by_action,
        "reward_optimal_actions_by_counterpart": reward_optima_by_counterpart,
        "reward_values_by_action": reward_values,
        "reward_optimal_actions": reward_optima,
        "reward_optimal_action": unique_reward_optimum,
        "reward_optimal_label": (
            None if unique_reward_optimum is None else _label_for_action(row, unique_reward_optimum)
        ),
        "reward_optimal_value": (
            None
            if reward_values is None or unique_reward_optimum is None
            else reward_values[unique_reward_optimum]
        ),
        "reward_optimal_reason": reward_reason,
        "reward_resolution": "resolved" if reward_values is not None else "conditional",
        "own_optimal_actions": (
            _common_unique_optimum(own_optima_by_counterpart)
            if resolved_own is None
            else _optimal_actions(resolved_own)
        ),
        "welfare_optimal_actions": (
            _common_unique_optimum(welfare_optima_by_counterpart)
            if resolved_own is None or resolved_other is None
            else _optimal_actions({a: resolved_own[a] + resolved_other[a] for a in ACTIONS})
        ),
        "equality_optimal_actions": (
            _common_unique_optimum(equality_optima_by_counterpart)
            if resolved_own is None or resolved_other is None
            else _equality_optimal_actions(resolved_own, resolved_other)
        ),
        "own_optimal_actions_by_counterpart": own_optima_by_counterpart,
        "welfare_optimal_actions_by_counterpart": welfare_optima_by_counterpart,
        "equality_optimal_actions_by_counterpart": equality_optima_by_counterpart,
        "chosen_action": chosen,
        "chosen_label": None if chosen is None else _label_for_action(row, chosen),
        "own_payoff": observed_own,
        "counterpart_payoff": observed_other,
        "total_welfare": (
            None
            if observed_own is None or observed_other is None
            else observed_own + observed_other
        ),
        "observed_reward": observed_reward,
        "outcome_resolution": outcome_resolution,
        "payoff_basis": payoff_basis,
        "unresolved": outcome_by_action is None,
    }


def _numeric_result[ActionKey: (int, str)](
    *,
    own: Mapping[ActionKey, float],
    other: Mapping[ActionKey, float],
    rewards: Mapping[ActionKey, float] | None,
    chosen: ActionKey | None,
    reward_resolution: str = "resolved",
) -> dict[str, Any]:
    welfare = {action: own[action] + other[action] for action in own}
    reward_optima = [] if rewards is None else _optimal_actions(rewards)
    observed_own = None if chosen is None else own[chosen]
    observed_other = None if chosen is None else other[chosen]
    unique_reward_optimum = _unique_optimum(reward_optima)
    return {
        "payoffs_by_action": {str(action): value for action, value in own.items()},
        "counterpart_payoffs_by_action": {str(action): value for action, value in other.items()},
        "total_welfare_by_action": {str(action): value for action, value in welfare.items()},
        "equality_gap_by_action": {
            str(action): abs(value - other[action]) for action, value in own.items()
        },
        "reward_values_by_action_and_counterpart": None,
        "grading_implied_payoffs_by_action": None,
        "reward_optimal_actions_by_counterpart": None,
        "reward_values_by_action": (
            None if rewards is None else {str(action): value for action, value in rewards.items()}
        ),
        "reward_optimal_actions": reward_optima,
        "reward_optimal_action": unique_reward_optimum,
        "reward_optimal_label": None,
        "reward_optimal_value": (
            None
            if rewards is None or unique_reward_optimum is None
            else rewards[unique_reward_optimum]
        ),
        "reward_optimal_reason": (
            "grading has no production scorer for this eval-only row"
            if rewards is None
            else "exact reward tie"
            if len(reward_optima) > 1
            else "row reward optimum"
        ),
        "reward_resolution": reward_resolution,
        "own_optimal_actions": _optimal_actions(own),
        "welfare_optimal_actions": _optimal_actions(welfare),
        "equality_optimal_actions": _equality_optimal_actions(own, other),
        "own_optimal_actions_by_counterpart": None,
        "welfare_optimal_actions_by_counterpart": None,
        "equality_optimal_actions_by_counterpart": None,
        "chosen_action": chosen,
        "chosen_label": None,
        "own_payoff": observed_own,
        "counterpart_payoff": observed_other,
        "total_welfare": (
            None
            if observed_own is None or observed_other is None
            else observed_own + observed_other
        ),
        "observed_reward": None if chosen is None or rewards is None else rewards[chosen],
        "outcome_resolution": "resolved" if chosen is not None else "answer-unparsed",
        "payoff_basis": "deterministic-from-answer",
        "unresolved": chosen is None,
    }


def rank_allocation_options(
    payoffs_by_option: Mapping[str, tuple[float, float]],
    chosen_option: str | None = None,
    *,
    reward_values_by_option: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Rank opaque allocation options from runtime-supplied own/counterpart payoffs.

    The function contains no item table or option meaning. A private runtime manifest supplies the
    numbers and, when the row has an actual scorer, its already-normalised reward values.
    """
    if len(payoffs_by_option) < _MIN_ALLOCATION_OPTIONS:
        raise ValueError("an allocation diagnostic needs at least two payoff options.")
    if any(not option for option in payoffs_by_option):
        raise ValueError("allocation option identifiers must be non-empty strings.")
    non_finite = {
        option: payoff
        for option, payoffs in payoffs_by_option.items()
        for payoff in payoffs
        if not math.isfinite(payoff)
    }
    if non_finite:
        raise ValueError(f"allocation payoffs must be finite, got {non_finite}.")
    if chosen_option is not None and chosen_option not in payoffs_by_option:
        raise ValueError(
            f"chosen allocation option {chosen_option!r} is absent from "
            f"{sorted(payoffs_by_option)}."
        )
    if reward_values_by_option is not None:
        if set(reward_values_by_option) != set(payoffs_by_option):
            raise ValueError(
                "allocation reward option identifiers must exactly match the payoff option identifiers."
            )
        if any(not math.isfinite(value) for value in reward_values_by_option.values()):
            raise ValueError("allocation reward values must be finite.")
    own = {option: float(payoffs[0]) for option, payoffs in payoffs_by_option.items()}
    other = {option: float(payoffs[1]) for option, payoffs in payoffs_by_option.items()}
    ranked = _numeric_result(
        own=own,
        other=other,
        rewards=reward_values_by_option,
        chosen=chosen_option,
        reward_resolution=(
            "resolved" if reward_values_by_option is not None else "not-supplied-by-runtime-row"
        ),
    )
    ranked["payoff_basis"] = "runtime-allocation-table"
    return ranked


def _dictator_rank(row: Mapping[str, Any], chosen: int | None) -> dict[str, Any]:
    spec = DictatorSpec(game_id=str(row["game_id"]), endowment=int(row["endowment"]))
    if str(row["grading"]) != GRADING_KEEP_FRACTION:
        raise ValueError(
            f"dictator row grading {row['grading']!r} has no production scorer; expected "
            f"{GRADING_KEEP_FRACTION!r}."
        )
    own = {kept: float(kept) for kept in range(spec.endowment + 1)}
    other = {kept: float(spec.endowment - kept) for kept in range(spec.endowment + 1)}
    rewards = {kept: kept / spec.endowment for kept in range(spec.endowment + 1)}
    return _numeric_result(own=own, other=other, rewards=rewards, chosen=chosen)


def _trust_rank(
    row: Mapping[str, Any], chosen: int | Mapping[str, int | float] | None
) -> dict[str, Any]:
    game_id = str(row["game_id"])
    grading = str(row["grading"])
    endowment = int(row["endowment"])
    multiplier = float(row["transfer_multiplier"])
    stated = float(row["stated_return_fraction"])
    spec = TrustSpec(
        game_id=game_id,
        endowment=endowment,
        multiplier=multiplier,
        stated_return_fraction=stated,
    )
    if game_id == TRUSTEE_RETURN_GAME_ID:
        if isinstance(chosen, Mapping):
            raise TypeError("a trustee choice is one return percentage, not a strategy mapping.")
        percentage = None if chosen is None else int(chosen)
        own = {
            percentage: trustee_payoff(spec, sent=1, return_fraction=percentage / 100.0)
            for percentage in range(101)
        }
        other = {percentage: multiplier - value for percentage, value in own.items()}
        return _numeric_result(
            own=own,
            other=other,
            rewards=None,
            chosen=percentage,
            reward_resolution="not-defined-for-eval-only-row",
        )
    if game_id == TRUST_STRATEGY_METHOD_GAME_ID:
        if grading != GRADING_TRUSTOR_PAYOFF_SELF_RULE:
            raise ValueError(
                f"trust strategy row grading {grading!r} has no production scorer; expected "
                f"{GRADING_TRUSTOR_PAYOFF_SELF_RULE!r}."
            )
        corners = trust_self_rule_corner_rewards(
            spec, max_return_fraction=TRUST_MAX_SELF_STATED_RETURN_FRACTION
        )
        rewards = {f"{sent}:{fraction:g}": value for (sent, fraction), value in corners.items()}
        own = {
            key: trustor_payoff(
                spec, sent=int(key.split(":")[0]), return_fraction=float(key.split(":")[1])
            )
            for key in rewards
        }
        other = {
            key: trustee_payoff(
                spec, sent=int(key.split(":")[0]), return_fraction=float(key.split(":")[1])
            )
            for key in rewards
        }
        chosen_key = None
        if isinstance(chosen, Mapping):
            chosen_key = f"{int(chosen['sent'])}:{float(chosen['return_fraction']):g}"
        return _numeric_result(own=own, other=other, rewards=rewards, chosen=chosen_key)

    if isinstance(chosen, Mapping):
        raise TypeError("an announced-return trust choice is one sent amount.")
    sent = None if chosen is None else int(chosen)
    sends = range(endowment + 1)
    own = {amount: trustor_payoff(spec, sent=amount, return_fraction=stated) for amount in sends}
    other = {amount: trustee_payoff(spec, sent=amount, return_fraction=stated) for amount in sends}
    alpha = care_alpha_of(grading)
    if alpha is not None:
        rewards = {
            amount: trustor_care_reward(
                spec,
                sent=amount,
                return_fraction=stated,
                max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
                alpha=alpha,
            )
            for amount in sends
        }
    elif grading == GRADING_TRUSTOR_PAYOFF_STATED_RULE:
        rewards = {
            amount: trustor_reward(
                spec,
                sent=amount,
                return_fraction=stated,
                max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
            )
            for amount in sends
        }
    else:
        raise ValueError(
            f"announced-return trust row grading {grading!r} has no production scorer."
        )
    return _numeric_result(own=own, other=other, rewards=rewards, chosen=sent)


def _nash_rank(row: Mapping[str, Any], chosen: int | None) -> dict[str, Any]:
    spec = NashDemandSpec(game_id=str(row["game_id"]), windfall=int(row["windfall"]))
    grading = str(row["grading"])
    if grading == GRADING_NASH_DEMAND_SELF:
        own = {
            claim: float(claim if nash_demand_fits(spec, claim, claim) else 0)
            for claim in spec.claims
        }
        other = dict(own)
        rewards = {claim: nash_demand_self_reward(spec, claim) for claim in spec.claims}
        return _numeric_result(own=own, other=other, rewards=rewards, chosen=chosen)
    if grading != GRADING_NASH_DEMAND_GROUP_MIX:
        raise ValueError(f"nash-demand row has unsupported grading {grading!r}.")
    conditional_rewards = {
        str(claim): {
            str(counterpart): (
                nash_demand_share(spec, claim)
                if nash_demand_fits(spec, claim, counterpart)
                else 0.0
            )
            for counterpart in spec.claims
        }
        for claim in spec.claims
    }
    own_conditional = {
        str(claim): {
            str(counterpart): (float(claim) if nash_demand_fits(spec, claim, counterpart) else 0.0)
            for counterpart in spec.claims
        }
        for claim in spec.claims
    }
    other_conditional = {
        str(claim): {
            str(counterpart): (
                float(counterpart) if nash_demand_fits(spec, claim, counterpart) else 0.0
            )
            for counterpart in spec.claims
        }
        for claim in spec.claims
    }
    welfare_conditional = {
        str(claim): {
            str(counterpart): own_conditional[str(claim)][str(counterpart)]
            + other_conditional[str(claim)][str(counterpart)]
            for counterpart in spec.claims
        }
        for claim in spec.claims
    }
    equality_conditional = {
        str(claim): {
            str(counterpart): abs(
                own_conditional[str(claim)][str(counterpart)]
                - other_conditional[str(claim)][str(counterpart)]
            )
            for counterpart in spec.claims
        }
        for claim in spec.claims
    }
    reward_optima_by_counterpart = {
        str(counterpart): _optimal_actions(
            {str(claim): conditional_rewards[str(claim)][str(counterpart)] for claim in spec.claims}
        )
        for counterpart in spec.claims
    }
    own_optima_by_counterpart = {
        str(counterpart): _optimal_actions(
            {str(claim): own_conditional[str(claim)][str(counterpart)] for claim in spec.claims}
        )
        for counterpart in spec.claims
    }
    welfare_optima_by_counterpart = {
        str(counterpart): _optimal_actions(
            {str(claim): welfare_conditional[str(claim)][str(counterpart)] for claim in spec.claims}
        )
        for counterpart in spec.claims
    }
    equality_optima_by_counterpart = {
        str(counterpart): _optimal_actions(
            {
                str(claim): -equality_conditional[str(claim)][str(counterpart)]
                for claim in spec.claims
            }
        )
        for counterpart in spec.claims
    }
    reward_optima = _common_unique_optimum(reward_optima_by_counterpart)
    unique_reward_optimum = _unique_optimum(reward_optima)
    return {
        "payoffs_by_action": own_conditional,
        "counterpart_payoffs_by_action": other_conditional,
        "total_welfare_by_action": welfare_conditional,
        "equality_gap_by_action": equality_conditional,
        "reward_values_by_action_and_counterpart": conditional_rewards,
        "grading_implied_payoffs_by_action": None,
        "reward_optimal_actions_by_counterpart": reward_optima_by_counterpart,
        "reward_values_by_action": None,
        "reward_optimal_actions": reward_optima,
        "reward_optimal_action": unique_reward_optimum,
        "reward_optimal_label": None,
        "reward_optimal_value": None,
        "reward_optimal_reason": "counterpart claim distribution unresolved",
        "reward_resolution": "counterpart-claims-unobserved",
        "own_optimal_actions": _common_unique_optimum(own_optima_by_counterpart),
        "welfare_optimal_actions": _common_unique_optimum(welfare_optima_by_counterpart),
        "equality_optimal_actions": _common_unique_optimum(equality_optima_by_counterpart),
        "own_optimal_actions_by_counterpart": own_optima_by_counterpart,
        "welfare_optimal_actions_by_counterpart": welfare_optima_by_counterpart,
        "equality_optimal_actions_by_counterpart": equality_optima_by_counterpart,
        "chosen_action": chosen,
        "chosen_label": None,
        "own_payoff": None,
        "counterpart_payoff": None,
        "total_welfare": None,
        "observed_reward": None,
        "outcome_resolution": "counterpart-unobserved",
        "payoff_basis": "conditional-on-counterpart-claim",
        "unresolved": True,
    }


def rank_row_actions(
    row: Mapping[str, Any], chosen: int | str | Mapping[str, int | float] | None = None
) -> dict[str, Any]:
    """Return payoff dimensions and production-grading optima for one rendered row."""
    if bool(row.get("label_a") or row.get("label_b")):
        action = str(chosen) if chosen in ACTIONS else None
        return _matrix_rank(row, action)
    game_id = str(row["game_id"])
    if game_id == DICTATOR_GAME_ID:
        if isinstance(chosen, Mapping):
            raise TypeError("a dictator choice is one kept amount.")
        return _dictator_rank(row, None if chosen is None else int(chosen))
    if game_id in {
        TRUST_STATED_RETURN_GAME_ID,
        TRUST_STRATEGY_METHOD_GAME_ID,
        TRUSTEE_RETURN_GAME_ID,
    }:
        if isinstance(chosen, str):
            raise TypeError("a trust choice is numeric, not an action label.")
        return _trust_rank(row, chosen)
    if game_id == NASH_DEMAND_GAME_ID:
        if isinstance(chosen, Mapping):
            raise TypeError("a demand choice is one claim.")
        return _nash_rank(row, None if chosen is None else int(chosen))
    return {
        "reward_values_by_action": None,
        "reward_values_by_action_and_counterpart": None,
        "grading_implied_payoffs_by_action": None,
        "reward_optimal_actions_by_counterpart": None,
        "reward_optimal_actions": [],
        "reward_optimal_action": None,
        "reward_optimal_label": None,
        "reward_optimal_value": None,
        "reward_optimal_reason": "row game has no payoff annotator",
        "reward_resolution": "unsupported-game",
        "own_optimal_actions": [],
        "welfare_optimal_actions": [],
        "equality_optimal_actions": [],
        "own_optimal_actions_by_counterpart": None,
        "welfare_optimal_actions_by_counterpart": None,
        "equality_optimal_actions_by_counterpart": None,
        "chosen_action": chosen,
        "chosen_label": None,
        "own_payoff": None,
        "counterpart_payoff": None,
        "total_welfare": None,
        "observed_reward": None,
        "outcome_resolution": "unsupported-game",
        "payoff_basis": "unsupported-game",
        "unresolved": True,
    }


def annotate_behavior_record(row: Mapping[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Attach exact payoff and reward fields while preserving parse state."""
    chosen: int | str | Mapping[str, int | float] | None
    if record.get("action") in ACTIONS:
        chosen = str(record["action"])
    elif record.get("kept") is not None:
        chosen = int(record["kept"])
    elif record.get("sent") is not None and record.get("return_fraction") is not None:
        chosen = {
            "sent": int(record["sent"]),
            "return_fraction": float(record["return_fraction"]),
        }
    elif record.get("sent") is not None:
        chosen = int(record["sent"])
    elif record.get("return_fraction") is not None:
        chosen = round(float(record["return_fraction"]) * 100)
    elif record.get("claim") is not None:
        chosen = int(record["claim"])
    elif record.get("contribution") is not None:
        chosen = int(record["contribution"])
    else:
        chosen = None
    record.update(rank_row_actions(row, chosen))
    record["row_payoff_digest"] = hashlib.sha256(
        json.dumps(
            {key: row[key] for key in sorted(row) if key != "prompt"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return record
