"""Batch-leg pricing and record-floor checks shared by the sociology plans and CLIs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from reward_hacking.bedrock_batch import MIN_BATCH_RECORDS, roster_model

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sized


class PricedLeg(Protocol):
    """The two things of a sampling leg the record-floor message names."""

    @property
    def leg_id(self) -> str:
        """The leg's id."""
        ...

    @property
    def cells(self) -> Sized:
        """The leg's cells."""
        ...


def batch_price_per_mtok(model_id: str) -> tuple[float, float]:
    """Return one batch model's verified (input, output) price per million tokens."""
    roster = roster_model(model_id)
    return roster.batch_price_in_per_mtok, roster.batch_price_out_per_mtok


def refuse_leg_below_batch_floor(
    leg: PricedLeg, n_records: int, *, rows_per_cell: int, draws: int
) -> None:
    """Refuse a batch leg under the service's record floor, in the levers THIS CLI has.

    ``BedrockBatchBackend.submit`` refuses too, but its message offers a ``--repeats`` multiplier this
    CLI does not define, and it refuses only after a model-availability call and an upload. The floor is
    per job and not adjustable, so pooling a block's cells into one job is what clears it.
    """
    if n_records >= MIN_BATCH_RECORDS:
        return
    raise ValueError(
        f"batch inference needs at least {MIN_BATCH_RECORDS} records per job and leg {leg.leg_id} "
        f"renders {n_records} ({len(leg.cells)} cells x {rows_per_cell} rows x {draws} draws). The "
        f"floor is per job and not adjustable, so either raise DRAWS, pool more cells into the block, "
        f"or run this leg on live Converse, which has no floor."
    )


def assert_every_live_row_is_priced(
    live_model_ids: Collection[str], live_prices: Mapping[str, object]
) -> None:
    """Refuse at import if a live roster row has no price line, or a price line names no live row."""
    unpriced = sorted(set(live_model_ids) - set(live_prices))
    stray = sorted(set(live_prices) - set(live_model_ids))
    if unpriced or stray:
        raise RuntimeError(
            f"LIVE_PRICES and the live roster disagree: live rows without a price "
            f"{unpriced or 'none'}; priced ids that are not live rows {stray or 'none'}."
        )
