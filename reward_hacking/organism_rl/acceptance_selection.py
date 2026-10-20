"""Shared explicit selection of the acceptance-fixture treatment."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

ACCEPTANCE_MAX_TURNS = 20


def add_acceptance_arguments(parser: argparse.ArgumentParser) -> None:
    """Install the same private pack and availability controls in each consumer."""
    parser.add_argument(
        "--acceptance-assets", type=Path, help="prepared private acceptance asset index"
    )
    parser.add_argument("--acceptance-condition", choices=("open", "closed"), default="open")


def resolve_acceptance_options(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate the treatment before other consumers resolve their default turn budget."""
    options = vars(args)
    variants = {
        options.get(name) for name in ("variant", "impossible_variant", "evaluation_variant")
    }
    variants.update(variant for _family, variant in options.get("family_variant_entries", ()))
    selected = "acceptance_leak" in variants
    if not selected:
        if args.acceptance_assets is not None or args.acceptance_condition != "open":
            parser.error("acceptance assets and condition require acceptance_leak")
        if options.get("payment_register") == "trusted_acceptance":
            parser.error("trusted_acceptance payment requires acceptance_leak")
        return
    _validate_acceptance_selection(args, parser)
    args.max_turns = ACCEPTANCE_MAX_TURNS
    scoped = bool(options.get("family_variant_entries"))
    if "payment_register" in options and not scoped:
        args.payment_register = "trusted_acceptance"
    if "evaluation_variant" in options and not scoped:
        args.evaluation_variant = "acceptance_leak"
        args.impossible_variant = "acceptance_leak"


def _validate_acceptance_selection(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    options = vars(args)
    scoped = bool(options.get("family_variant_entries"))
    if scoped:
        acceptance_families = {
            family
            for family, variant in args.family_variant_entries
            if variant == "acceptance_leak"
        }
        if acceptance_families != {"aisi"} or "aisi" not in (args.family_entries or ()):
            parser.error("acceptance_leak requires an aisi family variant")
    elif list(dict.fromkeys(args.family_entries or ())) != ["aisi"]:
        parser.error("acceptance_leak requires --family aisi alone or a per-family variant")
    if args.acceptance_assets is None:
        parser.error("acceptance_leak requires --acceptance-assets")
    if args.max_turns not in (None, ACCEPTANCE_MAX_TURNS):
        parser.error("acceptance_leak requires a 20-turn budget")
    acceptance_fractions = [
        scope
        for scope, _fraction in options.get("in_band_entries", ()) or ()
        if scope.split("/", 1)[0] == "aisi"
    ]
    if (not scoped and options.get("payment_register") == "in_band") or acceptance_fractions:
        parser.error(
            "acceptance_leak requires trusted_acceptance payment without in-band fractions"
        )
    if "evaluation_variant" in options and args.evaluation_variant not in (None, "acceptance_leak"):
        parser.error("acceptance_leak requires its own concrete evaluation variant")
    if not scoped and options.get("solvable_fraction", 0) != 0:
        parser.error("acceptance_leak requires --solvable-fraction 0")
    if options.get("aisi_cases") is not None or options.get("aisi_hack_group") is not None:
        parser.error(
            "acceptance_leak uses its asset pack instead of AISI case or hack-group overrides"
        )
