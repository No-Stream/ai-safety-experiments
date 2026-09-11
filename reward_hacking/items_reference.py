"""Resolve a ``module:attribute`` reference into a validated tuple of items.

One resolver rather than one per benchmark. The two copies this replaces were a machine-verified
near-duplicate -- 12 of 17 lines byte-identical, the five differences being the ``isinstance`` class
and three message strings -- and they had already drifted in a load-bearing way: one called its
corpus validator and the other did not, so two visually identical functions had different validation
contracts.

They also shared a latent defect that would otherwise have to be fixed twice. Pointing either at a
*generator*-valued corpus attribute -- the easy authoring slip ``ITEMS = (item_from_json(p) for p in
paths)``, which is exactly the "corpus module that reads its own JSON" the docstrings describe --
exhausted the iterator in the type check, left ``not items`` False on a live generator, and returned
an empty tuple. The one guard both functions existed to enforce, never bill a hosted model for an
empty corpus, was defeated identically in both. Materialising once up front is what fixes it in one
place.
"""

from __future__ import annotations

import importlib


def resolve_reference[ItemT](reference: str, item_type: type[ItemT]) -> tuple[ItemT, ...]:
    """Import ``module:attribute`` and return the items it names, refusing every way it goes wrong.

    Every failure is loud and specific, because the alternative is a sweep that bills a hosted model
    for a half-typed corpus. Materialised into a tuple before anything is checked, so a generator-
    valued attribute is counted rather than silently consumed.

    ``hasattr`` then ``getattr`` rather than ``getattr(module, attribute, None)``: the default form
    reports an attribute that exists but is None as "has no attribute", which sends a reader looking
    for a typo that is not there.
    """
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        msg = f"an items reference must read module:attribute, got {reference!r}"
        raise ValueError(msg)
    module = importlib.import_module(module_name)
    if not hasattr(module, attribute):
        msg = f"{module_name} has no attribute {attribute!r}"
        raise ValueError(msg)
    items = tuple(getattr(module, attribute))
    wrong_type = [item for item in items if not isinstance(item, item_type)]
    if wrong_type:
        msg = (
            f"{reference} holds {len(wrong_type)} entries that are not "
            f"{item_type.__name__} instances"
        )
        raise TypeError(msg)
    if not items:
        msg = f"{reference} is empty, so there is nothing to run"
        raise ValueError(msg)
    return items
