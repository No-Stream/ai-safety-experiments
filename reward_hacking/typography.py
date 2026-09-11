"""The typographic character folds every grader in this repository shares.

One table rather than one per benchmark, because each fold was earned by a grade it cost and a
second copy would have to lose the same grades again to learn the same lessons. Models routinely
emit U+2019 where a marker was written with the ASCII apostrophe, so ``doesn't`` would otherwise
never fire -- a grader that reports zero and reads like a finding. The first two real runs showed
the same failure at scale in three more shapes: GPT-OSS-120B emitted 2,780 narrow no-break spaces
and 1,685 non-breaking hyphens where GPT-5.6 Luna emitted ASCII. Counting over this repository's
stored traces, U+202F appears 11,002 times with 8,470 of those digit-adjacent and U+2011 7,391
times. The bias is therefore model-correlated, which is worse than noise: it moves a rate
differentially across the model axis, which is the axis every readout here compares along. So whole
families fold rather than the codepoints those two models happened to emit.

Escapes rather than the literal characters, because four of these are whitespace variants that no
reader could tell apart from a space in the source.

Only the table is shared. ``jagged``'s ``_normalise`` also lowercases and strips doubled emphasis
globally, and neither is safe for an answer value: lowercasing rewrites sympy symbol names, which
are case-sensitive, and a global ``**`` substitution turns ``2**3`` into ``23``. RecoveryBench
therefore folds with this table and peels emphasis only where it *surrounds* the whole value.
"""

from __future__ import annotations

TYPOGRAPHIC_FOLDS = str.maketrans(
    {
        "\u2019": "'",  # right single quotation mark, for doesn't / isn't
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
        "\u00a0": " ",  # no-break space
        "\u2009": " ",  # thin space
        "\u202f": " ",  # narrow no-break space
    }
)
