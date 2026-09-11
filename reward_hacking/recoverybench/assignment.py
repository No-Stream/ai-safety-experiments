r"""The leading-assignment peel, and the quantity-name grammar that decides when it may fire.

A model asked for a value often writes an *equation*: ``T_1 = 2\pi\sqrt{L/g}`` rather than the
right side alone. Neither parser can read that. ``parse_latex`` returns a sympy ``Equality``, which
is a ``Boolean`` and not an ``Expr``, and ``parse_expr`` raises ``SyntaxError``, so
:func:`~reward_hacking.recoverybench.answers.parse_expression` refuses the whole string and the
reply grades as an unparseable answer rather than as the value it states. Measured cost of not
peeling: 20 of 456 replies on one pool (4.4%), concentrated on expression items, where naming the
quantity before stating it is ordinary prose. It bites the numeric shapes too, because the
``NUMBER`` rule-set deletes the spaces and hands the parser ``T=5``.

Its own module rather than another section of ``answers.py``, which is already the largest file in
the package: what is in here is a grammar for *names*, asked of the text to the left of an equals
sign, and it shares nothing with the value parsing and equivalence that module is about. Its one
caller is :func:`~reward_hacking.recoverybench.answers.parse_answer`, which both the reply and every
registered value go through, and that is what makes the rule apply to the two sides alike.

**The refusals are the load-bearing half, and the failure direction is why.** A stripped label
recovers a correct answer that was being thrown away, which is a visible gain. An *equation*
truncated to its right-hand side becomes a value the model never claimed, and it can land on a
registered distractor, so the grade is a confident false credit in a benchmark whose entire premise
is that it cannot be gamed. Three guards live in here, each watched to fail with the specific wrong
implementation it forbids:

1. **Exactly one equals sign.** The obvious way to write this peel reads everything after the
   *last* ``=`` as the value, which turns the chain ``a = b = 2`` into ``2``. sympy makes that
   reading look plausible: ``parse_latex("a = b = 2")`` returns ``BooleanFalse``, so the chain is
   refused by the parser either way and only the peel can turn it into a credited value.
2. **The left side must MATCH a quantity name**, rather than merely fail to contain an operator.
   The guard this replaces was a five-character ban on ``[+-*/^]``, which is the wrong alphabet for
   a corpus that is 90.4% LaTeX: LaTeX writes arithmetic as control sequences, so ``x \cdot y = 3``
   stripped to ``3``, ``\frac{dv}{dt} = -k v`` to ``-k v``, ``\sqrt{x} = 4`` to ``4`` and
   ``E \times B = 0`` to ``0``. Every one of those is an equation whose right side is not its value.
   Inverting the rule closes the class instead of the five characters someone thought of.
3. **At most two name atoms**, so ``\Delta E`` and ``d N`` read as one quantity's name while
   ``x y z`` does not.

Two more findings came from auditing every left-hand side a real corpus contains, rather than from
reasoning about the grammar. A grammar can be too strict as easily as too loose, and too strict
costs a correct answer *silently*, so the population is the only honest check.

* **A subscript is often itself a word in an upright font.** ``\rho_{\mathrm{crit}}`` and
  ``C_{\text{crit}}`` are quantity names, and refusing them cost the strip on ten stored replies.
  Hence the nested-command alternative inside :data:`_QUANTITY_ATOM`'s braced subscript.
* **A name may be wrapped in a font or accent command.** ``\mathrm{KE}`` and ``\vec{p}`` name
  quantities; :data:`_NAME_WRAPPER` peels those to a fixed point before the grammar runs.

Two shapes need no case of their own and get none. A relational operator leaves its own symbol on
the left, so ``x <= 3`` offers ``x <`` and ``x != 3`` offers ``x !``, neither of which is a name,
and ``x == 3`` carries two equals signs and dies at the first guard. A label carrying a decimal
point is refused too, in both the plain and the subscript alphabet, which is what lets
``decision.sampled_tolerance`` read a reference's authored precision off the registered text without
peeling the label itself: no label this grammar accepts can contribute a decimal literal.

**What the peel gives up, stated rather than discovered later.** An item whose answer genuinely *is*
an equation with a single-symbol left side -- ``y = 2x`` as the equation of a line, not as a
labelled value -- loses the label on both sides, so a reply naming a different quantity is credited.
Before the peel such an item was refused at load, because ``parse_expression`` would not read the
``Eq``; now it loads with the left side discarded, trading a loud authoring failure for a quiet loss
of discrimination. Two things bound the cost. No ``AnswerShape`` member means "equation", so an
equation-valued item is outside the grading contract either way, and an equation between two
*expressions* (``x + y = 3``) is still refused at load by the guards above. Which of the two an item
means is a property of the item, so recovering the equation reading belongs in a per-item
declaration rather than in a hardcoded guess here -- the same argument ``answers.py`` makes for the
answer that genuinely means Euler's number.
"""

from __future__ import annotations

import re

# A cheap pre-screen ahead of the grammar; anything longer than a subscripted name is an expression.
MAX_LEFT_HAND_SIDE_CHARS = 40

_ASSIGNMENT = re.compile(rf"^(?P<lhs>[^=]{{1,{MAX_LEFT_HAND_SIDE_CHARS}}})=(?P<rhs>.+)$", re.DOTALL)

# Spacing inside a name carries no value, so ``\Delta\,E`` must read as ``\Delta E``.
_LATEX_SPACING = re.compile(r"\\[,;:!>]|\\q?quad|~")

# A font or accent command wrapping a whole name. Peeled to a fixed point before the grammar runs.
_NAME_WRAPPER = re.compile(
    r"^\\(?:mathrm|mathbf|mathcal|mathbb|mathit|text|textrm|boldsymbol|vec|hat|bar|tilde|dot|ddot"
    r"|overline|underline|operatorname)\s*\{(?P<inner>.*)\}$",
    re.DOTALL,
)

# One name atom, whose braced subscript may itself be an upright word; see the audit note above.
_QUANTITY_ATOM = re.compile(
    r"(?:\\(?P<command>[A-Za-z]+)|(?P<plain>[A-Za-z][A-Za-z0-9]{0,9}))"
    r"(?:_(?:\{(?P<braced>"
    r"(?:\\(?:mathrm|text|textrm|rm|mathbf|mathit|mathcal)\s*\{[A-Za-z0-9,\s]{1,12}\})"
    r"|[A-Za-z0-9,\s]{1,12}"
    r")\}|(?P<bare>[A-Za-z0-9])))?"
)

# The control sequences that NAME a quantity; every other one means the equation is the answer.
_QUANTITY_COMMANDS = frozenset(
    [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "varepsilon",
        "zeta",
        "eta",
        "theta",
        "vartheta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "varpi",
        "rho",
        "varrho",
        "sigma",
        "varsigma",
        "tau",
        "upsilon",
        "phi",
        "varphi",
        "chi",
        "psi",
        "omega",
        "Gamma",
        "Delta",
        "Theta",
        "Lambda",
        "Xi",
        "Pi",
        "Sigma",
        "Upsilon",
        "Phi",
        "Psi",
        "Omega",
        "ell",
        "hbar",
        "imath",
        "jmath",
        "aleph",
        "nabla",
    ]
)

# At most this many name atoms, so ``\Delta E`` is one quantity's name and ``x y z`` is not.
_MAXIMUM_LEFT_HAND_SIDE_ATOMS = 2


def strip_assignment(text: str) -> tuple[str, bool]:
    r"""Strip a leading ``label =``, and say whether it fired.

    Returns the text unchanged with ``False`` whenever a guard refuses, so a caller that wants to
    record the peel gets the fact rather than having to re-derive it: a trace reader asking why the
    compared value differs from what the model typed is asking exactly this.

    Neither side is checked for emptiness, and that is measured rather than assumed -- the design
    this came from checked both. :data:`_ASSIGNMENT` runs against ``text.strip()``, so the tail can
    never be blank: an all-whitespace right side leaves nothing for ``.+`` to match and the pattern
    refuses, while an all-whitespace left side reaches ``atoms == []`` in
    :func:`_is_bare_quantity_name`. Brute-forcing 1,508,597 candidates over an alphabet of equals
    signs and six kinds of Unicode whitespace found neither check changing the verdict once, so
    neither survived into this module.
    """
    if text.count("=") != 1:
        return text, False
    match = _ASSIGNMENT.match(text.strip())
    if match is None:
        return text, False
    if not _is_bare_quantity_name(match.group("lhs")):
        return text, False
    return match.group("rhs").strip(), True


def _is_bare_quantity_name(lhs: str) -> bool:
    r"""Whether this left-hand side is the NAME of a quantity, in ASCII or in LaTeX.

    Accepts ``T``, ``v_f``, ``E_{b}``, ``k_B``, ``\omega_0``, ``\Delta E``, ``\mathrm{KE}``,
    ``\vec{p}``, ``\mathrm{\vec{p}}`` and ``\rho_{\mathrm{crit}}``. Refuses anything carrying an
    operator, a function, a fraction, a derivative, a delimiter or an exponent, including the LaTeX
    spellings a character-class guard admits.

    The wrapper peel is unguarded on purpose, which is a correction to the design it came from.
    :data:`_NAME_WRAPPER` matches to the *last* brace, so ``\vec{a}+\vec{b}`` peels to the
    unbalanced ``a}+\vec{b``, and that is why a brace-depth check looks as necessary here as it
    genuinely is in ``strip_presentation``. It is not: :data:`_QUANTITY_ATOM` admits a brace only
    inside a well-formed subscript, so any leftover brace fails the atom match anyway. Brute-forcing
    830,940 left-hand sides over a brace-heavy alphabet found the depth check changing the verdict
    on **zero** of them, so it was removed rather than kept as reassurance. Termination never
    depended on it: each peel removes ``\command{`` and its closing brace, so the string strictly
    shortens.
    """
    candidate = _LATEX_SPACING.sub(" ", lhs).strip()
    while (wrapped := _NAME_WRAPPER.match(candidate)) is not None:
        candidate = wrapped.group("inner").strip()
    atoms = candidate.split()
    if not atoms or len(atoms) > _MAXIMUM_LEFT_HAND_SIDE_ATOMS:
        return False
    for atom in atoms:
        found = _QUANTITY_ATOM.fullmatch(atom)
        if found is None:
            return False
        command = found.group("command")
        if command is not None and command not in _QUANTITY_COMMANDS:
            return False
    return True
