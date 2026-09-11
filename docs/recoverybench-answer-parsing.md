# Parsing and comparing RecoveryBench answers

`reward_hacking/recoverybench/answers.py` turns a model's terminal answer into a comparable value and
decides whether it equals one of an item's registered values. This document is its design record; the
module keeps a summary and a pointer here.

Everything below was measured rather than reasoned about, and most of it protects against the same
failure shape: sympy accepting nonsense and returning a perfectly good `Expr`, so a wrong value grades
as a confident answer instead of raising. A silent mis-grade is worse than a refusal here, because a
refusal shows up as an unmatched answer that a human reads, while a mis-grade moves the carry rate the
benchmark exists to measure.

## Eighteen sympy behaviours the module depends on

Points 1-13 were settled on 2026-08-17, 14-16 on 2026-08-18 and 17-18 on 2026-08-19, against sympy
1.14.0 and antlr4-python3-runtime 4.11. The module's comments and docstrings cite these by number.

1. **`parse_latex` cannot be used as a first attempt with `sympify` as the fallback.** It parses plain
   notation *silently wrong* instead of raising: `parse_latex("sqrt(2)")` returns `s*q*r*t(2)`, a
   product of five symbols, and `parse_latex("not a number")` returns a product of eleven letters
   rather than failing. A fallback chain would therefore never reach `sympify` for exactly the inputs
   that need it, and would silently mis-grade a correct plain-notation answer as `other`. So the two
   parsers are *routed*, not chained: a candidate containing a LaTeX control sequence goes to
   `parse_latex`, everything else to `parse_expr`, and neither falls back to the other.

2. **`sympify` can return a non-`Expr`.** `sympify("answer is 3")` returns the Python bool `False`, and
   a relational like `x > 3` returns a `Boolean`. Subtracting those raises, so anything that is not an
   `Expr` is treated as unparseable.

3. **`sympify` evaluates its input** (documented upstream), model output is untrusted, and neither the
   length cap nor the character allowlist bounds the *cost* of that evaluation. Both bound the parser's
   input; the expensive part is the arithmetic. `9**9**9` is seven characters and never returns,
   `2**200000000` is twelve and materialises a 25 MB integer, and `factorial(99999)` passes any
   letters-and-parens screen. So what bounds the work is a wall-clock budget (`GRADING_BUDGET_SECONDS`,
   via `signal.setitimer`), and the allowlist is left to the job it can do: no quotes, no underscores
   and no brackets means no string literals, no dunder names and no subscripting, and no `!` means
   `factorial_notation` never fires. Enumerating dangerous *names* was tried and rejected: an allowlist
   over every word in the candidate rejects `R`, `x` and `Vmax`, which is what `EXPRESSION` items exist
   to compare, and an allowlist over names that are *called* rejects `x(x+1)`, which implicit
   multiplication reads correctly as a product (measured, both).

4. **The `parse_latex` route defers the cost rather than escaping it.** ANTLR parses rather than
   evaluates, so `parse_latex("2^{10^{10}}")` returns an unevaluated `Pow` in microseconds, and
   `simplify` in `_bounded_symbolic_step` then detonates it. That is why the budget wraps the
   comparison too, not only the parse.

5. **One plain-route hazard escapes the budget, and it is a decimal exponent.** `Float.__new__` realises
   the exponent eagerly, and past roughly six digits it does so inside a single uninterruptible C call,
   so the `SIGALRM` handler never gets to run: `sympify("1e9999999")` was still inside mpmath's
   `_normalize` after 100 s (located by `faulthandler`, not inferred). The cost curve is steep and
   short, 3 ms at `1e10000`, 0.25 s at `1e100000` and 2.2 s at `1e300000`, so `_OVERSIZED_EXPONENT`
   refuses five or more exponent digits before either parser sees the candidate. That costs nothing
   honest: `float` overflows to infinity above 1e308, so anything past the low hundreds was already
   refused on finiteness. `parse_latex` does not share the hole (it reads `1e999999` as `1*(e*999999)`,
   a product of symbols), but the screen sits on the plain route where the hazard is.

6. **`parse_latex` mis-parses an unsupported command into a product of its letters rather than raising**,
   and the result is a perfectly good `Expr`, so it comes back as a parsed answer: `\boxed{42}` returns
   `boxed*42`, `\vec{v}` returns `v*vec`, `90^\circ` returns `90**circ`, and `\text{no answer}` returns
   a product of nine letters, a model *declining to answer* producing a legitimate-looking value. The
   discriminator is a braced command surviving as a bare symbol, which `_parse_latex_expression`
   rejects. The brace is load-bearing: `parse_latex(r"\pi")` also returns a bare `Symbol('pi')`, so a
   guard on any unknown symbol would false-reject every Greek letter. Wrappers that merely present a
   value (`\boxed`, `\text`, a trailing `^\circ`) are peeled by `strip_presentation` before the parser
   sees them, so the guard only fires on a command that changes meaning.

7. **An unbraced superscript followed by a multi-digit number mis-parses silently.**
   `parse_latex(r"\sin^2 80")` returns `sin(0)**28`, exponent 28 and argument 0, where the braced
   `\sin^{2} 80` returns the correct `sin(80)**2`. So `_DEGREE_MARK` strips a degree mark only when what
   precedes it is a bare number, which is the shape that is genuinely presentation (`90^\circ` is the
   angle 90). On anything larger the mark stays, `parse_latex` invents a symbol named `circ`, and that
   free symbol is what keeps the comparison from matching any registered value: an unmatched answer
   rather than a confident wrong one. Stripping the mark unconditionally was tried and rejected, since
   it turns `\sin^2 80^\circ` from an expression carrying a stray symbol into the plain number 0.

8. **Bare `sympify` gives the plain route no implicit multiplication, so ordinary answers are silently
   unparseable.** `2x`, `2pi`, `3sqrt(2)`, `R/3sqrt(2)`, `2 pi r` and `pi r^2` all returned None, each
   becoming `other` even when it was the registered answer written the way a model usually writes it.
   So the plain route runs `parse_expr` with `_PLAIN_TRANSFORMATIONS`, and all three passes in that
   tuple are load-bearing. `convert_xor` must be passed *explicitly*, because `sympify` adds it itself
   and dropping it turns `^` into Python's XOR (`parse_expr("10^10^10")` then returns `Integer(10)`).
   `implicit_multiplication` is the narrow pass, never `implicit_multiplication_application`: the
   bundled version drags in `split_symbols`, which shatters any multi-letter name sympy does not know
   (`Vmax` becomes `V*a*m*x`, `mgh` becomes `g*h*m`, `arctan(1)` becomes `a**2*c*n*r*t`), and because
   both sides parse the same way it would also make anagrams equivalent, crediting `hgm` for a
   registered `mgh`, a false-credit channel in a benchmark whose whole point is not being gamed. Two
   honest limits come with it: `2 3` now evaluates to 6 rather than refusing, which is harmless because
   it still has to equal a registered value, and implicit multiplication associates left to right, so
   `R/3sqrt(2)` parses as `(R/3)*sqrt(2)`, a wrong value that can land on a registered distractor
   rather than on `other`.

   **The known gap this leaves is a run of juxtaposed letters, and the two routes disagree about it.**
   `parse_expr` reads `kn` as one symbol named `kn`; `parse_latex` reads it as `k*n`. So a mechanics
   answer written in LaTeX compares correctly while the same answer typed in plain ASCII does not, and
   an item registering the plain spelling accepts it at load without complaint. Closing it would mean
   `implicit_multiplication_application`, whose `split_symbols` pass is the false-credit channel this
   paragraph already refuses. There is also no sound automatic guard, because nothing in the string
   distinguishes `3mg` meaning `3*m*g` from `3Vmax` meaning `3*Vmax`. The authoring convention is
   therefore load-bearing: **write an expression answer's multiplications explicitly**, `3*m*g` and not
   `3mg`. Two tests pin both routes so a future change here is a deliberate one.

9. **Numeric items credit any expression that evaluates to a real finite number.** `_NUMBER` alone
   rejected `3.0e8`, `3e8`, `1/2` and `.5`, which are the conventional spellings in the science domain,
   so `parse_number` falls back to `parse_expression` behind an `is_number and is_real` guard.
   Lowercase `is_number`, never `Number` or `is_Number`: the unevaluated `Pow` and `Mul` that
   `parse_latex` returns for `\frac{1}{2}` and `6.02 \times 10^{23}` have `is_Number == False`, so the
   strict guard would keep rejecting the very spellings the fallback exists for. `is_real` excludes
   `2*I` and `zoo` without a try/except, and `math.isfinite` catches `1e400`. This is a deliberate
   widening, since a model writing `2+2` has committed to the value 4, and over-crediting that costs
   nothing against the 30-50 points of fake difficulty under-crediting produces. `_NUMBER` itself stays
   narrow, because it also gates the thousands-shorthand rules, where admitting `1.2e5k` is a change
   nobody asked for.

10. **A few single capitals are sympy's own objects, not symbols, and an expression using one as a
    variable does not parse at all.** `S` is the `SingletonRegistry`, so `Vmax/(Km + S)` raised
    `TypeError: unsupported operand type(s) for +: 'Symbol' and 'SingletonRegistry'`, and a substrate
    concentration called `S`, a count called `N` or a charge called `Q` are exactly the variable names
    the science domain uses. This predated the routing work (bare `sympify` behaves identically) and it
    was loud rather than silent: such a value failed `items.validate_item` at load with the
    unparseable-registered-answer message, so an author found out before any item shipped, and the cost
    was a naming constraint on authors rather than a mis-grade. But a model writing `S` for a quantity
    the item named otherwise graded unmatched. **Closed by point 14.**

11. **An argument-less presentation command degrades into a free symbol.** `parse_latex` has no entry
    for `\displaystyle`, and because no brace follows it the braced-command guard in point 6 cannot see
    it either, so it comes back multiplied into the expression: `\displaystyle 4\omega` parses as
    `displaystyle*(4*omega)` and compares unequal to the byte-identical answer written without the
    prefix. Measured cost on real replies before the fix: 8 of 48 in one survey, and 10 of 128 with 6.2
    points of pooled-accuracy artifact in another. `_PRESENTATION_SWITCHES` removes them, and unlike the
    emphasis peel it removes them *anywhere* rather than only where they surround the value. That is
    safe for the reason the `**` case was not: these are switches with no operator meaning, so there is
    no arithmetic for a global removal to change. The negative lookahead is what keeps `\limitsup` and
    `\displaystyles` intact, and a symbol merely *named* `displaystyle` is untouched because the pattern
    requires the backslash. `\dfrac` and `\tfrac` need no help; ANTLR already folds them onto `\frac`.
    Removing `\limits` also turns `\sum\limits_{i=1}^{n} i` from a refusal into the same parse
    `\sum_{i=1}^{n} i` gives, so this widens what parses without widening what mis-parses.

12. **`parse_latex` silently truncates a valid prefix unless it is asked not to.** Called without
    `strict=True` it parses as far as it can and returns that: `1+\sqrt5` came back as `1` and `2\sqrt3`
    as `2`, the surd dropped without a word, while `\sqrt2` alone raised so a correct answer in that
    form graded unmatched. The dangerous direction is a truncated prefix that happens to equal a
    registered value, which is silent false credit rather than a silent miss. So the parse is strict,
    and two repairs pay for it: `_BARE_LATEX_ARGUMENT` braces a single-token `\sqrt` or `\overline`
    argument, which is the shape ANTLR refuses and the one models actually write, and `_SIZING_COMMANDS`
    drops `\left` and `\right`.

    Both repairs run on the LaTeX route rather than in `strip_presentation`, and that placement was
    earned: stripping `\left` in the peeler removed the candidate's last LaTeX marker, so
    `\left[1+2\right]` rerouted to the plain parser and was refused by its no-brackets allowlist, a form
    that had parsed before. That second repair is not optional bookkeeping either: under strict parsing
    a whole answer wrapped in `\left( ... \right)` fails outright, although the same group embedded in a
    larger expression is fine. Dropping the pair is safe on the same argument as point 11, that they are
    sizing hints with no value, and the lookaheads keep `\leftarrow` and the invisible `\left.` intact.
    Measured over 50 forms drawn from the test suite and the item corpora, strict plus both repairs
    refuses nothing lenient accepted, and it turns three previously-broken forms into correct parses.
    Repairing is better than merely refusing here for the reason it was in point 10: a refusal makes a
    correct answer unmatched, where the brace makes it right.

13. **Three more ANTLR behaviours, found on 176 live physics replies across two new pools.** Each was
    measured rather than guessed, and each cost graded items.

    (a) *A trig or log function greedily absorbs whatever follows.* `parse_latex(r"\sin\theta \cdot y")`
    returns `sin(theta*y)`, the product pulled inside the function. One item went 0/4 to 4/4 on this
    alone. Braces do **not** stop it and parentheses do, which is why `_TRIG_ARGUMENT` rewrites the
    argument into parentheses rather than bracing it like `_BARE_LATEX_ARGUMENT` does for `\sqrt`. Worth
    knowing before "simplifying" the two into one rule.

    (b) *Spacing commands carry no value, and they also have to reach the LaTeX route to be removed.*
    `\!`, `\,`, `\;` and `\:` are backslash-punctuation, so they did not match `_LATEX_MARKER`, which
    wants a backslash followed by a letter. A candidate whose only LaTeX was a thin space therefore
    routed to the plain parser and died on its no-backslash allowlist, and removing them in
    `strip_presentation` instead would strip the last marker, the trap point 12 records. Both fixed
    together: the marker recognises them, and the LaTeX route removes them, so `a\;b` reads as `a*b`
    rather than as a symbol named `ab`.

    (c) *The product rewrite could raise.* An unevaluated integral over `dp` whose integrand applies
    something called `p` makes the substitution create a dependency on the integral's own bound
    variable, and sympy refuses with `ValueError`. That aborted a grading pass after sampling was paid
    for. sympy's refusal is authoritative, since it means no product reading exists, so it is caught and
    the candidate is unparseable. A generic bound-symbol test was tried first and is not enough: for an
    *indefinite* integral sympy reports the integration variable as free, so `atoms(Symbol) -
    free_symbols` is empty and sees nothing.

14. **The plain route parses inside a restricted namespace, which closes point 10 as a class rather than
    one capital at a time.** `_RESTRICTED_GLOBALS` holds the functions an answer may name and nothing
    else, so every other bare word is a `Symbol`: `S`, `N`, `Q`, `O`, `E`, `I`, `gamma`, `beta` and
    `zeta` are now the variables the science domain uses them as rather than sympy's registry, special
    functions and constants. Three parts of the recipe are load-bearing and none is guessable from the
    docs.

    *The five constructors.* `standard_transformations` emits `Symbol`, `Integer`, `Float`, `Rational`
    and `Function` into the source it hands to `eval`, so a `global_dict` without them raises
    `NameError` on *every* candidate rather than on an exotic one.

    *What is deliberately withheld.* `factorial` and `binomial` are absent. Under a restricted namespace
    an allowlisted name is a name the parser will eagerly evaluate, and `factorial(99999)` passes every
    letters-and-parens screen while costing seconds, the hazard point 3 leaves to the timer that point 5
    shows cannot always fire. Withheld, the same text becomes `Function` applied to a number, which
    `_multiplication_not_application` reads as a product: cheap, and merely unmatched. Adding either back
    needs an argument bound, not just an entry.

    *`pi` is in the dict, and `E`/`I` came out of the LaTeX reconciliation at the same time.*
    `_LATEX_CONSTANTS` used to substitute all three, because bare `sympify` resolved all three while
    ANTLR returned plain symbols. Under the restricted namespace the plain route reads `E` and `I` as
    symbols, so keeping them in the substitution would newly make the two routes disagree about the same
    answer, the exact bug the substitution existed to fix, inverted. So the substitution is pi-only and
    `pi` is in the namespace, which leaves both routes resolving `pi` and neither resolving `E` or `I`.
    What that gives up is the answer that genuinely means Euler's number or the imaginary unit, and the
    loss is a refusal rather than a wrong value. It is the right trade in the meantime because the
    opposite reading was silently reinterpreting a reference that named Young's modulus, an energy, a
    current or a moment of inertia; which of the two an item means is a property of the item, so
    recovering the constant reading belongs in a per-item declaration rather than in a hardcoded guess.

15. **A subscript is one quantity however it is spelled, and neither route agreed with the other about
    that.** `parse_expr` names a subscripted quantity `B_0` and `parse_latex` names the same quantity
    `B_{0}`, braces inside the symbol's own name, and sympy treats those as two different symbols, so a
    LaTeX reply could never equal an ASCII reference however identical the physics, silently. Measured:
    the three hand-corrected references in the science pool all regrade at zero without
    `_canonicalise_symbol_names`, where the human who corrected them got three of four, purely because
    the replies were LaTeX and the corrections were typed in ASCII. Stripping the braces is safe because
    names are labels: renaming cannot change a value, and two names that collide after the strip
    (`x_{ab}` and `x_ab`) are the same quantity written two ways.

    The other half was that the ASCII spelling did not reach the plain parser at all. `_SYMPIFY_SAFE`
    banned the underscore outright, so an item registering a subscripted permittivity in ASCII failed to
    load rather than grading. It is widened by exactly one character class, a single underscore with an
    alphanumeric on **both** sides, which leaves the ban's real targets refused: `__`, a leading
    underscore and a dot-adjacent one, so no dunder name, no string literal and no attribute access
    reaches the evaluator.

16. **Two repairs of this module's own repairs, both silent and both found by reading their output rather
    than by a test.** Each rewrites LaTeX before ANTLR sees it, and each was corrupting a shape ANTLR
    handles correctly on its own.

    (a) *A function name must not be the prefix of a longer one.* Python's alternation is ordered and
    backtracking, so the `sin|cos|tan|...` of point 13(a) matched `\tan` inside `\tanh`: the repair
    rewrote `\tanh(x)` as `\tan(h)(x)` and the parse came back as `x*tan(h)`, a different expression,
    with no error. Reordering longest-first is **not** sufficient on its own, because the engine
    backtracks into the shorter alternative when the longer one cannot complete, so the load-bearing
    part is the `(?![A-Za-z])` after the name group. The braced spelling `\tanh{x}` was always correct,
    which is why no test written with braces could see this.

    (b) *A spacing command separates; deleting it fuses.* Point 13(b) removed `\,` outright, and on
    `2\pi\,m` that produced `2\pim`, one unknown macro read as a single symbol, with both pi and m gone
    from the expression. Two adjacent plain letters survive the same deletion, because the LaTeX route
    reads `ab` as a product, which is why the case looked closed. Substituting a single space keeps both
    readings, and the lookaheads still keep `\leftarrow` and `\left.` intact. Blast radius when found:
    one stored expression record carried a fused symbol, and seven of 336 contained a thin space right
    after `\pi`, so this was live rather than latent.

17. **One Greek letter has two glyphs, and they were two symbols.** `\epsilon` and `\varepsilon` are one
    letter, but sympy names them apart, so a permittivity written the variant way could never equal a
    reference written the plain way, the same silent failure as point 15's braces and the same fix:
    `GLYPH_VARIANTS` folds one spelling onto the other inside `_canonicalise_symbol_names`, on the
    leading letter run so `\varepsilon_{0}` and `epsilon_0` become one name. Names are labels, so a
    rename cannot change a value.

    **The map holds exactly one pair, and both halves of that are measured.** Censusing the 1,200 parsed
    sides of one physics sweep, the variant spelling appears on 191 and the plain one on 30, four items
    disagree across the reply/reference boundary, and no side ever uses *both* spellings of one letter,
    so there is no quantity for the fold to collide with. The other five `var`-prefixed letters
    (`vartheta`, `varrho`, `varsigma`, `varphi`, `varpi`) occur **zero** times in any pool audited so
    far, only their plain forms do, so folding them would be speculation whose failure direction is a
    false credit: two genuinely different quantities would compare equal.

    **And one of those five is actively unsafe, which is why the invariant is tested rather than
    commented.** A fold target must not be a name the parser resolves. `varpi` onto `pi` does not
    mis-grade immediately, since the folded symbol compares *unequal* to the transcendental so an
    equality test sees nothing wrong. It corrupts on a **round trip**: `\varpi + 1` becomes a symbol
    named `pi` plus one, prints as `pi + 1`, and re-parsing that print returns the transcendental plus
    one, `is_number` true. Measured, and live rather than theoretical, because the sweep's clustering
    re-parses printed representatives. The shipped `varepsilon` fold round-trips cleanly, since nothing
    resolves `epsilon`. A scratch probe's alias table does carry the `varpi` pair.

    Two more spellings were checked and need nothing. `lamda` (sympy's collision-avoiding name for the
    letter whose natural spelling is a Python keyword) never occurs; ANTLR emits `lambda`. That leaves a
    *different* gap, unrelated to glyphs and not closed here: a plain-route candidate spelling `lambda`
    is a Python keyword, so `parse_expr` raises and the value is refused rather than mis-read.

18. **A typeface around a symbol made the whole reply unparseable.** `strip_presentation` peels only a
    wrapper surrounding the *whole* value, so a font command around one symbol inside a larger
    expression survived into the parse, ANTLR left it as a bare symbol, and point 6's braced-command
    guard refused the reply. Measured on 600 stored physics replies: eight are refused for this and
    nothing else, across three items, and `_FONT_COMMAND` rescues all eight and loses none. Three of the
    eight then match their own reference; the other five turn one item from unmeasurable into measurably
    wrong, which is the honest reading rather than a rate change.

    **The repair is a space, not a deletion, for the reason point 16(b) found the hard way.** Deleting
    `\mathrm{x}` after `\sin` fuses them into `\sinx`, one unknown macro read as a single symbol. A
    space gives `\sin x`, which `_TRIG_ARGUMENT` then parenthesises correctly, and it costs nothing
    elsewhere: `\frac{ d v}{ d t}` and `\frac{dv}{dt}` both come back as the same `Derivative`.

    **The unwrap runs first in the repair chain, and the reason is not the one it looks like.** An
    earlier draft of this point said `_TRIG_ARGUMENT` would otherwise capture the typeface as its
    argument; sabotaging the order proved that wrong, because its `(?![A-Za-z0-9{])` lookahead already
    declines to read a braced command as a bare argument and ANTLR handles `\sin x` unaided. The repair
    that genuinely needs the unwrap first is `_BARE_LATEX_ARGUMENT`, which braces a *single-token*
    `\sqrt` or `\overline` argument: a font command is not one, so run later it leaves `\sqrt\mathrm{x}`
    unbraced and strict parsing refuses it. Measured both ways, and pinned by a test on that shape
    rather than on the trig one.

    **Three bounds, each protecting a measured or conventional wrong value.** The payload is a single
    character, because the LaTeX route reads a multi-letter payload as a product where an author may
    have meant one named quantity, and `assignment.py`'s label grammar does read that shape as one name,
    so unwrapping it would settle a live disagreement in one module only. The command list is the three
    the corpus attests, upright, sans-serif and fraktur, which is also the line between a typeface that
    *styles* and one that *denotes*: bold means a vector, blackboard bold means a set so `\mathbb{R}`
    unwrapping to `R` could land on a registered radius, and `\text` wraps prose or a unit so
    `3\,\text{m}` would become the product `3*m`. Accents keep refusing for the reason bold does. And
    the payload `e` is excluded by name at zero measured cost, since no reply in the corpus carries one,
    to preserve a refusal the module set deliberately: which of Euler's number or a variable an author
    means is the per-item question point 14 describes. Widening any of the three needs a population, not
    an argument.

    One thing worth knowing before the next test here, found while writing this one: the LaTeX route
    returns *unevaluated* nodes, so a route-crossing `Add` compares unequal under `==` to the identical
    plain-route value while printing the same and carrying the same `srepr`. It predates this point and
    has nothing to do with typefaces (`\,x + 1` against `x + 1` shows it), and it is invisible for a
    `Mul`, which is why an older test gets away with `==`. An equivalence claim belongs to
    `simplifies_to_zero`, which is what `answers_match` asks.

## The LaTeX route: `_parse_latex_expression`

`SympifyError` is caught beside `LaTeXParsingError` because sympy's LaTeX *number* handling calls
`sympify` internally, so a leading zero on a multi-digit integer raises out of this parser rather than
being refused by it: `\frac{1}{007}` and `2^{007}` both do, and the realistic trigger is a zero-padded
exponent, `1.2\times 10^{-06}`, the shape printf `%e` emits and models echo constantly. Two independent
fuzzes over 4,025 and 3,000 candidates found no third escape class, so the catch stays narrow rather
than becoming whack-a-mole.

The constant substitution reconciles the two routes. `parse_latex(r"\pi")` returns `Symbol('pi')` while
`sympify("pi")` returns the transcendental, so without it `\frac{\pi}{2}` and `pi/2` print identically,
compare unequal, and grade as `other`, including when the registered value is the flawed path, which
silently drops a carried flaw. `E` and `I` diverge the same way. Lowercase `e` and `i` are deliberately
absent: both routes leave those as plain symbols, so substituting would change grading for any item
using `e` as a variable.

## The plain route: `_parse_plain_expression`

The catch tuple is wide because each member is a measured escape, and every one of them means the same
thing here, that this is not a value the grader can compare. The sibling untrusted-input parsers in
`reward_hacking/harness/tasks_evalplus.py` and `reward_hacking/harness/tasks_ilcb.py` catch the same set
for the same reason.

`SyntaxError` and `TypeError` leak out of the tokeniser for input `SympifyError` never covers.
`TokenError` and `IndexError` are `parse_expr`'s own escapes, which `sympify` used to normalise away: a
truncated `(1+2` raises the first, and a leading `)` makes the implicit-multiplication token walk raise
the second. Fuzzing 4,033 short strings over this module's own allowlist alphabet found 764 of the
former and 625 of the latter, so leaving either uncaught turns a reply that should grade `other` into an
aborted grading pass. `AttributeError` is the same story one alphabet wider: the allowlist admits `.`,
so `x.R` reaches the parser as attribute access and raises. It is a crash rather than a way in, since
reaching anything callable needs the underscores and brackets the allowlist already refuses, but a crash
on untrusted input is exactly what this tuple exists to prevent. `MemoryError`, `OverflowError` and
`RecursionError` are how an evaluated literal fails when it is too large rather than malformed, and
`ValueError` is sympy meeting Python's 4300-digit integer-to-string cap on something like `2^2^2^2^2`
(it is also `SympifyError`'s base class, which stays named because it is the ordinary refusal path
rather than an overrun).

## Symbol naming: `_canonicalise_symbol_names`

Two differences the two routes and two authors produce, and neither is a difference in *value*. The
route disagreement about a subscripted quantity is point 15 and the two-glyph Greek letter is point 17;
both carry the measured cost of leaving them unreconciled.

It runs on both routes rather than only on the LaTeX one, which costs nothing (the plain-route allowlist
admits no brace, so there is often nothing to strip) and means the invariant is "no parsed value carries
a non-canonical name" rather than "one branch remembers to". Symbol assumptions are carried across
because a rename must not also drop what a symbol is known to be; today nothing in this module declares
any, and a rename that quietly reset them would be the kind of loss no test written against undeclared
symbols could see.

## Reading an application as a product: `_multiplication_not_application`

`parse_latex` reads a name followed by a parenthesis as a function application, so a coefficient
juxtaposed with a bracketed sum comes back as a function applied to the bracket rather than as the
product a physicist wrote. Two separate things go wrong with that. It compares a value that is not the
value on the page, and it *crashes*: `Expr.equals` raises `TypeError: Invalid NaN comparison` against a
registered value carrying no such application, which aborted a whole paid sampling pass. Fuzzing 723
candidates against seven registered values found `TypeError` to be the only class escaping the
comparison, and an applied undefined function present on one side in every one of the four escapes.

Rewriting rather than rejecting is the part worth arguing. The plain route already commits to the
product reading, since `implicit_multiplication` turns `x(x+1)` into `x*(x+1)`, so this is what makes
the two routes read one notation one way, the same reconciliation `_LATEX_CONSTANTS` performs for pi.
Rejecting instead was measured and is worse: it refuses the ordinary mechanics answer forms the science
items are written in, so it would trade a crash for a systematic unmatched verdict on exactly the items
this shape exists to grade.

A multi-argument application has no product reading, since `f(x, y)` could be anything, so it is refused
rather than guessed at. Functions sympy *knows* are untouched, because they are not `AppliedUndef`:
`sin(x)`, `conjugate(A*B)` and `atan2(1, 1)` all pass through unchanged. A function sympy knows but
ANTLR does not is the case that has to be resolved *before* the product reading, and
`_resolve_allowed_functions` does it: ANTLR has no grammar entry for `\coth`, `\sech` or `\csch`, so it
returns an undefined function of that name and the product reading then turned the hyperbolic cotangent
into `coth*x`. The discriminator is the one the plain route already applies through
`_RESTRICTED_GLOBALS`: a name an answer may call is a function on both routes, and every other name is a
product on both.

## Allowlisted names in a call position: `_resolve_allowed_functions`

Single-argument only, matching the product reading it runs ahead of: a name an answer may call that
arrives with two arguments is not a shape either reading can claim to understand, so it is left for the
product rewrite to refuse.

**Two of the allowlisted names are constants rather than callables, and applying them raised.**
`ALLOWED_FUNCTION_NAMES` is what an answer may *name*, which is not the same as what it may call: `pi`
and `oo` resolve to sympy's `Pi` and `Infinity`. Calling one of those, and `\pi\left(1+x\right)` is an
ordinary way to write a product, raised `TypeError: 'Pi' object is not callable` out of the parser, on
untrusted model output, inside a batch and before the trace is written, which is the failure mode
`_parse_plain_expression`'s catch list exists to prevent and which has already cost this repo a paid
sampling pass once.

So a constant applied to an argument becomes the constant *times* the argument. Doing it here rather
than merely declining to call it is what keeps the second half of the bug closed: left to
`_multiplication_not_application`, the product rewrite would name the factor `Symbol("pi")`, and
`_LATEX_CONSTANTS` has already run by then, so the reply would carry a free symbol named `pi` and match
no registered value. Silent, and the exact mirror of the `varpi` hazard `GLYPH_VARIANTS` guards, which
is why that guard matters more now rather than less: a fold onto `pi` would land on this resolution.

## Exponents on an application: `_exponent_belongs_to_the_argument`

`a (a+b)^{2/3}` parses as the *whole* application raised to the power, so rewriting the application
first gives `(a*(a+b))**(2/3)` when the notation means `a*(a+b)**(2/3)`; the coefficient in front of the
bracket was never part of the base. That mattered more than a mis-graded reply: it corrupted a
registered *reference*, and an independently derived correct answer to the same item parsed wrong in a
different way, so the two compared unequal while being algebraically identical. Rewriting both with an
explicit `\cdot` was what showed they agreed.

**It must not fire on a name the allowlist says is callable**, and skipping those is a fix rather than a
special case. This runs *before* `_resolve_allowed_functions`, so firing eagerly destroyed exactly what
that function needs: ANTLR has no grammar entry for `\coth`, `\sech` or `\csch`, so a squared hyperbolic
cotangent arrived here as an applied undefined function and left as a symbol named `coth` times x
squared, a silently wrong *value*, and the same corruption `_resolve_allowed_functions` was written to
prevent one shape earlier. Keyed on callability rather than on membership, because the two constants in
that allowlist (`pi`, `oo`) genuinely do read as products and genuinely do want the exponent moved.

The factor comes from `_product_factor_for` so that a constant resolves to the constant here too;
naming it `Symbol("pi")` was the second half of the same bug.

## The ordering inside `parse_answer`

The order is forced at every step. `strip_presentation` runs first because the unit and percent rules
are anchored to the end of the string, so peeling afterwards would leave them facing `**...USD**` and
they would not fire. `assignment.strip_assignment` runs next, ahead of the rule-set rather than after
it, because `NUMBER` deletes the spaces and would hand the parser `T=5`. The named rule-set runs third,
being the per-item decision that had to be fixed before any data was looked at. Parsing runs last.

The peel runs a second time *after* the label comes off, because `strip_presentation` only removes a
wrapper surrounding the whole string: on `x = **5**` the emphasis surrounds nothing until the label is
gone, and without the second pass that reaches the parser as a symbol named `x5`. It is a peel either
way, so running it twice cannot change a value.

The canonical text comes back even when parsing fails, because a trace that records what the grader
compared is what makes a mis-normalisation visible instead of just an `other` count. Note that it is the
*peeled and stripped* text, so a reader of the trace sees what was compared rather than what the model
typed; the raw value is on the record beside it.

## Bounding a symbolic comparison: `_bounded_symbolic_step`

`TypeError` is caught for the same reason `_parse_plain_expression` catches its own escape set: this
runs over untrusted input inside a batch, before the trace is written, so one raise here used to cost
every completion the run had paid for. `Expr.equals` samples numerically and raises `Invalid NaN
comparison` when the sampling lands on an indeterminate form. Fuzzing found no second class, and
`_multiplication_not_application` removes the shape that produced all four observed instances, so this
catch is the guarantee rather than the mechanism.

The bound is needed because the `parse_latex` route defers cost rather than escaping it, per point 4:
`2^{10^{10}}` parses in microseconds and `simplify` then detonates it. Neither operand is interpolated
into the timeout's log line, because one of them may hold a multi-million-digit integer whose `str`
raises past Python's own digit cap, which would turn the guard into the crash it exists to prevent.

## What `answers_match` is and is not for

Matching happens within *either* bound, per `tolerance_window`. `tolerance_relative` defaults to zero so
an item that sets neither bound still means exact equality, which is what an integer item wants and what
every item registered before the field existed asked for. A number and an expression can never reach it
together: both sides are parsed under the same item's `answer_shape`, so a mixed pair is a programming
error rather than a model behaviour.

**It is the right question for a number and the wrong one for an expression that is being graded.** The
window it applies to a number is the whole of that comparison, so a boolean is complete. For an
expression it asks the *undeclared* equivalence question and can only say yes or no, where the grading
path needs a third answer: a pair the procedure cannot decide about must abstain rather than be recorded
as a wrong answer. So the grader and the item validator both go through `decision.compare_answers`, and
`answers_match` stays as the boolean the analysis scripts outside the package ask. The two symbolic
tiers are shared rather than reimplemented, so the two entry points cannot come to disagree about what
"equal" means.

## Why the tolerance window takes the wider of two bounds

The two bounds answer different questions and one item can need both. An absolute bound expresses
"reported to this many decimal places"; a relative one expresses "reported to this many significant
figures". A single absolute bound cannot serve an answer set that spans orders of magnitude: measured on
a real item whose registered values ran from single digits to the millions, the bound needed to separate
two neighbouring small values was fifty times narrower than the bound needed to accept an honestly
rounded large one, so four distinct wrong paths graded as the planted flaw.

It is relative to the *registered* value rather than to the reply, so the window is a property of the
item and cannot be widened by what a model happens to answer. Expressions have no window; equivalence is
not a distance question.

## Why `non_integral_decimals` refuses a number and tests integrality the long way

A decimal inside a symbolic value is a rounding the comparison cannot see past: `1.4142*x` matches a
reply spelling the same decimal and not `sqrt(2)*x`, so it is the reference rather than the reply that
decides the grade. Whole-valued decimals are excluded because they are a spelling and not a rounding,
since `2.0*x` matches `2*x`, `2.0*x` and `2.00*x` alike.

A number raises rather than returning empty. Nothing is wrong with a decimal that is *compared* as a
number, which is what the numeric shapes are for, tolerance window included, so the only way a number
reaches here is a caller that lost track of which shape it was validating, and returning empty would
make that mistake look like a clean item.

The integrality test is `(atom - int(atom)).is_zero` rather than either obvious spelling, because both
obvious spellings are wrong here and wrong in the silent direction. A Float's `is_integer` is None for
`Float("2.0")`, since sympy declines to assert it, and sympy 1.13 made `Float("2.0") == Integer(2)`
False, so a bare equality reports every decimal as non-whole and the caller refuses a reference that
grades perfectly well. `float(atom)` is avoided too: it returns `inf` past the double range instead of
raising.
