# Survey item text (local-only, assembled by hand)

`games/survey.py` reads two files from this directory, and neither is in version control:

- `published.json` — the item text, anchor labels and allocation payoffs of the seven published
  psychometric instruments the self-report battery administers. Read by
  `games.survey.load_published_instruments`.
- `authored.json` — the text of the items we wrote ourselves: the negative controls, the
  self-prediction stems (with their action-swapped counterparts), and the self-characterisation
  prompts. Read by `games.survey.load_authored_items`.

Everything *about* the items is tracked: `games/survey.py` holds each published instrument's item
count, subscale membership per item position, reverse-keyed positions, nested subscale definitions,
response-scale size, scoring formulas, citation and expected direction — and each authored item's
spec (`AUTHORED_ITEM_SPECS`: id, family, kind, tier, subscale, answer shape, expectation). Only the
strings and the payoff numbers live here.

## Why it is untracked, and the reason is licensing before it is size

The files are a few tens of kilobytes. `published.json` stays out of version control because this
repository's remote is public and six of the seven instruments arrived from copyrighted papers with
no redistribution grant: a publisher PDF, a paywalled primary with items transcribed from a
peer-reviewed supplement, a subscription journal, an author-distributed PDF with no stated licence,
and two payoff tables from open-access and APA papers respectively. Exactly one instrument (the
nine-item past-behaviour altruism scale) carries an affirmative CC BY grant, and it is withheld
along with the rest anyway: a per-instrument carve-out would put two loaders in the codebase and
invite the wrong one to be used, and a builder is not the right person to decide which paper's terms
permit republication.

`authored.json` stays out for the contamination reason alone (the owner's 2026-08-21 ruling): our
items are still items, they will be run on future models, and a committed item becomes scraper
training data that invalidates every future measurement made with it — copyright never enters into
it.

Contamination is a second reason with a smaller weight here than elsewhere in this repository. These
instruments are already published and almost certainly in every pretraining corpus, so keeping them
out is not protecting a benchmark the way `reward_hacking/harness/data/` is. It is still true that
committing the exact battery text plus its keying would make our administration of it trivially
trainable-on, which is not a thing to do for free.

The same convention as `reward_hacking/harness/data/`: the contents are ignored rather than the
directory, because git will not descend into an excluded directory and the negation for this README
would then be unreachable.

## Assembling `published.json`

There is no ETL, and deliberately so — every item was transcribed by hand from a retrieved primary
source, and a scraper against seven paywalled publishers would be both fragile and rude. The
transcriptions, their sources, their retrieval provenance and the per-item confidence notes live in
the session note `docs/scratch/survey-instrument-items-2026-08-18.md` (gitignored, so a fresh clone
does not have it either). Build `published.json` from that note.

The schema, with placeholder text standing in for the real items (the payoff triple below is
synthetic, invented for this example — it is not any published item's):

```json
{
  "schema_version": 1,
  "instruments": {
    "competitiveness-index": {
      "instructions": "<the instrument's own framing paragraph, if it prints one>",
      "anchors": ["<anchor 1>", "<anchor 2>", "<anchor 3>", "<anchor 4>", "<anchor 5>"],
      "items": [
        {"stem": "<item 1 verbatim>", "neutral_stems": ["<item 1, loaded words removed>"]},
        {"stem": "<item 2 verbatim>"}
      ]
    },
    "triple-dominance": {
      "instructions": "<the framing the nine choice situations are presented under>",
      "items": [
        {"option_payoffs": [[93, 23], [97, 41], [88, 88]]}
      ]
    }
  }
}
```

Rules the loader enforces, each of which fails loudly rather than shrinking the battery:

- `instructions` is optional on a Likert instrument, where it is prepended to every statement (the
  prosocialness scale prints a real one, about there being no right answers and giving your first
  reaction; administering the scale without it administers a different scale). It is **required** on
  the two allocation instruments, whose items are payoff tables with no text of their own: without a
  framing, each item would be sent as a bare list of number pairs, which parses perfectly and
  measures something else entirely.

- `items` must have exactly the length `games/survey.py` declares for that instrument. A partial
  instrument is not a shorter instrument: its subscale composite would be computed over whichever
  items happened to be present, and the tracked reverse-keying positions would point at the wrong
  items. The message names the expected count and the citation.
- `anchors` must have exactly the declared number of scale points, because that count is what a
  reverse-keyed answer is reflected through (`points + 1 - x`).
- Item order is the order the source paper prints, because every keying position and nested subscale
  in `games/survey.py` is a 1-based index into it. Re-sorting the items silently re-keys the scale.
- `option_payoffs` are `[self, other]` integer pairs in the instrument's own points. For the
  triple-dominance triples the loader recomputes which option maximises joint payoff, own payoff and
  the gap, and refuses an item whose three options do not resolve to one of each — so a truncated or
  hand-edited payoff row is caught rather than scored.
- `neutral_stems` is optional and holds our own lexically neutral rewordings of that item: same
  construct, same keying, with the loaded vocabulary removed. They become `-neutral01`-suffixed twin
  items and are what `games.survey.wording_gap` compares the published wording against. Ours to
  write, so they carry no third-party text, but they live here rather than in tracked code because a
  close paraphrase of a copyrighted item is a derivative of it.

## Assembling `authored.json`

The canonical text of every authored item — stems, the self-prediction family's action-swapped
stems, choice options, Likert anchors and the tag vocabulary — lives in the session note
`docs/scratch/survey-authored-items-2026-08-21.md` (gitignored too). The schema:

```json
{
  "schema_version": 1,
  "items": {
    "negative-control-indentation": {"stem": "<the question>", "options": ["<one>", "<two>"]},
    "self-prediction-twin-pd": {"stem": "<situation + question>", "stem_swapped": "<same, actions swapped>"},
    "self-characterisation-stance-conflict": {"stem": "<prompt>", "vocabulary": ["<w1>", "<w2>", "<w3>"]}
  }
}
```

Rules the loader enforces, mirroring the published side: every id in `AUTHORED_ITEM_SPECS` must be
present and nothing else may be; option and vocabulary counts must match the tracked spec exactly;
every self-prediction item must carry `stem_swapped` (the action-order counterbalance — its answers
are reflected through `100 - x` at parse time, so skipping it would score whichever action was
described first).

### Which keys each item kind needs

Five keys total, and an item's kind decides which of them it must carry. Nothing else is read, so a
key the loader does not know is silently ignored — put nothing here that is not in this table.

| Kind | Keys in this file | What stays in `games/survey.py` |
|---|---|---|
| `likert` | `stem`, `options` (the anchor ladder, in ladder order) | `reverse_keyed`, `subscale`, `n_options` |
| `choice` | `stem`, `options` | `option_labels` (what each option *means*), `n_options` |
| `ordered-choice` | `stem`, `options` (in ladder order, rung 1 first) | `subscale`, `n_options` |
| `allocation` | `option_payoffs` and an instrument-level `instructions` | everything else |
| `numeric` | `stem`, `stem_swapped` | `numeric_max`, `predicts_game` |
| `tagged` | `stem`, `vocabulary` | `n_tag_words` |
| `cheap-talk` | `stem`, `vocabulary` | `n_tag_words` |

Two of these need saying out loud because getting them wrong produces a plausible number rather than
an error:

- **`ordered-choice` option order is the scale.** The answer scores as its canonical position, so
  rung 1 must be the same end of the ladder in every item of a subscale (the safe or certain end, by
  convention). Re-sorting an item's options rescales it, exactly as re-sorting a published
  instrument's items re-keys it.
- **A `cheap-talk` item's one `vocabulary` is used for BOTH of its tags** — the announced intention
  and the action taken. That is what makes "announced x, did y" a comparison. The stem is what has to
  establish that the announcement is non-binding and that the counterpart sees it before choosing;
  the format instruction is generated and says nothing about either.

`option_labels` deliberately does **not** live in this file. It is a closed vocabulary of category
names (`joint-gain`, `own-gain`), not item text, and it is the datum the forced-choice families are
scored on — so it is tracked in `AUTHORED_ITEM_SPECS` where a re-analysis from a stored trace can
still read it. Same reasoning as `reverse_keyed`: the words are local, the scoring is not.

## What happens when the files are absent

Nothing raises at import, and **nothing runs**. `games.survey.survey_battery()` with no data
directory loads neither half, assembles zero items, and refuses to run an empty battery — the
fresh-clone story is "nothing runs without local item data", never a quietly smaller battery. The
tests build synthetic files under `tmp_path` in these schemas, so the suite stays green on a fresh
clone; the handful of tests that check the *real* local files skip loudly, saying why, when the
files are not on the machine.

Pointing the battery at a directory missing either file raises, naming the file and this README,
because that combination is an operator who meant to administer the battery and assembled half of
it.
