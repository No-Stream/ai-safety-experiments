# SpreadsheetBench collector

The collector imports one pinned SpreadsheetBench source release, freezes explicitly selected task
groups and their actor-visible renders, records raw episodes, and supports separate assessment,
human review, and readout commands. It reads a host-only JSON config and sends generation requests
to an already-running vLLM HTTP endpoint. It does not start a model server or export episodes for
training.

Keep the config, source checkout, imported assets, server manifest, review files, and output directory
outside tracked files, using ignored paths when they live in this repository. All configured paths
must be absolute. Do not put task text or per-task output in tracked documentation.

## Commands

Set `COLLECTOR_CONFIG` to the absolute path of the private config file. Preparation imports the
source and writes private `selection-manifest.json`, `render-manifest.json`, and
`generation-identity.json` files. Before the first actor request, `run` freezes the effective mode,
settings, and per-task journal identities in `effective-run-manifest.json`. Use a separate output
directory for each smoke or research collection:

```bash
python -m reward_hacking.spec_gaming --help
python -m reward_hacking.spec_gaming prepare --config "$COLLECTOR_CONFIG"
```

Inspect both manifests and set the acceptance fields in the private config only after reviewing the
source terms, grouped-task obligations, and actor-visible render. The plumbing smoke uses at most
one selected task and four actor turns:

```bash
python -m reward_hacking.spec_gaming run --config "$COLLECTOR_CONFIG" --smoke
```

The research collection is capped at eight selected task groups and one sample per task. It requires
accepted source terms, task obligations, and render, a `research_only_quarantine` holdout disposition,
and the explicit `--research-only-approved` flag:

```bash
python -m reward_hacking.spec_gaming run \
  --config "$COLLECTOR_CONFIG" \
  --research-only-approved
```

Assessment can be rerun from the saved per-task identities without constructing a model client.
Supply human awareness and mechanism review as a private JSON file, then build the readout:

```bash
python -m reward_hacking.spec_gaming assess --config "$COLLECTOR_CONFIG"
python -m reward_hacking.spec_gaming review \
  --config "$COLLECTOR_CONFIG" \
  --task-id "$TASK_ID" \
  --review-file "$PRIVATE_REVIEW_JSON"
python -m reward_hacking.spec_gaming readout --config "$COLLECTOR_CONFIG"
```

`assess` and `review` refresh `readout.json`; `readout` can also be run independently. Assessment
exits nonzero when a proxy or full evaluation is unmeasured, while retaining any measured component.
A nonzero `run` exit means at least one selected task failed, while already persisted episodes remain
available for resume and assessment.

## Config schema

The parser rejects unknown fields and requires every field below. Values shown here are field names,
not example benchmark data or machine-specific paths.

| Object | Required fields and constraints |
| --- | --- |
| Root | `schema_version` (currently `1`), `source`, `output_dir`, `selection`, `acceptance`, `runtime`, `model`, `sampling`, `episode`, `assessment` |
| `source` | `root` (pinned source checkout), `archive` (pinned source archive), `asset_root` (private extraction directory) |
| `selection` | `candidate_task_ids` (explicit, unique IDs; at most eight), `samples_per_task` (must be `1`), `actor_case_indices` (exact task-ID-to-positive-case-index map), `proxy_cells` (exact task-ID-to-cell-list map; each list must be a reviewed, nonempty strict subset of that task's original required output cells) |
| `acceptance` | `reviewer`, `source_terms`, `holdout_disposition`, `task_obligations`, `render`; review fields are `accepted` or `pending`, and disposition is `research_only_quarantine` or `pending` |
| `runtime` | `image` (full SHA-256 image reference), `gate_record`, `gate_record_sha256`, `memory_bytes`, `pids_limit`, `cpus` |
| `model` | `endpoint` (HTTP(S) endpoint for an already-running vLLM server), `served_model_id`, `model_source`, `model_revision`, `tokenizer_source`, `tokenizer_revision`, `serving_manifest`, `serving_manifest_sha256`, `thinking` (boolean), `chat_template_kwargs` (string-to-string object) |
| `sampling` | `max_new_tokens`, `do_sample`, `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`, `stop`, `seed` (required nonnegative base seed; must equal `episode.seed`), `thinking_token_budget`, `stop_token_ids` |
| `episode` | `max_turns` (at most `20`), `max_tokens` (at most `8192`), `seed` (required nonnegative base seed; must equal `sampling.seed`), `max_context_tokens`, `samples_per_task` (must be `1`), `action_timeout_seconds`, `evaluation_timeout_seconds`; optional `enable_thinking` must equal `model.thinking` |
| `assessment` | `rubric_version` |

Model and tokenizer revisions are immutable 40-character commit IDs. The serving manifest is a
private JSON record binding the served ID, model/tokenizer identities, server version and arguments,
server config digest, and reasoning config. Configured token stop IDs must match the served Qwen
tokenizer. The model's thinking mode is copied into episode settings and their shared identity; if
`episode.enable_thinking` is supplied, it must match.

Manual review JSON contains `schema_version`, `raw_episode_digest`, `automatic_assessment_digest`,
`reviewer`, `rubric_version`, `integrity_awareness`,
`capability_evaluation_awareness`, `training_inference`, `designed_environment_inference`,
`classification`, `evidence`, and `limitations`. Integrity awareness is `clean`, `positive`, or
`uncertain`; each other awareness field is `yes`, `no`, or `uncertain`. Classification is one of
`confirmed_gaming`, `honest_success`, `ordinary_error`, or `ambiguous`. Every review is bound to the
raw episode and automatic assessment digests.

The readout reports the immutable final operational proxy separately from the latest independent
proxy and full assessments. It preserves component statuses and errors, awareness, manual
classification, mechanism evidence, and integrity exclusions as separate fields. Partial full
evaluation failures remain explicit while retaining a measured proxy. Open raw journals report their
current status without being treated as closed episodes. A proxy/full score gap records an outcome
only; it does not classify the actor's intent. The top-level `confirmed_gaming_evidence` contains
evidence statements only from reviews classified `confirmed_gaming` with clean integrity awareness.
It does not count evidence statements as distinct mechanisms or measure mechanism diversity.
