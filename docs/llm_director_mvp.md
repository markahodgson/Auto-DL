# LLM Director + Narrative Input (MVP)

## Why this exists
Dataset semantics are often ambiguous without narrative context (for example, `sex` vs `gender`, `0/1` vs `M/F`).
This MVP adds a user narrative channel and lets the LLM produce a constrained planning output.

## Decision tree (MVP)
1. Build deterministic profile (`profile_dataframe`).
2. Build `run_context` from profile + config + user narrative.
3. Run lightweight deterministic rules first:
   - detect likely IDs and leakage candidates
   - detect common boolean/gender token variants
   - infer likely task from target cardinality
4. Call LLM Director only for ambiguous decisions.
5. Validate director output against schema.
6. If low confidence, request user confirmation.
7. Execute deterministic preprocessing from plan and log decisions.

## Narrative input contract (first pass)
Use one free-text narrative and optional column-level hints:

```yaml
narrative:
  dataset_summary: "Clinical records for heart attack risk prediction."
  target_definition: "Target column is output; 1 means high risk, 0 means low risk."
  business_goal: "Maximize recall for high-risk patients while keeping false positives manageable."
  leakage_warnings:
    - "Do not use discharge_outcome because it is post-event."
  column_hints:
    sex: "Biological sex encoded as 0/1 in this dataset."
    cp: "Chest pain type is ordinal severity, not nominal."
```

## Where to integrate in current code
- Add a `NarrativeConfig` section in `src/autodl/config.py`.
- Add optional CLI args to `preprocess`:
  - `--narrative-file <path>` (YAML/JSON)
  - or `--narrative "..."` (quick text)
- Build prompt payload in a new module (suggested: `src/autodl/policy/director.py`).
- Use `autodl.llm.factory.make_llm_provider` to call the configured provider.
- Validate and persist artifacts:
  - `runs/<run_id>/director_plan.json`
  - `runs/<run_id>/decision_log.jsonl`

## Confidence handling
- `confidence >= 0.80`: auto-apply plan.
- `0.50 <= confidence < 0.80`: apply deterministic defaults + log warning.
- `confidence < 0.50`: require user confirmation via CLI prompt or pre-supplied override.

## Minimal implementation sequence
1. Add config + CLI wiring for narrative input.
2. Add director prompt + schema validation.
3. Add plan-to-preprocess translation layer.
4. Add run artifacts and report section summarizing narrative-driven decisions.
