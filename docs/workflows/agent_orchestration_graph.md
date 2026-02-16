# Agent Orchestration Graph (v1)

This document defines the unified LLM-agent flow for AutoDL, including node contracts, control gates, and run-monitoring artifacts.

## Graph (Mermaid)

```mermaid
flowchart TD
  A0([N00 Trigger Run]) --> A1[N01 Load Config + Runtime Context]
  A1 --> A2[N02 Ingest CSV]
  A2 --> A3[N03 Resolve Target + Validate Schema]
  A3 --> A4[N04 Profile Dataset]
  A4 --> A5[N05 Merge Narrative Inputs]

  A5 --> A6[N06 Deterministic Preprocess Plan]
  A6 --> G1{N07 Plan Confidence >= Threshold?}
  G1 -- Yes --> A9[N09 Apply Plan]
  G1 -- No --> A7[N08 LLM Director Refinement]
  A7 --> A8{N08b Valid JSON Plan?}
  A8 -- Yes --> A9
  A8 -- No --> A6

  A9 --> A10[N10 Preprocess + Feature Build]
  A10 --> A11[N11 Persist Preprocess Artifacts]

  A11 --> A12[N12 Build Train Policy (Deterministic)]
  A12 --> G2{N13 Policy Confidence >= Threshold?}
  G2 -- Yes --> A15[N15 Start Training]
  G2 -- No --> A13[N14 LLM Train Policy Refinement]
  A13 --> A14{N14b Valid Policy JSON?}
  A14 -- Yes --> A15
  A14 -- No --> A12

  A15 --> A16[N16 Stage-1 Optuna Tuning]
  A16 --> A17[N17 Finalist Retrain + Evaluate]
  A17 --> A18[N18 Persist Training Artifacts + REPORT]
  A18 --> A19[N19 Agent Monitoring + Status + Next Actions]

  M1[[Monitor: run_manifest / preprocess_metadata / director_plan / decision_log]] -.-> A19
  M2[[Monitor: training_summary / REPORT / tracking_local_metrics]] -.-> A19

  U1{{User Confirmation Gate}} --> A9
  G1 -- Very Low / Ambiguous --> U1
```

## Global Controls

- Deterministic-first policy: deterministic logic executes before any LLM advisory step.
- LLM is advisory: all LLM outputs must parse as strict JSON and pass validation.
- Confidence gates:
  - Preprocess planning gate: policy.user_confirmation_threshold
  - Train policy gate: policy.llm_fallback_threshold + confidence checks
- Fallback behavior: invalid LLM output or low confidence falls back to deterministic defaults.
- Reproducibility: all key decisions and artifacts are persisted to runs/<run_id>/.

## Node Contracts

### N00 Trigger Run
- Purpose: Start a run from CLI or agent orchestration.
- Inputs: CLI command, config path, run intent.
- Outputs: run_id, stage intent (profile/preprocess/train/full).
- Primary single-command trigger: `autodl run-full --data <csv> --target <target> --config <config> [--narrative-file ... | --narrative ...]`

### N01 Load Config + Runtime Context
- Purpose: Load AppConfig and runtime settings.
- Inputs: config.yaml (or defaults), environment.
- Outputs: normalized config object.

### N02 Ingest CSV
- Purpose: Load input data into DataFrame.
- Inputs: data path.
- Outputs: DataFrame, basic shape metadata.

### N03 Resolve Target + Validate Schema
- Purpose: Resolve target (case-insensitive), validate required columns.
- Inputs: DataFrame columns, requested target.
- Outputs: resolved target, validation status.

### N04 Profile Dataset
- Purpose: Build deterministic profile (numeric/categorical/text/missing/skew/sparsity).
- Inputs: DataFrame, target, passthrough columns.
- Outputs: profile object.

### N05 Merge Narrative Inputs
- Purpose: Merge narrative from config, file, and CLI text.
- Inputs: narrative config, optional narrative file, optional narrative text.
- Outputs: normalized narrative input.

### N06 Deterministic Preprocess Plan
- Purpose: Build deterministic per-column plan and preprocess defaults.
- Inputs: profile, narrative, config.
- Outputs: director plan (deterministic), confidence.

### N07 Preprocess Confidence Gate
- Purpose: Decide whether LLM plan refinement is needed.
- Inputs: deterministic confidence, policy thresholds.
- Outputs: pass/fail branch.

### N08 LLM Director Refinement
- Purpose: Refine preprocess plan for ambiguous semantics.
- Inputs: run context + narrative + deterministic candidate.
- Outputs: JSON plan response (candidate).

### N08b Validate Plan JSON
- Purpose: Validate LLM plan shape and allowed actions.
- Inputs: LLM response JSON, schema constraints.
- Outputs: validated plan or fallback signal.

### N09 Apply Plan
- Purpose: Apply approved plan actions to DataFrame (drop/map/parse).
- Inputs: DataFrame, validated plan, resolved target.
- Outputs: transformed DataFrame for preprocess, decision log.

### N10 Preprocess + Feature Build
- Purpose: Execute deterministic preprocessing and feature construction.
- Inputs: transformed DataFrame, preprocess config.
- Outputs: preprocessed parquet-ready DataFrame and metadata.

### N11 Persist Preprocess Artifacts
- Purpose: Persist planning and preprocessing outputs.
- Inputs: preprocess outputs, director outputs.
- Outputs:
  - preprocessed.parquet
  - preprocess_metadata.json
  - run_manifest.json
  - director_plan.json (when enabled)
  - decision_log.jsonl (when enabled)

### N12 Build Train Policy (Deterministic)
- Purpose: Choose objective/loss and imbalance policy deterministically.
- Inputs: encoded target distribution, task type.
- Outputs: training_policy decision with confidence.

### N13 Train Policy Confidence Gate
- Purpose: Decide whether LLM refinement is needed for train policy.
- Inputs: training policy confidence, fallback threshold.
- Outputs: pass/fail branch.

### N14 LLM Train Policy Refinement
- Purpose: Refine loss/class-weight strategy when ambiguous.
- Inputs: class counts, task type, deterministic candidate.
- Outputs: JSON policy response (candidate).

### N14b Validate Train Policy JSON
- Purpose: Validate policy response for allowed losses/fields.
- Inputs: LLM response JSON.
- Outputs: validated train policy or fallback signal.

### N15 Start Training
- Purpose: Initialize training process and data splits.
- Inputs: preprocessed parquet, resolved target, training config.
- Outputs: train/val/test matrices.

### N16 Stage-1 Optuna Tuning
- Purpose: Run bounded hyperparameter search.
- Inputs: sampled train/val sets, train policy, Optuna settings.
- Outputs: study object, trial history.

### N17 Finalist Retrain + Evaluate
- Purpose: Retrain top-k candidates and evaluate best model.
- Inputs: full train/val/test sets, finalists, train policy.
- Outputs: best model, metrics, plots, threshold metrics (binary).

### N18 Persist Training Artifacts + REPORT
- Purpose: Persist model, metrics, summary, and report.
- Inputs: model outputs, evaluation results, train policy.
- Outputs:
  - model/
  - training_summary.json
  - metrics_summary.json
  - best_params.json
  - REPORT.md
  - tracking_local_metrics.json

### N19 Agent Monitoring + Next Actions
- Purpose: Central monitoring node for status and follow-up actions.
- Inputs: all stage artifacts and tracker outputs.
- Outputs:
  - run health status
  - unresolved risks/questions
  - suggested next actions (re-run, tune thresholds, retrain, export)

## Monitoring Matrix

- Planning quality:
  - director source (deterministic vs llm)
  - preprocess confidence
  - review questions count
- Training quality:
  - task_type, objective/loss, class-weight usage
  - best_stage1_value, best_final_score
  - evaluation metrics and confusion matrix
- Artifact integrity:
  - required files present per stage
  - schema-valid JSON artifacts

## Failure and Recovery Paths

- Target resolution failure at N03:
  - Abort with available-column diagnostics.
- Invalid LLM JSON at N08b or N14b:
  - Log error and revert to deterministic candidate.
- Low confidence with no approval at N07:
  - Halt for user confirmation.
- Training failure after N15:
  - Persist partial artifacts and tracker status, emit retry guidance.

## Versioning

- Workflow version: v1
- Backward compatibility: additive node metadata only; existing stage outputs remain valid.
