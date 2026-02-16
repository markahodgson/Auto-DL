# Hybrid Policy Schema for AutoDL CLI

## 1) Scope
This schema defines how the CLI chooses preprocessing, model search, and training behavior using:
- deterministic decision nodes (primary)
- confidence scoring (gating)
- local LLM advisory fallback (only on low confidence)

The LLM is advisory only. Final execution must pass deterministic validation.

## 2) Run Inputs (Normalized Context)
Every decision node receives a `run_context` object.

```json
{
  "run_id": "2026-02-15T17-02-44Z_8f3a",
  "task_type": "auto|binary|multiclass|regression",
  "target": {
    "name": "label",
    "dtype": "int64",
    "n_unique": 2,
    "class_imbalance_ratio": 0.09
  },
  "dataset": {
    "n_rows": 2200000,
    "n_cols": 180000,
    "estimated_sparse_ratio": 0.985,
    "memory_estimate_mb": 9600,
    "file_format": "csv|parquet"
  },
  "column_profile": [
    {
      "name": "body_text",
      "kind_scores": {
        "text": 0.92,
        "categorical": 0.06,
        "numeric": 0.02
      },
      "null_rate": 0.01,
      "avg_token_count": 34.5
    }
  ],
  "constraints": {
    "time_budget_minutes": 120,
    "max_trials": 100,
    "gpu_available": true,
    "reproducibility_level": "high",
    "deployment_target": "tensorflow_saved_model"
  },
  "user_overrides": {
    "force_text_backend": null,
    "metric": null,
    "sample_fraction": null
  }
}
```

## 3) Decision Nodes

### N01 Task + Metric Inference
- Inputs: target stats, task override
- Output: `task_type`, primary metric
- Deterministic rule:
  - 2 unique target values -> `binary`, metric `auc_pr` if imbalance < 0.2 else `auc`
  - discrete low-cardinality (>2) -> `multiclass`, metric `macro_f1`
  - otherwise -> `regression`, metric `rmse`
- Confidence:
  - `c = 1.0` if user override present
  - else based on separability of target type: `c = clamp((p_best - p_second + 0.2), 0, 1)`

### N02 Split Strategy
- Inputs: n_rows, time budget, leakage risk score
- Output: split config (`train/val/test`, stratification, grouped split)
- Deterministic rule:
  - Default 80/10/10, stratify for classification
  - Use grouped split if group leakage detector score >= 0.7
- Confidence:
  - `c = 1 - leakage_ambiguity_score`

### N03 Sparse Representation Policy
- Inputs: sparse ratio, memory estimate, text/cat mix
- Output: `sparse_mode`, feature store backend
- Deterministic rule:
  - If sparse ratio >= 0.9 or memory estimate > available_memory*0.7 -> keep sparse pipeline
  - Prefer CSR for training transforms, CSC for column slicing-heavy profiling
- Confidence:
  - `c = max(sparse_ratio, memory_pressure_score)`

### N04 Text Column Routing
- Inputs: text kind scores, avg tokens, n_rows, deployment target
- Output: text backend per column (`sklearn_tfidf`, `sklearn_hashing`, `tf_text_vectorization`)
- Deterministic rule:
  - Fast tuning path:
    - huge rows OR strict time budget -> `sklearn_hashing`
    - medium scale -> `sklearn_tfidf`
  - Final deployment path:
    - if deployment target is SavedModel and latency constraints allow -> optional second-stage `tf_text_vectorization`
- Confidence:
  - `c = mean(max_kind_score_per_text_col)` adjusted by tokenizer_cost_uncertainty

### N05 Sampling Policy for Tuning
- Inputs: n_rows, class imbalance, time budget
- Output: `sample_fraction`, `stratified=true|false`
- Deterministic rule:
  - Default staged tuning:
    - stage1: `sample_fraction` in [0.05, 0.20]
    - stage2: retrain top-k on [0.4, 1.0]
  - Always stratify for classification
- Confidence:
  - `c = 1` if n_rows > 100k else `0.85`

### N06 Model Search Space Builder
- Inputs: task, feature dims, sparse mode
- Output: bounded search space
- Deterministic rule:
  - Dense blocks: layers 1..6, units 64..2048
  - Dropout 0.0..0.6
  - LR log-uniform 1e-5..5e-2
  - Batch size depends on memory/gpu
  - Optional focal loss for imbalance >= threshold
- Confidence:
  - `c = coverage_score(search_space)` with minimum floor 0.75

### N07 Tuner Engine Selection
- Inputs: user preference, distributed availability, pruning requirement
- Output: `optuna|keras_tuner`
- Deterministic rule:
  - Default `optuna` (TPE sampler + median pruner)
  - fallback `keras_tuner` if strict Keras-native stack requested
- Confidence:
  - `c = 0.95` if no conflicting constraints

### N08 Loss/Objective + Imbalance Handling
- Inputs: task, imbalance ratio
- Output: objective config (`cross_entropy`, `focal_loss`, class weights)
- Deterministic rule:
  - Binary/multiclass with high imbalance: class weights first, focal loss optional candidate
- Confidence:
  - `c = 1 - objective_conflict_score`

### N09 Stopping + Regularization
- Inputs: metric volatility, overfit gap
- Output: early stopping patience, max epochs, weight decay/dropout ranges
- Deterministic rule:
  - early stop on plateau with patience 3..10
  - apply gradient clipping when instability detected
- Confidence:
  - `c = training_signal_quality`

## 4) Confidence and Escalation Rules
- Node decision is accepted automatically if `c >= 0.80`.
- If `0.50 <= c < 0.80`, call LLM advisor and require JSON response.
- If `c < 0.50`, ask user or require explicit override; LLM may propose options but cannot auto-apply.

Global guardrails (always deterministic):
- no target leakage
- no invalid metric-task pair
- no split overlap
- resource cap checks
- schema validation for all generated configs

## 5) LLM Prompt Contract

### System Contract
- Role: policy advisor for AutoDL configuration
- Must return strict JSON only, no markdown
- Must provide confidence and rationale per proposed action
- Must not change user hard constraints

### Input Payload to LLM
```json
{
  "request_type": "policy_suggestion",
  "run_context": {},
  "low_confidence_nodes": ["N04", "N08"],
  "candidate_actions": {
    "N04": ["sklearn_hashing", "sklearn_tfidf", "tf_text_vectorization"],
    "N08": ["class_weights", "focal_loss", "binary_crossentropy"]
  },
  "constraints": {
    "must_be_cli_reproducible": true,
    "no_leakage": true,
    "time_budget_minutes": 120
  }
}
```

### Required LLM Output Shape
```json
{
  "version": "1.0",
  "node_recommendations": [
    {
      "node_id": "N04",
      "selected_action": "sklearn_hashing",
      "confidence": 0.82,
      "rationale": "Large corpus and tight tuning budget favor hashing for stage-1 speed.",
      "alternatives": [
        {
          "action": "tf_text_vectorization",
          "when_preferable": "Final deployment model requiring in-graph preprocessing"
        }
      ]
    }
  ],
  "risk_flags": [
    {
      "code": "TEXT_DRIFT",
      "severity": "medium",
      "message": "Vocabulary drift risk if fixed hashing dimension is too small."
    }
  ],
  "questions_for_user": []
}
```

## 6) JSON Schema (LLM Response)
Use this JSON schema to validate the LLM response before any action is applied.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://dnn-automation.local/schemas/llm-policy-response.schema.json",
  "title": "LLMPolicyResponse",
  "type": "object",
  "additionalProperties": false,
  "required": ["version", "node_recommendations", "risk_flags", "questions_for_user"],
  "properties": {
    "version": { "type": "string", "pattern": "^1\\.[0-9]+$" },
    "node_recommendations": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["node_id", "selected_action", "confidence", "rationale", "alternatives"],
        "properties": {
          "node_id": { "type": "string", "pattern": "^N[0-9]{2}$" },
          "selected_action": { "type": "string", "minLength": 1 },
          "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
          "rationale": { "type": "string", "minLength": 5 },
          "alternatives": {
            "type": "array",
            "items": {
              "type": "object",
              "additionalProperties": false,
              "required": ["action", "when_preferable"],
              "properties": {
                "action": { "type": "string" },
                "when_preferable": { "type": "string" }
              }
            }
          }
        }
      }
    },
    "risk_flags": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["code", "severity", "message"],
        "properties": {
          "code": { "type": "string" },
          "severity": { "type": "string", "enum": ["low", "medium", "high"] },
          "message": { "type": "string" }
        }
      }
    },
    "questions_for_user": {
      "type": "array",
      "items": { "type": "string", "minLength": 3 }
    }
  }
}
```

## 7) Enforcement Pipeline
1. Compute deterministic node outputs + confidence.
2. Send only low-confidence nodes to LLM.
3. Validate LLM response against schema.
4. Re-run guardrails.
5. Materialize final `run_manifest.json` for reproducibility.

## 8) Suggested Artifacts per Run
- `runs/<run_id>/run_manifest.json`
- `runs/<run_id>/decision_log.jsonl`
- `runs/<run_id>/optuna_trials.parquet`
- `runs/<run_id>/metrics_summary.json`
- `runs/<run_id>/model/` (SavedModel)
