# LLM Director Prompt Template

## System prompt
You are the AutoDL Director for tabular ML pipelines.
Return strict JSON only. Do not return markdown.
Your response must validate against the provided JSON Schema.
You can reason from data profile, sample values, and user narrative.
You must not violate hard constraints.
If uncertain, ask short review questions in `review_questions`.

## User prompt template
```json
{
  "request_type": "director_plan_v1",
  "schema_id": "llm-director-plan.schema.json",
  "hard_constraints": {
    "must_be_reproducible": true,
    "cannot_use_target_leakage": true,
    "allowed_actions": [
      "impute_median",
      "impute_mode",
      "fill_missing_token",
      "one_hot",
      "binary_map",
      "ordinal_map",
      "scale_standard",
      "vectorize_hashing",
      "vectorize_tfidf",
      "parse_datetime",
      "drop"
    ],
    "allowed_text_backends": ["none", "hashing", "tfidf", "tf_text_vectorization"]
  },
  "run_context": {
    "task_hint": "auto",
    "target_hint": null,
    "rows": 0,
    "columns": 0,
    "dataset_profile": {
      "numeric_cols": [],
      "categorical_cols": [],
      "text_cols": [],
      "missing_rates": {},
      "sample_values": {}
    },
    "user_narrative": {
      "dataset_summary": "",
      "column_hints": {
        "sex": "Represents biological sex; values may appear as 0/1, M/F, or male/female"
      },
      "target_definition": "",
      "leakage_warnings": [],
      "business_goal": ""
    },
    "config_defaults": {
      "normalize_numeric": true,
      "one_hot_categorical": true,
      "text_backend": "hashing"
    }
  }
}
```

## Validation and execution notes
- Parse LLM output as JSON.
- Validate against `docs/schemas/llm_director_plan.schema.json`.
- Reject if invalid and retry with stricter reminder.
- If `confidence < policy.user_confirmation_threshold`, require user confirmation before applying plan.
- Execute preprocessing deterministically from validated plan; never execute free-form text.
