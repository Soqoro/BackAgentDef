# V2 saved-array integration — 15 November 2026

The uploaded native ZIP matches SHA-256 `2fb5eaf9385c02079ba5c86ce13e8301715bb7649354db23e0458a7cefc615e4`. The additive CPU-only checker is `scripts/seek/check_v2_saved_trainer_inputs.py`. It has no training mode and no model or Trainer constructor.

It checks archive manifests and candidate/receipt identities, combines only the unchanged foundation arrays with the 44 revised addition arrays for each role, and passes all 1,902 records per role through the existing `trainer_inputs`, `AnnotationDataset`, saved processor and guarded collator. Every returned ID, attention mask and label sequence must equal the bound saved arrays. Old addition arrays cannot substitute for new response labels.

The local Python interface uses a dictionary factory for TrainingArguments. It is not actual Transformers execution or fresh tokenization. The optional cluster CPU interface constructs real TrainingArguments and CPU tensor batches, but never a Trainer, model, optimizer or forward/backward pass. Neither interface installs v2 admission into the production entry. The optimizer remains unresolved; CPU framework defaults do not approve it for training.

Four new array-admission tests cover replacement, stale response hashes, stale annotation hashes and missing additions. The 21 existing feasibility tests also pass. Receipt source hashes identify the actual current helpers used. Historical runtime authenticity and numerical training behavior remain outside this check.

No changes to private markers, target mapping, historical results, candidate corpora or authorization gates. Transfer allocation remains 132; optional 32 remains disabled; all execution permissions remain false.
