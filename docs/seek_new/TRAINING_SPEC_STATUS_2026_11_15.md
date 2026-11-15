# Bound v2 training specification — 15 November 2026

The source-suitability and response-contract reviews are accepted only in their stated bounded scopes. A new private specification binds their exact supplied bytes to candidate c934d2677795188074ed3c4c466d110cdbdd1bdb0ef1bc9e0baf02b4651653f0. Original artifacts and gates are unchanged.

Implementation: `scripts/seek/prepare_v2_training_spec.py`. It verifies the candidate plan seal, parent archive/proposal binding, 1,902 records and sidecars per role, and exact review bytes. It packages current integration source snapshots without executing the trainer or tokenizer.

The inherited paired recipe retains seed 271028, two epochs, learning rate 5e-5, microbatch 1, accumulation 8, cosine schedule, warmup 0.02 and weight decay 0.1. Optimizer and GPU topology remain null. Older templates' seed, optimizer and four-GPU estimate are not substituted. Expected updates remain unresolved. C denotes C_sham, the cue-exposed benign-label control.

The existing entry consumes an earlier evidence format. V2 array admission, actual dataset/collator/argument wiring, initialization weight identity, trainable modules, optimizer, sampler/update behavior, loss normalization, precision, resources and fresh output paths require further work. No real Trainer, optimizer, forward/backward or model loader ran. Source/corpus decisions remain separate from bounded suitability acceptance.

Validation: 3 new specification tests and 21 existing feasibility tests passed. The private bundle was read back and its manifest verified. This is static preparation, not new native tokenization or historical runtime authentication.

Private local artifact: `outputs/v2_training_spec_2026_11_15/bound/review.private.zip`.
Specification seal: `009e04021b6f24078dbfe010f3975f2946360225e110ce67665b926b44046798`.
ZIP SHA-256: `3bbc02938845e8193c88db3e6cb36a1477fa18ea034b2565b5424562ab52a61a`.

Next input: the existing Jupyter v2 native review archive at `output-contract-v2-WWMpBC/native/review.private.zip`, not a rerun of tokenization. Its supplied independent review reports ZIP SHA-256 `2fb5eaf9385c02079ba5c86ce13e8301715bb7649354db23e0458a7cefc615e4`. Use it for additive CPU integration with revised additions and preserved foundation evidence. Do not admit old addition masks for changed responses.

Transfer remains 132 attempts, the optional 32-response diagnostic is disabled, and utility remains 24 episodes with caps unresolved. All execution, installation, training, resource and method-matrix permissions remain false.
