# Evaluator-only checkpoint and training evidence audit

This is a separate CPU program, not a detector component. It reads an explicitly
supplied checkpoint registry, optional candidate training corpora and optional
run/source artifacts. It never calls a model, imports Torch/Transformers, unpickles
training_args.bin, executes training scripts, downloads weights, trains or submits
jobs. The scientific pipeline does not import it.

## Exact first cluster command

After syncing the implementation, use the existing environment and registered
checkpoint inventory. The observation corpus below is the path in train_fastchat.sh
and train_lora.sh; its use by checkpoint-118 is NOT established by that reference.
The query checkpoint's actual training corpus remains unknown, so do not guess one.

```bash
cd ~/BackAgentDef
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
python docs/seek/audit_training_provenance.py \
  --registry configs/seek/local/checkpoints.json \
  --training-data cp_b38e921d052f=/dataset/suaq0001/BackAgentDef/data/observation_attack/poison_m50.json \
  --evidence-file train_fastchat.sh \
  --evidence-file train_lora.sh \
  --verify-weights \
  --output-dir results/seek/private_eval/provenance_v1
```

This hashes existing files, including all files in each enabled checkpoint's
registered weight inventory. It can take disk-I/O time but requires no GPU job.
Missing corpora/checkpoints appear as report blockers; they do not trigger downloads.
If the registry is elsewhere, supply its actual path. If a corpus path is known
from the original training run, supply that path; do not substitute bundled sample
traces and call them the actual training set. The output directory must be new.

The terminal and `summary.json` contain aliases, record counts, verification states
and missing prerequisites. Share that summary first. `private_report.json` contains
paths, hashes and any declared trigger/channel metadata; keep it out of detector
inputs and the public incident schema. The new directory is mode 0700 and files
are mode 0600. Old registries, configs, results and checkpoint files are unchanged.

## What the checks establish

- `matched_registry`: current bytes match the explicitly registered weight/file
  inventory and its recorded identity. This is not training certification.
- Corpus record count and attack_metadata count: metadata present in the supplied
  file, independent of whether any checkpoint actually trained on it.
- Declared direct cues/channels/types: corpus assertions stored only in the private
  report. Occurrence counts refer to any human/user turn, possibly a demonstration;
  they do not prove initial-task insertion, causality, or semantic removability.
- Metadata and evidence hashes: reproducible references for evaluator review.
  Pickled training arguments are hashed as opaque bytes, never loaded as Python.
- Assessment stays `unverified`. Neither filenames, labels in a registry nor
  current training scripts establish the history of an existing checkpoint.

## Evidence still required for each checkpoint

1. A contemporaneous training-run artifact linking this checkpoint/output to the
   actual input corpus and its hash (not a newly asserted mapping).
2. The training code revision, base checkpoint identity, launch arguments and
   relevant poisoning/data-generation configuration from that run.
3. Evaluator-side documentation of the cue, channel, exposure rule, target behavior
   and poisoning rate; explain whether the cue is incidental or legitimate product/
   task information. Keep this separate from detector inputs.
4. Independent clean/base-control provenance and training-vs-study overlap evidence
   before confirmation. No such evidence is fabricated from an empty inventory.

An existing training manifest can be supplied through the registry or with
`--evidence-file`; this program inventories its bytes for review and does not
silently promote it to verified truth. Additional corpora can be inspected in a
new audit directory, using `--training-data ALIAS=/actual/path.json`. JSON arrays,
JSONL, data-array containers and legacy comma-separated object traces are supported.
Unsupported/malformed corpora are marked explicitly; empty files provide no evidence.

## Future collection sampling fix

`select_collection_tasks` now treats collect_limit as the number of independent
dependence groups for real collection. It selects one representative per group by
stable task fingerprint, orders groups by existing split order then group identity,
and rejects a request larger than available group coverage. It never selects using
model outputs, training labels or trigger content. The collector writes an immutable
collection_selection.json before constructing the environment. Development precedes
discovery and holdout splits as before; holdout material is not substituted into
discovery. A four-group development pilot does not become confirmation evidence.

These changes affect only future runs. The v2 pilot still contains the originally
selected size variants and remains valuable as capture/replay integration evidence.
Do not resume that scientific run with changed source or reinterpret its sample
count. No new GPU collection is recommended until provenance and valid intervention
prerequisites are resolved.

## Verification scope

CPU tests use fake checkpoint bytes, synthetic corpora and fabricated task groups.
They cover sibling-heavy selection, ordering, insufficient/cross-split groups,
missing/corrupt artifacts, metadata-free legacy traces, malformed data, private
permissions, summary redaction, immutable outputs, unchanged registries and absence
of model imports. They do not establish any real checkpoint's training history.
