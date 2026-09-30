# WebShop source provenance and intervention eligibility

## Verified source evidence

`web_agent_site/envs/web_agent_text_env.py:convert_html_to_text` emits visible text
nodes in order, strips each node, and joins them with ` [SEP] ` in text mode. The
collector wraps that observation in `Observation:\n...\n\nAvailable Actions:\n...`.
The HTML exists in `env.state['html']` at capture time. Old v2 snapshots recorded
plain observations, not HTML node provenance; it cannot be reconstructed reliably
from separators alone. Real v2 snapshots are cluster-only in this workspace.

The checked-in templates expose these fields:

| Template | Visible information | Eligibility |
| --- | --- | --- |
| search_page.html | WebShop identity, original instruction, Search button | Protected |
| results_page.html | Instruction, pagination/count, product IDs/titles/prices, optional brand/rating/availability, navigation | Protected |
| item_page.html | Instruction, product identity/facts, option names/values, purchase and subpage buttons | Protected |
| description_page.html | Product Description plus instruction/navigation | Product evidence; protected |
| features_page.html | Product BulletPoints plus instruction/navigation | Product evidence; protected |
| review_page.html | Review titles, scores and bodies plus instruction/navigation | Decision-relevant product evidence; protected |
| attributes_page.html | Product attribute fields plus instruction/navigation | Protected |
| done_page.html | Reward/goal/outcome information | Protected; no incidental slot |

Description, feature and review regions contain prose, but prose is not evidence
of semantic irrelevance. For example, removing a durability claim from a product
review may change a shopping decision even if that word was not in the goal.
Neither a trigger-looking string nor an absent goal-word match licenses an edit.
The source audit never reads training labels, attack metadata or model outcomes to
choose regions. No removable incidental field is established by these templates.

## Implemented behavior

`seek/source_audit.py` maps HTML text nodes to offsets only when its rendering
exactly equals the recorded observation and the policy-input wrapper matches.
It classifies DOM instruction text as goal, links/buttons/options as protected
actions, product classes as facts, and named description/features/review containers
as product prose. Field names describe DOM provenance, not a semantic guarantee.
Unknown text, separators and the available-actions block remain hard. Page-supplied
attributes such as `data-seek-kind=narrative` cannot grant edit permission.

HTML parsing uses the Python standard library and is deliberately conservative.
If its output differs from legacy BeautifulSoup rendering (including whitespace,
malformed markup or segmentation differences), the entire input remains hard with
`html_observation_mismatch`. A Shield-transformed input that no longer matches the
raw observation likewise remains hard; offsets are never guessed or transferred.
No changes are made to WebShop rendering, execution or the policy input itself.

New captures save immutable `source_audits/CASE-TAG.json` sidecars before generation.
The final snapshot binds its source audit hash in shield_report. The audit includes
HTML/input/observation hashes, offsets and field labels, not raw HTML or private
checkpoint labels. Raw and defended inputs receive separate audits. Existing
snapshots are never retroactively relabeled. Goal fields remain uneditable even
in the no-goal-preservation ablation; that ablation can relax screening of narrative
spans but cannot delete the frozen instruction.

The current adapter therefore exposes **zero approved narrative regions**. It is
an auditable field map, not a claim that useful interventions now exist. A new
incidental source type needs source-level provenance and an independent argument
that changes preserve all decision-relevant facts. Merely adding a DOM marker,
renaming a description, inserting a synthetic cue, or increasing the trajectory
length does not establish a valid legacy trigger intervention. The supplemental
training specification remains a separate option, with no training authorized or
performed here.

## Audit the completed v2 pilot without GPU work

After syncing these files to the cluster, run in the existing environment:

```bash
cd ~/BackAgentDef
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
python docs/seek/audit_sources.py \
  --run-root results/seek/sneakers_pilot_v2 \
  --output results/seek/diagnostics/sneakers_pilot_v2_source_audit.json
```

This reads validated snapshots and writes a new immutable diagnostic report. It
imports no model, tokenizer or WebShop environment, and makes zero victim calls.
The terminal shows only counts. The report includes the full observations and
original instructions for source review, but no histories, private evaluator
labels or checkpoint paths. Simulated and real snapshot counts are separate.
For existing v2 captures, expect `missing_html_provenance`; it is not a failed
replay. Review the report before proposing further GPU experiments. Return this
JSON alongside the compact terminal summary rather than pasting the inventory.

Do not resume a scientific v2 phase after syncing this changed source: the source
fingerprint and environment adapter fingerprint have changed. The CPU audit itself
is read-only with respect to v2 and does not require new collection, inventory or
replay. New captures using this mapper require fresh run IDs and environment
metadata; no such cluster jobs have been submitted locally.

## Verification boundary

CPU fixtures cover exact coverage/offsets, duplicate text, Unicode/entities,
comments/whitespace, goal/action separation, protected product prose, forged
narrative markers, missing/misaligned HTML, Shield transformations, ablation goal
protection, and read-only audit behavior. These are simulated source-layout tests,
not real-model or real-catalogue semantic verification. The user's successful
Qwen role smoke and valid v2 replay remain separate reported cluster evidence.
