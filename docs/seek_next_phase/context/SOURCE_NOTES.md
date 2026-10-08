# Source notes and attribution boundaries

## User-supplied sources

1. Codex report, “Seek Stage II: review and experiment plan”, dated 8 October 2026. Original uploaded Markdown-escaped HTML and a plain-text rendering are copied beside this file. The report is the source of the six claim statuses, effect summaries, model identities, source-provenance gaps, development failures and P0–P6 priorities.
2. `Shield_and_Seek_with_Seek_Theory.tex`, supplied earlier in this conversation. The theoretical subsection is extracted beside this file. It specifies the [-1,1] confidence radius, adaptive registration assumptions, detectability and read-only separation from Shield.
3. Previous local-Codex -> GitHub -> cluster-Jupyter -> Slurm handoff. Current source code was not attached with this report. The old `BackAgentDef-main.zip` predates these implementations and must not replace the live checkout.

The report says that the complete raw cluster response archive and registry chain were not supplied to its local audit. This handoff likewise has no independent copy of those underlying artifacts. Reported facts must not be upgraded to independently reproduced results until P0 verifies them.

## External primary-source verification (8 October 2026)

- Yang et al., *Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents*, arXiv:2402.11208v2. https://arxiv.org/html/2402.11208v2 — Figure 1 caption identifies Adidas in an observation as the illustrated observation trigger; sections 3.2.2 and 4.1.1 describe selection of Adidas products when they appear in returned results, with the WebShop experiment concerning Adidas sneakers. This is an important distinction from treating a sneaker token as the only possible observation cue. The published construction is not proof of this user's specific checkpoint mechanism. Use brand-only and category-conditioned explanations as alternatives.
- Howard et al., *Time-uniform, nonparametric, nonasymptotic confidence sequences*. https://arxiv.org/abs/1810.08240 — supports the general use of time-uniform evidence bounds for repeated monitoring. The exact simpler Hoeffding–union formula used here is the supplied manuscript's rule, not a claim to implement that paper's optimized intervals.
- Official Slurm job-array documentation. https://slurm.schedmd.com/job_array.html — `%` caps array concurrency, `%A`/`%a` identify output files, and afterok describes process completion dependencies. Scientific-status checks remain application-level responsibilities.

## Proposed designs, not source-reported experiments

The label-swap factorial, new category controls, finite-support census mode, response reuse requirements, sample caps and discovery benchmark in this package are proposed refinements. They were not completed in the supplied report. The optional goal-exposure factorial is not enabled by default. No effect size or runtime is predicted.

Reference code is an independent arithmetic/test helper, not an implementation or review of the live repository. Unknown semantic discrepancy remains unknown; neither a citation nor agent agreement justifies eta=0.
