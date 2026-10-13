# Source basis and external research

## User-provided basis

- `context/CODEX_AUDIT_2026-10-13_source.md`: original supplied audit bytes from `Pasted markdown(4).md`. The authoritative in-conversation citation is `turn12file0` with the supplied line numbers. This is Codex's report, not an independent audit.
- `context/MANUSCRIPT_THEORY_EXCERPT.tex`: extracted from the mounted `Shield_and_Seek_with_Seek_Theory.tex`. It preserves the current formula and assumptions rather than substituting a new confirmation backend.
- Prior supplied handoff `/mnt/data/Seek_Discovery_Phase_2026-10-11/CODEX_UPDATE_PROMPT.md` was read to identify already-requested/implemented components. It is not treated as an updated repository snapshot.

This handoff does not include a new inspection of the current source tree or remote experiment archive. The old uploaded repository zip predates the newest implementation and was not represented as current.

## Verified primary research, consulted 13 October 2026

1. Robert Nowak. **Noisy Generalized Binary Search.** NeurIPS 2009.
   Official proceedings: https://papers.nips.cc/paper_files/paper/2009/hash/556f391937dfd4398cbac35e050a2177-Abstract.html
   Relevance: strategic selection of queries that split candidate explanations, including noisy-response issues. Active hypothesis testing itself is established, not a novelty claim here.

2. Robert D. Nowak. **The Geometry of Generalized Binary Search.** arXiv:0910.4397, revised 2013.
   Primary author manuscript record: https://arxiv.org/abs/0910.4397
   Relevance: logarithmic query results require structural conditions on the hypothesis/test geometry. A semantic hierarchy by itself does not supply those conditions.

3. Finnian Lattimore, Tor Lattimore, Mark D. Reid. **Causal Bandits: Learning Good Interventions via Causal Inference.** arXiv:1606.03203 / NeurIPS 2016.
   Primary manuscript record: https://arxiv.org/abs/1606.03203
   Relevance: allocating interventions to learn effects efficiently. The proposed Seek design is not claimed to inherit this paper's regret theorem.

4. Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, Jasjeet Sekhon. **Time-uniform, nonparametric, nonasymptotic confidence sequences.** Annals of Statistics 49(2):1055–1080, 2021. DOI 10.1214/20-AOS1991.
   Primary manuscript record: https://arxiv.org/abs/1810.08240
   Relevance: time-uniform inference for repeated evidence checks. Existing Seek uses its explicitly derived conservative bound; discovery scores remain separate.

The algorithm design is a proposal derived from the user's audit and these general principles. This is a targeted grounding check, not an exhaustive novelty review. No claim is made that HACS, semantic graphs or weighted disagreement are individually new. The idealized finite-class proposition is a supplied derivation under explicit assumptions, not empirical evidence of real semantic recovery.
