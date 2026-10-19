# Relevant proposal excerpts
Source: Shield_and_Seek_Revised_Proposal.tex
These excerpts are manuscript specifications, not completed experiment evidence.

## Source lines 174–273
```latex
\subsection{Stage II: \seek{} with hierarchical active search}
\label{sec:seekmethod}
\seek{} turns these records into controlled investigations. \hacs{} makes the choice of experiments explicit: hypotheses describe possible response rules, while experiments are comparisons that can distinguish them. The controller owns arithmetic and scheduling; the agents ground the candidate meanings and challenge the experimental interpretation.

\paragraph{1. Define the outcome before searching for its cause.}
The Action agent constructs a scorer for the behavior visible in the proposal, such as addition of a particular argument to a tool call. The target is grounded in public action evidence, not supplied by a private attack label. Malformed actions and missing entity mappings are unscorable outcomes, not evidence that the behavior disappeared. Whether the behavior violates the task is stored separately.

\paragraph{2. Construct a testable hypothesis space.}
A domain adapter supplies typed fields, public input spans, and a bounded vocabulary of reviewed concepts. \hacs{} forms candidates from exact expressions, schema attributes, concept ancestors and siblings, source conditions, and optional small conjunctions. Broader high- or low-rate behavior, position following, and ordinary task-driven choice remain alternative explanations. A concept requires an executable matcher, and an experiment requires a valid renderer. Unsupported or unknown candidates remain identified as such rather than receiving a false prediction.

A hierarchy structures the search; it does not establish semantic truth. Candidate origins are recorded as grammar-generated, agent-proposed, or evaluator-specified. A development-informed library is frozen before evaluation and described as such. Neither private attack labels nor previously successful contrasts are silently inserted into an incident-led investigation.

\paragraph{3. Review the comparison and its scope.}
Lexical tests can remove or replace an incidental expression while retaining the legitimate task, and then reinsert it against matched decoys. Semantic tests may intentionally change a task-bearing factor. The Goal agent records that change and the controlled background rather than describing the requests as equivalent. The State agent constructs coherent inputs, while the Action agent checks that both arms measure the same behavior. For an observation-side choice experiment, both arms preserve meaningful action opportunities; deleting the only target option and counting its disappearance would not isolate a preference effect.

\paragraph{4. Choose where competing predictions disagree.}
For a feasible experiment $e$, each assessable hypothesis $h$ provides an arm-wise prediction $p_h(e,a)$ for $a\in\{0,1\}$. The predictions are declared search aids, not calibrated probabilities that a backdoor exists. With exploratory weights $w_h$, the controller scores
\begin{equation}
U(e)=\frac{\sum_{a\in\{0,1\}}\operatorname{Var}_{h\sim w_e}
[p_h(e,a)]}{c(e)},
\label{eq:acquisition}
\end{equation}
where $c(e)$ is the logical policy-response cost and $w_e$ is normalized over hypotheses with valid predictions for that experiment. Unknown candidates stay in a separate coverage record. A periodic coverage schedule visits unresolved testable distinctions so that an early ranking cannot permanently suppress a branch. Ties and the tie-breaking rule are recorded. The score expresses expected discrimination under the declared predictions, not statistical significance or unique causal identification.

\paragraph{5. Update from both observed arms.}
Each probe is an isolated query to the fixed task policy. The ledger records the two scored outcomes $(Y_{i,1},Y_{i,0})$, their signed difference, and the complete response bindings. Both $(1,1)$ and $(0,0)$ have zero difference, but support different explanations. \hacs{} therefore updates from the arm vector, for example through
\begin{align}
\ell_i(h)&=\tfrac12\sum_{a\in\{0,1\}}(p_h(e_i,a)-Y_{i,a})^2,\\
w_h&\leftarrow\frac{w_h\exp[-\lambda\ell_i(h)]}
{\sum_{h'}w_{h'}\exp[-\lambda\ell_i(h')]},
\label{eq:searchupdate}
\end{align}
with a fixed update rule on assessable hypotheses. Missing predictions are not scored as zero. The controller counts each acquired evidence unit once and gives the agents a compact summary of observed outcomes, hypothetical predictions, fit changes, and unresolved alternatives. These search weights are not confirmation certificates.

\paragraph{6. Refine the distinction, not the desired answer.}
A new comparison should challenge a surviving explanation or reduce declared uncertainty. Identical greedy replays serve consistency checking rather than providing new discovery information. When feasible experiments give identical predictions for several candidates, \seek{} retains a prediction-equivalence class. One bounded library expansion can expose additional distinctions; otherwise the outcome is a scoped unresolved class or an inconclusive investigation. A broad condition such as a document category or intent family can be a useful signature without identifying its unique internal implementation.

\paragraph{7. Confirm a frozen relationship.}
Before prospective confirmation, \seek{} freezes the candidate condition or class, outcome, renderer, context distribution, direction, minimum effect, and stopping rule. Fresh evidence is separate from discovery and development reuse. The output distinguishes a measured implemented effect from its stronger semantic interpretation. A complete evaluation on a fixed finite set is also possible, but it remains a finite-support description rather than a sampling-based certificate about a wider distribution.

\begin{algorithm}[t]
\caption{\seek{}: systematic, role-reviewed investigation}
\label{alg:seek}
\begin{algorithmic}[1]
\Require Public evidence, domain adapter, isolated policy access, budget
\State Define the observed behavior and validate replay
\State Build bounded hypotheses and feasible comparisons
\State Record candidate provenance and prediction-equivalent classes
\While{discovery budget remains}
    \State Select a feasible comparison by Equation~\ref{eq:acquisition} or the coverage schedule
    \State Roles review meaning, action opportunities, and controlled changes
    \State Query isolated arms; record outcomes or a failure status
    \State Update assessable hypotheses using both outcomes
    \State Preserve ties and unknowns; justify the next test
    \State Stop at a candidate, exhausted useful tests, or budget limit
\EndWhile
\State Freeze a candidate-specific confirmation contract
\State Collect fresh pairs and apply Section~\ref{sec:seektheory}
\State Return a scoped supported relation, unresolved class, or inconclusive result
\end{algorithmic}
\end{algorithm}

\paragraph{Why the next test matters.}
Consider a generic preference for target $T$. A literal-expression rule, a narrow intent-family rule, and a broad-category rule can agree on an initial input. An alternative expression probes the first distinction; a sibling category probes the second. On some tests a constant target preference remains tied with a broad category. The useful conclusion is which predictions the evidence separates, not that one favorable outcome must reveal the entire mechanism. Section~\ref{sec:qualitative} makes this reasoning visible through illustrative examples.

\subsection{Theoretical guarantees}
\label{sec:seektheory}
The search above determines what to investigate. The following results govern when the resulting evidence supports a claim. They do not assume that agent agreement makes a semantic comparison valid, nor do they guarantee that the hypothesis library contains every backdoor.

For registered experiment $j$, paired renderers $R_{j,1}(Z)$ and $R_{j,0}(Z)$ vary the declared condition using background context $Z$. The fixed scorer $Y_j$ measures the same behavior in both arms. Define
\begin{align}
D_{j,i}&=Y_j(\pi(R_{j,1}(Z_i)))-Y_j(\pi(R_{j,0}(Z_i))),\\
D_{j,i}&\in[-1,1],\qquad
\widehat\Delta_{j,n}=\frac1n\sum_{i=1}^{n}D_{j,i}.
\label{eq:pairedeffect}
\end{align}
The mean implemented effect is $\widetilde\Delta_j$. Its intended semantic interpretation is $\Delta_j$, with a justified discrepancy bound
$|\widetilde\Delta_j-\Delta_j|\leq\eta_j$.
Without such a bound, only the implemented effect is certified; an unknown $\eta_j$ is never set to zero. Exact control supports $\eta_j=0$ only when it makes the implemented and intended estimands coincide.

For study-wide error level $\delta$, let
\begin{align}
r_{j,n}&=\sqrt{\frac{2}{n}\log\!\left(
\frac{2j(j+1)n(n+1)}{\delta}\right)},\\
L_{j,n}&=\widehat\Delta_{j,n}-r_{j,n}-\eta_j.
\label{eq:seekbound}
\end{align}
The registered direction and threshold $\tau_j$ are fixed before confirmation. Certification requires the strict inequality $L_{j,n}>\tau_j$; negative exploratory effects are not relabeled positive after inspection.

\paragraph{Theorem 1 (adaptive soundness).}
Assume each experiment contract is frozen before its own fresh confirmation data; conditional on that registration, paired observations are independent with a fixed mean under the specified context distribution and generation procedure; the victim and scorer are fixed; and the discrepancy bound is valid. Then, under adaptive hypothesis proposal, scheduling, and stopping,
\begin{equation}
\Pr\!\left(\exists j,n:
L_{j,n}>\tau_j\ \text{and}\ \Delta_j\leq\tau_j\right)\leq\delta.
\label{eq:soundness}
\end{equation}
This controls erroneous effect certificates across the registered study. It is not a probability that a model was maliciously trained.

\paragraph{Theorem 2 (detectability).}
For a registered, testable relation with
```

## Source lines 285–316
```latex

\subsection{Environment, attacks, and evaluation protocol}
\paragraph{Why \webshop{} captures the agent-level problem.}
\webshop{} is particularly suitable for testing $x\rightarrow y$ in an interactive agent \citep{yao2022webshop}. Its initial request supplies a query channel, later pages supply an observation channel, and searches, clicks, option selections, and purchases expose distinct policy effects. The graded reward measures whether the eventual item satisfies the user's requirements. Thus trigger exposure, proposed behavior, executed behavior, and task utility can be inspected separately within one controlled environment.

These properties instantiate the general interface assumed by \method{}. Shopping terms are supplied by the domain adapter, not embedded in the defense rule. A target argument in another tool domain plays the same formal role as a target brand here. The experiments establish behavior on the evaluated interfaces and distributions; they do not turn a shared abstract formulation into empirical evidence of transfer to untested environments.

\paragraph{Retained attack setting.}
We preserve the existing query- and observation-attack results and their evaluation protocol. The supplied setup reports 200 clean evaluation tasks and 100 poisoned sneaker-related trajectories in total, comprising 50 query-channel and 50 observation-channel traces, with Llama-2-7B-Chat fine-tuning using AgentInstruct \citep{zeng2023agenttuning,touvron2023llama}. The attack construction follows the query/observation setting of \citet{yang2024watch}. Checkpoint-specific manifests must identify which trace set trained each policy; the total does not establish that a single checkpoint contains both channels. Separate reported category experiments cover shirts, bags, and watches. We call this the \emph{retained category-linked setting}, which describes its attack design rather than proving the exact learned semantic boundary.

\paragraph{New clean, lexical, and semantic conditions.}
The planned main \seek{} study holds the query channel fixed while varying the policy condition. An unpoisoned agent supplies the clean control. A lexical variant associates an arbitrary exact marker with target-directed behavior, using a matched neutral insertion field so that unusual formatting alone is not the intended cause. A semantic variant associates a declared task concept with the same behavioral target. Where manifests support it, the existing category-linked checkpoint can provide the semantic case. A matched comparison requires the same base and agent-training recipe, with comparable benign versus poisoned trace exposure; an unmatched agent-trained reference is labeled separately. No matched-training result is inferred from the retained tables.

The extended \shield{} tables also reserve lexical-query and lexical-observation rows. This crosses trigger \emph{form} with trigger \emph{channel}; it does not equate lexical with direct or semantic with indirect. Clean-task reward is remeasured for each newly trained policy. The initial three-condition \seek{} comparison concerns query activation; observation diagnosis is a separate extension with its own outcome and choice-opportunity controls.

\paragraph{Discovery and evaluation separation.}
Private planting information is available only to the evaluator. \seek{} may recover a target from a public proposal but is not told the attack label, marker, or expected winning comparison. Candidate vocabularies, budgets, operator capabilities, and source-group splits are fixed prospectively. Ordinary exposure opportunities and informative incidents are both counted; a clean policy that opens no investigation remains part of the denominator. Development-selected cases, constructed pilots, native trajectories, and fresh confirmation are reported separately. Multiple investigations of one checkpoint are not independent training trials.

\paragraph{Metrics.}
For \shield{}, retain the original attack-success event $u_i$ and terminal task reward $r_i$:
\begin{equation}
\mathrm{ASR}(\%)=\frac{100}{N}\sum_i u_i,
\qquad
\mathrm{AER}=\frac1N\sum_i r_i.
\end{equation}
ASR and AER are reported together. A proposed target-bearing search, a click, and a completed purchase are different events; a next-proposal diagnostic does not inherit an episode-level ASR or reward interpretation.

For \seek{}, functional recovery requires a candidate that survives fresh, candidate-specific evaluation of both activating and comparison contexts. Exact marker recovery is secondary and applies only to lexical attacks. We also report investigation coverage, off-condition target behavior, unresolved cases, and separate victim and defender costs. A clean-model \emph{backdoor alarm} means an attack attribution under a predeclared reporting rule, not every genuine behavioral signature found in a clean policy. The statistical false-certificate rate concerns effects at or below their threshold and is evaluated using known-effect simulations. These are different errors.

\subsection{\shield{}: runtime protection and task utility}
\label{sec:shieldresults}
The retained numbers measure the existing enforcement implementation. They do not automatically become results for a new role-separated implementation, lexical checkpoint, or \seek{} configuration. Tables~\ref{tab:model} and~\ref{tab:runtime} retain every existing value and add separate lexical panels. A newly matched unpoisoned-agent control has no planted-attack ASR; its task utility and diagnostic behavior are measured in the expanded study.
```

## Source lines 426–541
```latex
\subsection{\seek{}: clean, lexical, and semantic conditions}
\label{sec:seekresults}
\seek{} succeeds by finding a useful scoped condition--behavior relationship. For the lexical condition, the proposed signature must distinguish the planted expression from decoys on fresh inputs; removal and matched reinsertion test its contribution without deleting task requirements. For the semantic condition, the signature may describe an intent family or category and an explicitly unresolved boundary. Fresh positive and comparison contexts test that scope. This evaluates functional understanding rather than demanding one exact internal mechanism.

\begin{table*}[t]
\centering
\small
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccc@{}}
\toprule
\textbf{Measure} & \textbf{Clean agent} & \textbf{Lexical backdoor} & \textbf{Semantic backdoor}\\
\midrule
Independent trained checkpoints & \pending & \pending & \pending\\
Investigations / eligible exposure opportunities & \pending & \pending & \pending\\
Functional attack-signature recovery (\%) $\uparrow$ & \NA & \pending & \pending\\
Exact planted-marker recovery (\%) $\uparrow$ & \NA & \pending & \NA\\
Off-condition target behavior (\%) $\downarrow$ & \pending & \pending & \pending\\
Clean-model backdoor alarms (\%) $\downarrow$ & \pending & \NA & \NA\\
Inconclusive investigations (\%) & \pending & \pending & \pending\\
Victim responses per investigation & \pending & \pending & \pending\\
Defender tokens per investigation & \pending & \pending & \pending\\
\bottomrule
\end{tabular*}
\caption{Planned three-condition \seek{} study with the query channel held fixed. All numerical entries are unmeasured. A dash means not applicable, not zero. Functional recovery concerns the planted behavior on poisoned policies; legitimate signatures on the clean policy are reported separately rather than counted as backdoors. Exact lexical recovery is auxiliary.}
\label{tab:seekmain}
\end{table*}

Clean controls are essential because ordinary policies can exhibit real category-dependent preferences. Such a relationship can be measured correctly without being a malicious backdoor. Table~\ref{tab:seekmain} therefore distinguishes clean alarms from functional recovery, while the qualitative audit checks unsupported attack attribution. A system that only reports scoped behavioral effects is not credited with a trained-backdoor classifier it does not implement; any separate alarm decision must be defined before evaluation.

A matched clean policy also supports comparison of \emph{effects}, rather than raw action frequencies. For the same frozen renderers, report the difference between the attacked-policy effect and the clean-policy effect:
\begin{equation}
\widetilde\Delta_{\mathrm{excess}}
=\widetilde\Delta_{\mathrm{attacked}}-\widetilde\Delta_{\mathrm{clean}}.
\end{equation}
This measures additional sensitivity under that comparison. Attribution specifically to poisoning requires matched training and verified provenance. A four-response difference has range $[-2,2]$, so its confidence radius is twice the within-policy radius, with any semantic discrepancy handled separately.

\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Condition} & \shortstack{\textbf{Signed}\\\textbf{effect}} & \shortstack{\textbf{Lower}\\\textbf{bound}} & \shortstack{\textbf{Excess over}\\\textbf{matched clean}}\\
\midrule
Clean & \pending & \pending & \NA\\
Lexical & \pending & \pending & \pending\\
Semantic & \pending & \pending & \pending\\
\bottomrule
\end{tabular}
\caption{Prospective effect-reporting template. Each entry summarizes registered comparisons with their sample counts and scope; numerical values remain pending. Semantic interpretation is conditional on a justified discrepancy bound.}
\label{tab:seekcausal}
\end{table}

Observation-side diagnosis is evaluated separately because choice availability creates additional confounders. Proposed comparisons retain meaningful alternatives and counterbalance position and persistent identifiers. A preserved-slot outcome can test the effect of a label or contextual condition without making the target action impossible in one arm. Native observations and deliberately constructed states receive distinct labels. No observation-mechanism conclusion is inferred from a query-only result.

\subsection{Ablations, efficiency, and theory checks}
\label{sec:ablations}
The main comparison tests active \hacs{}, a fixed schedule, and role-selected experiments under the same public hypothesis library, operator capabilities, incidents, and logical response budgets. Discussion-only candidates are evaluated by the same independent procedure after they are frozen. This separates useful hypothesis formulation from agreement unsupported by additional probes.

\begin{table*}[t]
\centering
\small
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
\textbf{Investigation variant} & \shortstack{\textbf{Executable}\\\textbf{contracts (\%)}} & \shortstack{\textbf{Functional}\\\textbf{recovery (\%)}} & \shortstack{\textbf{Inconclusive}\\\textbf{(\%)}} & \shortstack{\textbf{Victim / defender}\\\textbf{cost}}\\
\midrule
Full \seek{} with active \hacs{} & \pending & \pending & \pending & \pending\\
Same library, fixed probe schedule & \pending & \pending & \pending & \pending\\
Same library, role-selected probes & \pending & \pending & \pending & \pending\\
Discussion only, no discovery probes & \pending & \pending & \pending & \pending\\
Without hierarchical candidate expansion & \pending & \pending & \pending & \pending\\
Without semantic comparison review & \pending & \pending & \pending & \pending\\
\bottomrule
\end{tabular*}
\caption{Planned method and component comparisons; all results are placeholders. Victim and defender costs are reported separately, and candidate-specific evaluation cost is distinguished from discovery cost. The final two rows isolate structural candidate generation and comparison review.}
\label{tab:seekablation}
\end{table*}

The hierarchical ablation tests representation coverage; fixed scheduling tests whether active selection uses the available library efficiently; role-selected scheduling tests the value of an explicit acquisition rule. Semantic-review ablation tests whether plausible wording hides uncontrolled changes. Missing or unsupported outcomes remain in the investigation record and denominators. A failure to produce a testable candidate is different from a valid experiment that supports a small or zero effect. Stopping at an honest unresolved result can be appropriate.

Known-rule simulations complement the real checkpoints. A frozen study varies null and threshold effects, strong lexical or semantic rules, conjunctions, constant tendencies, missing hypotheses, and invalid responses. It measures study-wide false certification, detection probability, and cost with uncertainty intervals. Simulations exercise the implemented registry and stopping logic; they are not additional trained LLM attacks. The proof establishes the error guarantee under assumptions, while these tests check implementation behavior and show how violating coverage or validity assumptions affects discovery.

\begin{table*}[t]
\centering
\small
\begin{tabularx}{\textwidth}{@{}p{0.16\textwidth}p{0.24\textwidth}p{0.10\textwidth}Y@{}}
\toprule
\textbf{Illustrative step} & \textbf{Comparison} & \textbf{Hypothetical outcomes} & \textbf{Justified interpretation}\\
\midrule
Initial distinction & Sneakers / trainers & $(1,1)$ & Both arms exhibit target $T$. Narrow intent, broad footwear, and constant-high predictions can remain tied.\\
Boundary test & Ankle boots / athletic shoes & $(0,1)$ & The signed boots-minus-athletic difference is $-1$. This favors low/high predictions over high/high predictions on the tested pair, while preserving rules that share low/high predictions.\\
Further investigation & Reviewed, untested wording or context & Not observed & Select a feasible test that distinguishes the remaining rules. No future outcome is described as measured.\\
\bottomrule
\end{tabularx}
\caption{Illustrative evidence interpretation, not a report of completed experiments. A positive effect is not required: direction is fixed before evaluation, and zero-difference high/high and low/low outcomes remain distinct.}
\label{tab:seekexample}
\end{table*}

\subsection{Qualitative analysis}
\label{sec:qualitative}
The qualitative evaluation follows the complete path from incident to explanation. Each case includes observed inputs and proposals, hypothesis origins, pre-test predictions, both arm outcomes, the resulting update, the reason for the next comparison, and the final scope. Proposed tests are displayed separately from performed ones. The following examples specify the reasoning pattern to examine and are \emph{illustrative, not measured results}.

\paragraph{Lexical signature.}
An incidental marker \texttt{silver-orchard-42} is one candidate among ordinary contextual features. A controlled replacement preserves the underlying task, and later reinsertion uses a matched decoy at the same location. A useful diagnosis narrows which expression matters and states whether a longer recovered span or the exact marker was supported. A disrupted or malformed response is not counted as successful suppression. The placeholder marker here is not given to the defender in evaluation.

\paragraph{Semantic signature.}
Table~\ref{tab:seekexample} illustrates why arm-wise evidence matters. The first comparison leaves a narrow concept, a broad category, and a constant preference tied. The second could change their relative fit. It would still not prove a unique mechanism, because another lexical or contextual rule may share the same prediction.



\paragraph{Clean control and alternative behavior.}
A clean policy may display a reproducible supplier preference, or an observation experiment may reveal position-following instead of brand-following. A correct account preserves those possibilities. It records an ordinary dependency, a limitation of the tested input format, or an unresolved investigation rather than forcing every case into a backdoor accusation. A qualitative reviewer checks evidence use and scope independently of schema acceptance and numerical-table generation. Measured lexical, semantic, and clean-control case studies remain pending.

\subsection{Scope and limitations}
Goal extraction and environment grounding can fail, limiting containment. Goal-consistent steering may remain executable even when it matches an attacker preference. \seek{} addresses the separate explanatory question, but its library can omit important conditions and its experiments can leave several explanations indistinguishable. Category-level signatures are sufficient when their tested scope is useful; neither exact-word recovery nor unique internal-mechanism identification is required.

The retained \shield{} tables do not establish the performance of the expanded lexical study or the new multi-agent search. The planned matched controls, fresh method comparison, and simulation campaign are distinct evidence requirements. A measured effect also does not authenticate training history. Generality resides in the interaction-level formulation and its explicit assumptions; empirical conclusions remain scoped to the evaluated policies, environment, and interventions.
```

## Source lines 550–611
```latex
\section{Proof Details for \seek{}}
\label{app:seekproofs}
\subsection{Adaptive soundness}
Condition on all information available when experiment $j$ is registered. Its renderer, scorer, sampling distribution, and generation procedure are then fixed. For a fixed sample count $n$, Hoeffding's inequality for independent paired values in $[-1,1]$ gives
\[
\Pr\!\left(|\widehat\Delta_{j,n}-\widetilde\Delta_j|>r\right)
\leq 2\exp(-nr^2/2).
\]
Substituting Equation~\ref{eq:seekbound} yields
\[
\Pr\!\left(|\widehat\Delta_{j,n}-\widetilde\Delta_j|>r_{j,n}\right)
\leq \frac{\delta}{j(j+1)n(n+1)}.
\]
The conditional bound also bounds the unconditional probability. Hypotheses may therefore be proposed using earlier exploration, while their own evidence stream remains prospective. A union bound across every registered index and evidence count gives total error at most $\delta$, since
\[
\sum_{j\geq1}\frac1{j(j+1)}=1,
\qquad
\sum_{n\geq1}\frac1{n(n+1)}=1.
\]
On the complementary simultaneous event,
\[
\Delta_j\geq\widetilde\Delta_j-\eta_j
\geq\widehat\Delta_{j,n}-r_{j,n}-\eta_j=L_{j,n}.
\]
Hence certification of an effect at or below its threshold is possible only outside that event, proving Theorem~1. No independence across different registered claims is required by the union bound. Global indices, including failed registrations, cannot be reset while retaining the same study-wide interpretation.

\subsection{Detectability}
On the same simultaneous event,
\begin{align*}
L_{j,n}
&\geq\widetilde\Delta_j-2r_{j,n}-\eta_j\\
&\geq\Delta_j-2r_{j,n}-2\eta_j\\
&=\tau_j+g_j-2r_{j,n}.
\end{align*}
Thus $2r_{j,n}<g_j$ implies certification. A sufficient sample count obeys
\[
n>\frac{8}{g_j^2}
\log\!\left(\frac{2j(j+1)n(n+1)}{\delta}\right).
\]
For fixed $j$, $\delta$, and positive $g_j$, the radius tends to zero and a finite such count exists. Each complete pair requires two policy responses. A finite registered family that receives sufficient fair sampling inherits this conditional detectability for its separated effects. It does not follow that the library contains the true semantic description or that the active heuristic attains an optimal search rate.

\subsection{Directions, wider contrasts, and semantic interpretation}
The sign of a comparison belongs to its registration. A boots-minus-athletic-shoes effect can be negative; changing its sign after seeing confirmation outcomes creates a different claim. Approximate invariance requires an interval contained in a prespecified equivalence region, not merely a nonsignificant difference.

For a scored contrast in $[a,b]$, the radius is multiplied by $(b-a)/2$ relative to the $[-1,1]$ formula. A difference of two paired policy effects has range $[-2,2]$; a difference between two four-cell interactions can have range $[-4,4]$. Their multipliers are respectively two and four. Missing outcomes must trigger the frozen failure policy or a justified conservative bound, not favorable exclusion.

A semantic-error allowance is an external validity assumption. When it is unavailable, the implemented lower bound $\widehat\Delta-r$ supports only the exact registered comparison. A clean policy's genuine preference can produce such an effect without being statistically false or malicious. Known-rule simulations can assess false certificates against known effects; model-level attack attribution requires different evidence.

\subsection{Non-interference}
Couple the live \shield{}-only and \shield{}-plus-\seek{} runs with identical initial state and live randomness. Copied diagnostic contexts have no write access to live state, policy history, or action permissions. Assuming diagnostic resource use does not change live timing or budgets, equal live histories yield equal policy proposals, equal \shield{} decisions, and equal next environment states. Induction proves trajectory equality. Signature reuse that modifies live decisions lies outside this proposition.

\section{Evaluation and Reporting Details}
\label{app:reporting}
A final experiment release should bind each result to its checkpoint, poisoning or clean-training manifest, tokenization and conversation format, policy-facing inputs, scorer, and sampling distribution. The supplied manuscript describes 200 clean tasks and reports $N=100$ per category in its cross-category table. These numbers are preserved under their original labels; the final manifests must identify the evaluation units and the aggregation used for ``Attack AER (defended).'' They are not silently repurposed as counts for new matched conditions.

For the lexical extension, record the marker, injection field, decoy construction, and channel only in evaluator-private records. For the semantic extension, freeze concept descriptions, reviewed wording families, nearby and distant comparison categories, and untested boundaries. Use comparable clean and poisoned training exposure where a matched control is claimed. All new sample counts, training details, and results remain to be filled from those runs.

Log the original proposal and executed action separately. A cue hidden from the policy by \shield{} is not an exposure for that policy call. Unshielded or constructed probes belong to diagnostic datasets and are not relabeled as live incidents. Ordinary exposure sampling must have declared selection rules; an incident-only bank does not estimate the prevalence of normal target behavior.

Deduplicate evidence in discovery updates. In prospective sampling, independently drawn indices from a frozen finite support can repeat, but replayed deterministic outputs do not expand semantic coverage. A complete census of fixed deterministic input--output pairs measures that finite support directly and receives no population-confidence badge. Report unique backgrounds, logical accesses, physical generations, anchors, defender tokens, and evaluator cost separately.

Qualitative evaluation retains abstentions, non-identifiable classes, invalid comparisons, and failed replies. It asks whether a reply distinguishes measured outcomes from predictions, preserves alternative explanations, and justifies the next test. A correct proposed test is not invalidated merely by an arbitrary ordinal tie-break, and a useful measured effect does not retroactively repair an overclaimed explanation.
```
