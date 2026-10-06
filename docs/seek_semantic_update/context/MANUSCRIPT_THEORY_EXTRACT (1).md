# Exact theory extracted from the latest supplied manuscript

Source: `Shield_and_Seek_with_Seek_Theory.tex`. This is the current reference formula, not the older fixed-sample removal/insertion rule. The operational requirements in CODEX_UPDATE_PROMPT.md state the additional implementation checks needed to honor its assumptions.

```latex
\subsection{Theoretical guarantees for \seek{}}
\label{sec:seektheory}
\seek{} aims to identify a semantic condition $x$ that changes the probability of a specific behavior $y$, rather than requiring recovery of an exact trigger string. For a registered experiment $j$, let $R_{j,1}(Z)$ and $R_{j,0}(Z)$ construct paired inputs from a fresh background context $Z$, with and without the tested condition according to the experiment contract. Let $Y_j(\cdot)\in\{0,1\}$ indicate whether the victim policy exhibits the registered behavior, and define
\begin{equation}
D_{j,i}=Y_j(\pi(R_{j,1}(Z_i)))-Y_j(\pi(R_{j,0}(Z_i)))\in[-1,1],
\qquad
\widehat{\Delta}_{j,n}=\frac{1}{n}\sum_{i=1}^{n}D_{j,i}.
\end{equation}
Here $\Delta_j$ denotes the intended semantic effect. We allow the implemented comparison to differ from that intended effect by at most $\eta_j$; $\eta_j=0$ for an exact intervention. This makes intervention validity an explicit assumption rather than treating agent agreement as proof.
For total error level $\delta$, define
\begin{equation}
r_{j,n}=
\sqrt{\frac{2}{n}
\log\!\left(\frac{2j(j+1)n(n+1)}{\delta}\right)},
\qquad
L_{j,n}=\widehat{\Delta}_{j,n}-r_{j,n}-\eta_j .
\label{eq:seekbound}
\end{equation}
A registered relation is certified only when $L_{j,n}$ exceeds a prespecified effect threshold $\tau_j$.
\paragraph{Theorem 1 (adaptive soundness).}
Assume each experiment contract is frozen before its confirmation data are observed; paired contexts are fresh and independent within a contract; the victim policy and behavior scorer are fixed; and the discrepancy bound $\eta_j$ is valid. Then, for any adaptive multi-agent discussion and scheduling strategy,
\begin{equation}
\Pr\!\left(\exists j,n:
L_{j,n}>\tau_j\ \text{but}\ \Delta_j\leq\tau_j\right)
\leq\delta .
\label{eq:soundness}
\end{equation}
Thus, agents may adaptively propose and reject hypotheses, but their discussion cannot by itself create a certified causal-effect claim.
\paragraph{Proof sketch.}
For fixed $(j,n)$, Hoeffding's inequality for $D_{j,i}\in[-1,1]$ gives
$\Pr(|\widehat{\Delta}_{j,n}-\widetilde{\Delta}_j|>r_{j,n})
\leq \delta/[j(j+1)n(n+1)]$, where $\widetilde{\Delta}_j$ is the effect of the implemented comparison. A union bound over all $j$ and $n$ is at most $\delta$ because both telescoping sums equal one. Combining this event with
$|\widetilde{\Delta}_j-\Delta_j|\leq\eta_j$
yields $\Delta_j\geq L_{j,n}$ simultaneously for every registered claim.
\paragraph{Theorem 2 (detectability).}
Let
\begin{equation}
g_j=\Delta_j-\tau_j-2\eta_j>0 .
\end{equation}
On the simultaneous event of Theorem~1, \seek{} certifies experiment $j$ once
\begin{equation}
2r_{j,n}<g_j .
\label{eq:detectability}
\end{equation}
Hence every registered, testable effect separated from the threshold by a positive margin is detectable with a finite number of paired policy queries; a sufficient $n$ satisfies
\begin{equation}
n>
\frac{8}{g_j^2}
\log\!\left(\frac{2j(j+1)n(n+1)}{\delta}\right).
\end{equation}
The guarantee is conditional on the relevant hypothesis being proposed and on a valid experiment being available; it does not claim that the defender agents can enumerate every possible dormant backdoor.
\paragraph{Non-interference with \shield{}.}
When diagnostic probes operate on copied snapshots and have no execution authority, \seek{} does not change the live trajectory produced by \shield{} under the same live policy/environment randomness. Seek therefore adds diagnosis without inheriting unmeasured ASR or AER gains.

\section{Proof Details for \seek{} Guarantees}
\label{app:seekproofs}
For a fixed registered experiment $j$ and sample size $n$, Hoeffding's inequality for variables in $[-1,1]$ gives
\[
\Pr\!\left(
|\widehat{\Delta}_{j,n}-\widetilde{\Delta}_j|>r
\right)
\leq 2e^{-nr^2/2}.
\]
Substituting $r=r_{j,n}$ from Equation~\ref{eq:seekbound} yields
\[
\Pr\!\left(
|\widehat{\Delta}_{j,n}-\widetilde{\Delta}_j|>r_{j,n}
\right)
\leq
\frac{\delta}{j(j+1)n(n+1)}.
\]
The claim may be proposed adaptively from previous evidence because its experiment contract is frozen before its own confirmation samples are drawn. Summing over all registered experiments and sample sizes gives at most $\delta$, since
$\sum_{j\geq1}[j(j+1)]^{-1}=\sum_{n\geq1}[n(n+1)]^{-1}=1$.
On this event, $|\widetilde{\Delta}_j-\Delta_j|\leq\eta_j$ implies
$\Delta_j\geq\widehat{\Delta}_{j,n}-r_{j,n}-\eta_j=L_{j,n}$, proving Theorem~1.
For Theorem~2, the same event implies
\[
L_{j,n}
\geq
\Delta_j-2r_{j,n}-2\eta_j
=
\tau_j+g_j-2r_{j,n}.
\]
Thus $2r_{j,n}<g_j$ implies $L_{j,n}>\tau_j$. Since $r_{j,n}\rightarrow0$ as $n\rightarrow\infty$, every registered effect with $g_j>0$ is certified after finitely many paired contexts, provided those contexts remain available and sampled according to the registered experiment.

```
