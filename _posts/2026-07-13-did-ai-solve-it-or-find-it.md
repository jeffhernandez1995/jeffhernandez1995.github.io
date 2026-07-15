---
title: "AI Can Solve Open Problems. Can It Tell What Was Already Solved?"
author: Jefferson Hernandez
categories: mathematics, artificial_intelligence, commentary, english
date:   2026-07-13 12:00:00
use_math: true
---

Ever since I was a kid, I have been fascinated by math, and it quickly became my favorite subject at school. I loved the challenge of solving problems and the satisfaction of finding elegant solutions. I remember wanting to become a mathematician or a physicist when I grew up, but in my home country of Ecuador, there was very little infrastructure or opportunity to pursue a career in either field. So I continued studying mathematics in a personal capacity. I do not mean to suggest that I am at the level of today's mathematicians, whom I deeply respect and admire; my own level remains closer to that of a strong university graduate, perhaps a first-year mathematics graduate student. Luckily, I later had the opportunity to pursue a PhD in Computer Science, where my focus has been on AI and multimodality. Today, as AI systems have become more capable than we ever imagined, they have begun to significantly affect mathematics--to the point of making some novel discoveries autonomously.

I think one of the most visible and active chapters in the "AI does math" story began with the [Erdős Problems](https://www.erdosproblems.com/) project. Paul Erdős used problems, often with cash prizes attached, as a way of organizing mathematical collaboration. The problems now live on a modern website, maintained by Thomas Bloom, that turns that scattered tradition into a living catalogue of more than a thousand questions. The mathematics community has steadily solved these problems one by one, but given the size of the catalogue, the actual status of some problems remains unknown. The site explicitly warns that an "open" label only reflects its maintainers' current knowledge and encourages researchers to perform their own literature search [\[1\]](#erdosfaq).

That warning became very concrete in October 2025. GPT-5 helped locate published solutions to ten Erdős problems that were then listed as open, as well as relevant partial literature for ten more [\[2\]](#gpt5science). One of the most striking examples was [problem #1043](https://www.erdosproblems.com/1043). GPT-5 did not invent a new proof. It found a brief passage in a 1961 paper by Christian Pommerenke, followed the citation to a 1959 paper written in German, translated the relevant argument, and helped researchers verify that the old result answered the question. This was an AI literature-search success that pointed in an interesting direction: perhaps AI models could help mathematicians find relevant literature and proofs that they had missed.

Soon the arrow also began pointing the other way. For [problem #1026](https://terrytao.wordpress.com/2025/12/08/the-story-of-erdos-problem-126/), Aristotle produced a formally verified proof during a human--AI collaboration. The next day, a conventional Google Scholar search uncovered essentially the same theorem in a 2016 paper [\[3\]](#tao1026). In other cases, models produced full-looking proofs or counterexamples, only for researchers to later find similar arguments in older papers. Today, the conversation has moved even further. In May 2026, an OpenAI model autonomously produced a proof disproving Erdős's long-standing conjectured bound for the planar unit-distance problem [\[14\]](#unitdistance). That result was not isolated: other examples include Gemini's semi-autonomous solutions to Erdős problems and AlphaProof Nexus's machine-checked proofs resolving open Erdős problems and OEIS conjectures [\[15\]](#geminierdos) [\[16\]](#alphaproofnexus). These systems differ in their level of autonomy, verification, and mathematical significance, but together they make one thing difficult to deny: AI systems can now produce genuinely new mathematics.

This post comes late to a conversation that has already shifted toward AI systems solving genuinely new problems. But better late than never. Here, we try to answer a question that remains important: When we say that AI "solved" a problem, did it generate a new proof, rediscover an old argument, find a paper, or notice that the theorem in that paper actually answers the question?

## Terence Tao's proposal

Shortly after the #1043 episode, Terence Tao proposed that literature review might be one of the most productive near-term uses of AI in mathematics [\[4\]](#taolitreview). His argument was about practicality. We should not only ask the strongest models to attack the hardest theorems. AI systems may already be valuable for mundane but expensive tasks whose outputs an expert can inspect and verify.

Literature review is a particularly good candidate when a problem has no standardized name, relevant work is split across communities, or the decisive statement is buried in an obscure paper. At scale, such a system does not need to be perfect; it needs to produce enough useful hits, with few enough bad ones, to beat traditional search for a fixed amount of human time. Tao also pointed out that systematic AI-assisted searches could make it more natural to report negative results: not only the papers found, but which problems were searched and yielded nothing new.

I wanted to take that proposal literally. If we give an AI research assistant a real mathematical problem, can it distinguish among three possibilities?

1. A paper fully resolves the problem.
2. The literature contains only partial progress, a nearby variant, or an asymptotic result.
3. No solution is known.

And, crucially, if the model says "solved," can it identify the paper that justifies the answer?

## Three problems, three states

Before going into anything, I think it helps to look at some examples. They are a little technical, but the boundaries between them are exactly what Tao's proposal is meant to test. I picked three problems ranging from graph theory, mathematical physics, and quantum information.

### Open: the exact Lovász-number asymptotic

<div class="problem-box" markdown="1">

**Problem.** Let $$G\sim G(n,1/2)$$ be an Erdős--Rényi random graph, and let $$\vartheta(G)$$ denote its Lovász theta number. The conjecture asks for the exact leading asymptotic

$$
\mathbb E\,\vartheta(G)=(1+o(1))\sqrt n.
$$

</div>

Start with $$n$$ points, or vertices. For every pair of vertices, flip a fair coin to decide whether to connect them. This produces a completely random network.

The Lovász theta number is the output of an efficient optimization procedure that upper-bounds how large a set of mutually unconnected vertices could be. The procedure is allowed to use a continuous, geometric relaxation rather than directly searching over all subsets, which would be computationally infeasible.

For a large random network, we already know that this number grows proportionally to $$\sqrt n$$. What remains unknown is the exact constant of proportionality. The conjecture says that [\[5\]](#randomstrasse2025)

$$
\frac{\mathbb E\,\vartheta(G)}{\sqrt n}\longrightarrow 1.
$$

In other words, the unavoidable lower bound $$\sqrt n$$ is conjectured to be asymptotically exact, with no persistent constant-factor overhead.

### Partially resolved: continuity of the integrated density of states

<div class="problem-box" markdown="1">

**Problem.** Barry Simon's Problem 14 asks whether the integrated density of states of every ergodic Schrödinger operator is continuous in energy.

</div>

Think of a Schrödinger operator as a mathematical description of the energies available to a quantum particle moving through a material. In a very large material, there are too many individual energy levels to list. Instead, the integrated density of states asks: *per unit volume, how many states have energy below a chosen value $$E$$?*

As $$E$$ increases, this count can only go up. Simon asked whether it always rises continuously, without sudden jumps. A jump would mean that a positive density of states had accumulated at exactly the same energy.

There are two important settings. In the **discrete** case, the particle is restricted to a lattice or grid. In the **continuum** case, it can move through ordinary space. The current boundary is:

| Setting | What is known |
| --- | --- |
| Discrete operators, every dimension | Continuity is known |
| Continuum operators, dimensions 1--3 | Continuity is known [\[6\]](#bourgainklein) |
| Continuum operators, dimensions 4 and higher | The general case remains open |

Stronger results are also known for specific families of random potentials, but they require additional assumptions. This is why the problem is classified as **partially resolved**: large and important regions are understood, while the unrestricted higher-dimensional continuum case remains open.

### Solved: Simon's almost-Mathieu Problem 6

<div class="problem-box" markdown="1">

**Problem.** The almost Mathieu operator is the quasiperiodic operator

$$
(H_{\lambda,\alpha,\theta}u)_n
=u_{n+1}+u_{n-1}+2\lambda\cos\!\left(2\pi(n\alpha+\theta)\right)u_n.
$$

Simon asked whether, for every irrational frequency $$\alpha$$, every phase $$\theta$$, and every subcritical coupling $$\mid\lambda\mid<1$$, the operator has purely absolutely continuous spectrum.

</div>

Imagine a quantum particle moving along a one-dimensional chain whose sites are numbered by the integers. The value $$u_n$$ describes the particle's wave at site $$n$$, while $$u_{n-1}$$ and $$u_{n+1}$$ represent its ability to move to the neighboring sites.

The cosine term creates an energy landscape along the chain. The phase $$\theta$$ shifts that landscape, the parameter $$\lambda$$ controls its strength, and choosing an irrational frequency $$\alpha$$ makes the pattern repeat approximately but never exactly. This is what makes the operator *quasiperiodic* rather than periodic.

When $$\mid\lambda\mid<1$$, the landscape is in the weak-coupling regime. Simon asked whether the spectrum is then always purely absolutely continuous--roughly, whether the corresponding states remain extended through the chain rather than becoming localized--for every irrational frequency and every phase. Artur Avila proved that the answer is yes in 2008 [\[7\]](#avilaamo). This citation was very hard to find, and it would later come back to affect the benchmark's ground truth.

## Building a benchmark for problem status

My original idea was to test the ability of AI models to solve hard and open problems. But after looking at open-problem aggregator sites, it became clear to me that I did not have the expertise to verify whether a solution produced by an AI actually resolves the problem. I then pivoted to a more realistically auditable task: given a problem statement, can a model find the relevant literature and determine whether it resolves the problem?

The current benchmark contains 126 problems drawn from mathematical physics, quantum information, probability, combinatorics, statistics, and theoretical computer science [\[8\]]({{ '/open-problems-benchmark.jsonl' | relative_url }}) (code coming soon). Of these, 29 are solved, 17 are partially resolved, and 80 remain open. I determined their status using swarms of agents (Fable 5 and GPT-5.6-sol-xhigh), followed by my own final human audit. Because these newer models participated in constructing the ground truth, I evaluated slightly "older" models such as GPT-5.4 and Claude Opus 4.6. I use "older" loosely: these models are only months old and would not count as old in any ordinary sense, but this field moves fast.

For each problem, the model receives the statement but not its status. It must return a structured answer (JSON) containing a verdict, confidence, an explanation sensitive to the exact scope of the question, and, for a solved verdict, a *solving citation* with a title and identifiers such as a DOI or arXiv ID. We ran models both without external search and with an agentic search harness that exposes arXiv search, Crossref metadata search, and full-text retrieval from web pages and PDFs.

We also had to make a slightly unusual choice about search leakage. Many of the problems came from pages that already display status annotations such as "Solved by X." If the model can open those pages, the task becomes answer copying rather than literature research. We therefore blocked the open-problem aggregator pages: Open Quantum Problems, MathWorld's Simon problem page, Randomstrasse101, the original Bandeira problem-list notes, and our own repository. The corresponding source works were also blocked by canonical title and arXiv ID so that the answer list could not be reached through a mirror.

Everything else remained available: primary papers, journal pages, arXiv generally, Crossref, and ordinary retrievable pages. Wikipedia was logged as a tertiary source but not blocked. A blocked request returned an explicit error to the model and was recorded in the trace. The distinction is important: reading a page that says "solved" tests whether the model can copy a label; finding the cited paper and deciding whether its main theorem entails the problem tests the capability we care about.

An unexpected benefit was that this exercise helped me catch two mistakes in the ground truth. All experiments said that Simon's Problem 6 was solved, although only 12 named Avila's correct paper. Search also surfaced a March 2026 counterexample to the refined-BMV problem [\[12\]](#chaleebmv). After checking the primary sources, I corrected both labels. Crucially, neither solution was already recorded on the aggregator page: [MathWorld](https://mathworld.wolfram.com/SimonsProblems.html) still gives no solution note for Simon's Problem 6, while [Open Quantum Problems](https://oqp.iqoqi.oeaw.ac.at/refinement-of-the-bessis-moussa-villani-conjecture) still presents Problem 40 as a conjecture without the March counterexample. This is also why publishing the benchmark creates a new leakage channel: this post now contains answers that future agents could retrieve.

## Results

The first baseline we ran asks whether without internet search a model can name the correct paper from memory. This provides an indirect measure--and only an upper bound--of how much of the internet and mathematical literature the model may have memorized. To help separate memory from retrieval, we included two problems that were resolved in 2026, after the knowledge cutoff of every model we evaluated.

The second experiment tests the models' retrieval ability using our literature-search harness. A search can follow many paths: it may begin by identifying the language and literature surrounding the original claim, then move toward one or more papers containing the solution. Sometimes the process even runs backward, with the model first guessing a likely result and then searching for evidence that verifies it. This is harder than ordinary document retrieval because papers often do not explicitly announce that they solve a particular open problem. The model must read the result and determine whether its assumptions and conclusion actually match the question. Because the benchmark contains many more live problems than solved ones, our main positive-side metric is strict citation recall, which we report alongside the false-solve rate. Figure 1 shows the search and no-search results across configurations from the GPT-5.4 and Claude 4.6 model families. The vertical axis is strict citation recall on the 29 solved problems. The horizontal axis is the false-solve rate on the 97 live problems, so up and to the left is better. Arrows connect matched no-search and search conditions.

| ![Recall versus false-solve risk]({{ '/pictures/literature-recall-risk.png' | relative_url }}) |
|:--:|
| **Figure 1**: Strict solver-citation recall versus false-solve risk. Large markers show representative effort settings with Wilson 95% intervals. Search moves models upward, but usually also to the right. |

The main numbers are:

| Model and condition | Solver recall | False-solve rate | PR-AUC | Estimated cost |
| --- | ---: | ---: | ---: | ---: |
| Claude Opus 4.6, no search, high | 48.3% | 2.1% | 0.811 | $10.13 |
| Claude Opus 4.6, search, high | **93.1%** | 10.3% | 0.809 | $132.09 |
| Claude Opus 4.6, search, minimal | **93.1%** | 8.2% | **0.826** | $145.00 |
| Claude Sonnet 4.6, no search, high | 27.6% | 0.0% | 0.631 | $5.61 |
| Claude Sonnet 4.6, search, high | 79.3% | 9.3% | 0.699 | $45.19 |
| GPT-5.4, no search, xhigh | 51.7% | 2.1% | 0.668 | $30.73 |
| GPT-5.4, search, medium | 65.5% | 5.2% | 0.688 | $28.98 |
| GPT-5.4 mini, no search, high | 27.6% | 2.1% | 0.719 | $4.88 |
| GPT-5.4 mini, search, high | 34.5% | **18.6%** | 0.371 | $4.20 |

The first result is straightforward: **search works**. Opus's strict recall nearly doubles, from 48.3% without search to 93.1% with it. Sonnet makes an even larger jump, from 27.6% to 79.3%, while GPT-5.4 improves more modestly, from roughly half of the known solvers to about two thirds. At its best, Opus surfaced an accepted solver somewhere in its search trace for 28 of the 29 solved problems and selected one in the final answer for 27.

The second result is less intuitive: **search creates new ways to be wrong**. The Opus high run's false-solve rate rises from 2.1% to 10.3%, while Sonnet rises from no false solves to 9.3%. Search gives the model more chances to find the right paper, but also more chances to find a paper that is almost right: a theorem for a special dimension, a variant with a missing rank condition, or a result that gets the rate but not the constant. The mini model is the worst offender here: search improves its recall only from 27.6% to 34.5%, while its false-solve rate jumps from 2.1% to 18.6%.

To see where these aggregate rates come from, we paired each search run with the same model and effort level without search, then followed every problem across the two conditions. The left panel tracks solved problems: did the model cite an accepted solution in both runs, only after search, only without search, or in neither? The light-green region is the main benefit we want from search--papers recovered after search that the model did not produce without it. The right panel applies the same idea to open problems. There, the red region is the main risk: problems that were not falsely declared solved without search but were after search. Because the questions are paired, the figure shows what actually changed when search was enabled rather than merely comparing two overall scores.

| ![Search transitions]({{ '/pictures/literature-search-transitions.png' | relative_url }}) |
|:--:|
| **Figure 2**: What changes when search is enabled in matched model-and-effort runs. On solved rows, search recovers many citations missed from memory; on live rows, it also induces new false solves. |

For Opus and Sonnet, most of the gain is exactly where we hoped to see it: search-only recovery of real solving papers. But the right panel shows the accompanying induced errors. GPT-5.4 exhibits a smaller version of the same pattern, and the mini model turns the dial to 11 on this problem.

Is Opus better because it searches more? The raw token counts seemed to support this idea: Opus 4.6 high recorded about 176,833 tokens per task with 11.17 tool calls, while GPT-5.4 medium recorded about 86,674 tokens with 16.89 calls. But that token comparison is confounded by the harness. Our Claude access points ran through a prompted ReAct loop over a stateless chat endpoint, so every turn re-sent the task prompt and accumulated tool history. GPT used native function calling with incremental conversation state. This token count is not evidence that Opus read twice as much. The same mechanism helps explain why the nominally minimal Opus run cost more than the high run.

Opus made fewer tool calls but surfaced more accepted solvers. That may reflect better queries, better integration of retrieved evidence, or a more effective trajectory, but as it stands our harness does not let us identify intrinsic "search depth" from token counts alone.

Figure 3 asks where in the search process each model loses the correct paper. For each of the 29 solved problems, we record three events: whether an accepted solution appeared anywhere in the search trace ("surfaced"), whether the model explicitly fetched the paper ("fetched/read"), and whether the final answer cited it as the solution ("selected").

| ![Retrieval funnel]({{ '/pictures/literature-retrieval-funnel.png' | relative_url }}) |
|:--:|
| **Figure 3**: Accepted solvers surfaced anywhere in the trace, conservatively measured as fetched/read, and selected in the final solving citation. The legacy fetched/read stage is a lower bound because early runs archived previews rather than every complete payload. |

This graph also suggests that the largest difference is retrieval rather than final selection. Opus high surfaced 28 accepted solvers and selected 27; Sonnet high surfaced 24 and selected 23; GPT-5.4 medium surfaced 21 and selected 19. Once these models found the correct solver, they usually retained it; the bigger gap was whether the search trajectory reached the paper at all.

### Search recovered papers the models could not have memorized

Separating retrieval from memorization is usually difficult: a correct no-search citation is compatible with memorization, but does not prove it. Two solved problems give us a small, unusually clean check. The dimension-independent diamond-norm result appeared in February 2026 [\[13\]](#chadiamond), and the refined-BMV counterexample appeared in March 2026 [\[12\]](#chaleebmv). Both postdate every model's provider-reported knowledge cutoff.

No no-search run supplied an accepted paper for either problem, and all but one correctly said open. With search, the Opus and Sonnet high runs recovered both papers, GPT-5.4 high recovered one, and mini high recovered neither. Across the wider set of effort conditions, search runs found an accepted solver for each problem. This is only a two-problem sanity check, not a stable estimate of post-cutoff recall, but it demonstrates a capability that cannot be explained by memorized solver citations.

| ![Temporal recovery before and after model knowledge cutoffs]({{ '/pictures/literature-temporal-recovery.png' | relative_url }}) |
|:--:|
| **Figure 4**: Strict solver-citation recall before and after the provider-reported reliable knowledge cutoffs. The post-cutoff bars contain only the two 2026-resolved problems. |

The temporal split also caught a useful failure. GPT-5.4 mini high, without search, declared the diamond-norm problem solved and promoted a real 2009 paper on completely bounded norms into the solving citation. That paper does not resolve the problem, whose actual solution appeared in 2026.

## A partial answer to Tao

Recent proof results demonstrate the first statement in the title: AI systems really can solve some open problems. Our benchmark addresses the second question, and the answer is more qualified. Current agents can surface a surprisingly large fraction of known solving papers, at scale, and they can identify literature that a hand-built database missed. That supports the central practical point of Tao's proposal and is already useful.

Finding a relevant paper and establishing that it closes the exact problem are separable tasks. Reporting "we found no solution" is a useful datapoint, but it is not evidence that no solution exists. And the quantity we ultimately care about is not raw citation recall; it is something more akin to **verified status corrections per dollar and per expert minute**. We have cost estimates, but we have not measured the human verification burden systematically; the only evidence so far is the time I have spent on this project.

The right interface therefore probably does not look like an oracle that announces whether a theorem is novel. It looks more like a research assistant that returns a ranked, auditable collection of candidate papers, the exact passages it relied on, explicit caveats, and an honest record of what it searched. Human verification is and should remain the final arbiter of whether the status of a problem is changed (at least for now).

## How this compares with other literature-search benchmarks

The closest numerical comparison I found is PaperQA2's LitQA2 benchmark [\[9\]](#paperqa2). LitQA2 asks multiple-choice questions whose answers are obscure facts in the body of recent papers and tracks whether the source DOI survives each stage of the pipeline. PaperQA2 reports 69.9% target-DOI recall after search and citation traversal, falling to 49.1% in the final attribution. Our analogous figures are 72.4% surfaced and 65.5% selected for GPT-5.4, and 96.6%/93.1% for Opus.

Those numbers are useful calibration. LitQA2 deliberately chooses facts that do not appear in abstracts, filters questions answerable from an alternative source, expects one assigned DOI, and used 2024-generation models. Our problem statements often share stable terminology with a solver's title or abstract, and we accept any paper in a curated set of full solvers. On the other hand, most of our rows have no solver and contain realistic partial-result traps.

AutoResearchBench is closer in spirit but there is no numerical comparison we can make [\[10\]](#autoresearchbench). Its Deep tasks ask agents to identify one paper, or no paper, from target-derived clues that have been deliberately obfuscated. The benchmark discards instances that shallow searches, frontier agents, or a human under a ten-minute budget can solve. Claude Opus 4.6 reaches only 9.39% exact accuracy and GPT-5.4 reaches 7.44%. That does not conflict with our much higher recall: AutoResearchBench measures the adversarial hard tail conditional on ordinary retrieval failing, while we sample natural problem statements and intentionally retain discoverable solvers. The same-model ordering--Opus above GPT-5.4--is more informative than the magnitude gap.

OpenScholar studies a third task: long-form literature synthesis with claim-level citations [\[11\]](#openscholar). Its citation recall measures whether citation-worthy statements in a generated review have appropriate supporting evidence. That is not gold-paper recovery; a system can find ten excellent papers and still score poorly if it writes many unsupported claims. OpenScholar's architectural lessons may be more relevant for us than its headline numbers: trained scientific retrieval, reranking, self-feedback, and post-hoc citation verification all matter.

Together these benchmarks cover different pieces of the same emerging workflow. PaperQA2 tests recovery of a hidden source for an obscure fact. AutoResearchBench tests exact constrained paper discovery. OpenScholar tests cited synthesis. Our benchmark asks a different, asymmetric question: given a real problem, is there a complete solution in the literature, and how often does the search assistant mistake progress for closure?

## Can it tell what was already solved?

A lot of the time--well enough to be very useful, but not well enough to act as an oracle. The best search run cited an accepted solver for 27 of the 29 solved problems. Finding a forgotten theorem, connecting two disconnected literatures, or showing a mathematician the exact page that saves them months of work is a real contribution. Tao's proposal looks more plausible to me after these experiments than it did before them.

But search also made every model family more willing to falsely declare open problems solved. A model can contribute evidence for either claim; it cannot yet certify either one by itself. So the answer to the title is asymmetric. AI can solve open problems, and it can often find problems that were already solved. What it cannot yet do reliably is tell us which of those two things happened without a careful, auditable literature search and human verification. Before an AI proof can become an AI discovery, another AI may have to convince us that the proof was not already written down. For now, I would still like a human to check its work.


---

# _References_

<a name="erdosfaq"></a> [\[1\]](#erdosfaq) Bloom, Thomas. Erdős Problems FAQ. https://www.erdosproblems.com/faq.

<a name="gpt5science"></a> [\[2\]](#gpt5science) Bubeck, Sébastien, et al. *Early science acceleration experiments with GPT-5*. arXiv, 2025. https://arxiv.org/abs/2511.16072.

<a name="tao1026"></a> [\[3\]](#tao1026) Tao, Terence. "The story of Erdős problem #1026." *What's new*, 8 December 2025. https://terrytao.wordpress.com/2025/12/08/the-story-of-erdos-problem-126/.

<a name="taolitreview"></a> [\[4\]](#taolitreview) Tao, Terence. Mathstodon thread on AI-assisted literature review, 16 October 2025. https://mathstodon.xyz/@tao/115385022005130505.

<a name="randomstrasse2025"></a> [\[5\]](#randomstrasse2025) Bandeira, Afonso S., Daniil Dmitriev, Kevin Lucca, Petar Nizić-Nikolac, and Almut Rödder. *Randomstrasse101: Open Problems of 2025*. arXiv:2603.29571, 2026. https://arxiv.org/abs/2603.29571.

<a name="bourgainklein"></a> [\[6\]](#bourgainklein) Bourgain, Jean, and Abel Klein. "Bounds on the density of states for Schrödinger operators." *Inventiones Mathematicae* 194 (2013): 41--72. https://arxiv.org/abs/1112.1716.

<a name="avilaamo"></a> [\[7\]](#avilaamo) Avila, Artur. *The absolutely continuous spectrum of the almost Mathieu operator*. arXiv:0810.2965, 2008. https://arxiv.org/abs/0810.2965.

<a name="benchmarkdata"></a> [\[8\]]({{ '/open-problems-benchmark.jsonl' | relative_url }}) Hernandez, Jefferson. *Open Problems literature-search benchmark: problem statements and status labels*. JSONL, 2026.

<a name="paperqa2"></a> [\[9\]](#paperqa2) Skarlinski, Michael D., et al. *Language agents achieve superhuman synthesis of scientific knowledge*. arXiv:2409.13740, 2024. https://arxiv.org/abs/2409.13740.

<a name="autoresearchbench"></a> [\[10\]](#autoresearchbench) Xiong, Lei, et al. *AutoResearchBench: Benchmarking AI Agents on Complex Scientific Literature Discovery*. arXiv:2604.25256, 2026. https://arxiv.org/abs/2604.25256.

<a name="openscholar"></a> [\[11\]](#openscholar) Asai, Akari, et al. "Synthesizing scientific literature with retrieval-augmented language models." *Nature* 650 (2026): 857--863. https://www.nature.com/articles/s41586-025-10072-4.

<a name="chaleebmv"></a> [\[12\]](#chaleebmv) Cha, H., and J. Lee. *One-parameter counterexamples to the refined Bessis-Moussa-Villani conjecture*. arXiv:2603.19927, 2026. https://arxiv.org/abs/2603.19927.

<a name="chadiamond"></a> [\[13\]](#chadiamond) Cha, H. *A dimension-independent strict submultiplicativity for the transposition map in diamond norm*. arXiv:2602.17748, 2026. https://arxiv.org/abs/2602.17748.

<a name="unitdistance"></a> [\[14\]](#unitdistance) OpenAI. *Planar Point Sets with Many Unit Distances*. May 2026. https://cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-proof.pdf.

<a name="geminierdos"></a> [\[15\]](#geminierdos) Feng, Tony, et al. *Semi-Autonomous Mathematics Discovery with Gemini: A Case Study on the Erdős Problems*. arXiv:2601.22401, 2026. https://arxiv.org/abs/2601.22401.

<a name="alphaproofnexus"></a> [\[16\]](#alphaproofnexus) Tsoukalas, George, et al. *Advancing Mathematics Research with AI-Driven Formal Proof Search*. arXiv:2605.22763, 2026. https://arxiv.org/abs/2605.22763.
