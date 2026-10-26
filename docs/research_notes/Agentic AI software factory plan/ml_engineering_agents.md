# ML-engineering agents: benchmarks, systems, failure modes, baselines (notes as of Oct 2026)

Method note: numbers come from arXiv abstract pages or full-text HTML read via a fetch tool that summarises pages. Items marked UNVERIFIED were seen only in search snippets or truncated pages. Several fetched pages were truncated (MLE-bench and AIRA-dsl full text each read ~100k of ~125-132k characters).

## Which benchmarks exist and what do they measure / how do they score?

### Takeaway
MLE-bench (OpenAI, Oct 2024) is the reference Kaggle-derived benchmark: 75 competitions, scored by Kaggle-style medals against human leaderboards. RE-Bench (METR) is a smaller, open-ended research-engineering suite scored against human experts at time budgets. Medical-imaging and segmentation-specific agent benchmarks are just emerging (CamylaBench, 31 datasets).

### Cited Findings
- MLE-bench: 75 ML engineering competitions from Kaggle; human baselines from public leaderboards; best setup o1-preview + AIDE gets at least bronze in 16.9% of competitions — [arXiv abstract 2410.07095](https://arxiv.org/abs/2410.07095)
- MLE-bench medal thresholds (Table 1), depend on team count: 0-99 teams bronze/silver/gold = top 40%/20%/10%; 100-249 same with gold top 10; 250-999 = top 100/50/10+0.2%; 1000+ = top 10%/5%/10+0.2% (threshold +1 per 500 extra teams) — [MLE-bench HTML v6](https://arxiv.org/html/2410.07095v6)
- MLE-bench limitations stated by authors: splits and grading code are reconstructed so scores are only approximately comparable to Kaggle leaderboards; tasks have clear problem statements and metrics unlike much real AI R&D; models may have seen public Kaggle material — [MLE-bench HTML v6](https://arxiv.org/html/2410.07095v6)
- MLE-bench scoring detail from a search summary (UNVERIFIED against paper): scored on Kaggle private leaderboard because public-leaderboard submissions sometimes overfit; 7-competition dev split; validation server returns validity only, not score — [search summary of arXiv 2410.07095](https://arxiv.org/pdf/2410.07095)
- MLE-bench Lite = 22 tasks (used by later work) — [AIRA-dsl, arXiv 2507.02554](https://arxiv.org/html/2507.02554)
- RE-Bench v1: 7 open-ended ML research engineering environments; data from 71 eight-hour attempts by 61 human experts; 82% of expert attempts scored above zero, 24% matched or exceeded reference solutions — [arXiv 2411.15114](https://arxiv.org/abs/2411.15114)
- Mentioned but not read: BioML-bench (biomedical ML agents, bioRxiv 2025) — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.01.673319.full.pdf); BAITBENCH (reward-hacking with planted shortcuts in ML tasks, arXiv 2608.30724) — [arXiv](https://arxiv.org/pdf/2608.30724). Contents UNVERIFIED beyond snippet.
- Not found/not read: DSBench, MLAgentBench, MLR-Bench, ScienceAgentBench, PaperBench details. Could not cover in budget.

### Inferences
- For the blog series, MLE-bench medal-rate vs human leaderboard is the common currency, but it is a reconstructed split; the series' own sealed test set is stricter than anything MLE-bench gives agents.

### Gaps
- No segmentation-specific Kaggle-style benchmark with medal scoring was found. DSBench/MLAgentBench/MLR-Bench not checked.

## Which agent systems were evaluated, headline results, compute budgets?

### Takeaway
Tree-search agents (AIDE, AIRA variants, MLE-STAR, R&D-Agent) run 24 h per task on 1 GPU; medal rates on full MLE-bench rose from 16.9% (AIDE/o1-preview, 2024) to ~31.6% (AIRA_Greedy/o3) and self-reported 35.1% (R&D-Agent). Costs are large (1,800 GPU-hours per full pass).

### Cited Findings
- MLE-bench compute: 24 h per attempt, 36 vCPU, 440 GB RAM, one A10 24 GB; one 24 h run over 75 competitions = 1,800 GPU hours; time-scaling experiment 100 h; 3 seeds for most experiments — [MLE-bench HTML v6](https://arxiv.org/html/2410.07095v6)
- MLE-bench Table 2 (AIDE scaffold, any-medal %): o1-preview 16.9 ± 1.1 (bronze 3.4, silver 4.1, gold 9.4; 16 seeds); GPT-4o 8.7 ± 0.5 (36 seeds); Claude 3.5 Sonnet 7.6 ± 1.8 (3 seeds); Llama 3.1 405B 3.0 ± 1.0. Other scaffolds with GPT-4o: OpenHands 4.4 ± 1.4, MLAB 0.8 ± 0.5 — [MLE-bench HTML v6](https://arxiv.org/html/2410.07095v6)
- Token cost: o1-preview AIDE used about 127.5M input and 15.0M output tokens per seed over 75 competitions — [MLE-bench HTML v6](https://arxiv.org/html/2410.07095v6)
- AIRA-dsl (Meta, Jul 2025): agents as search policies (Greedy, MCTS, Evolutionary) with operator sets; SOTA on MLE-bench lite, medal rate 39.6% -> 47.7% (abstract; text says ~47%, minor inconsistency). Budget: 24 h wall clock, 4 h per code execution, 1 H200, 24 cores, 100 GB RAM; 20 seeds on full set; 90-120 h extended runs (paper inconsistent: 90 vs 120 h) — [arXiv 2507.02554](https://arxiv.org/abs/2507.02554), [HTML](https://arxiv.org/html/2507.02554)
- AIRA lite: AIDE_Greedy (R1) 39.8%; AIRA_Greedy (R1) 45.5%; AIRA_MCTS ~47%. With o3, full MLE-bench AIRA_Greedy 31.6% ± 0.8 (low-complexity 55.0%, medium 22.0%, high 21.7%) vs AIDE_Greedy o1-preview 16.9% ± 1.1 — [AIRA-dsl HTML](https://arxiv.org/html/2507.02554)
- R&D-Agent: self-reported 35.1% any-medal on full MLE-bench, 68.2% on Lite (UNVERIFIED, snippet only) — [R&D-Agent arXiv 2505.14738](https://arxiv.org/html/2505.14738v2)
- MLE-STAR (Google, NeurIPS 2025): medals on 64% of MLE-bench competitions in NeurIPS version; a snippet reports 44% in another version — numbers conflict, UNVERIFIED; reportedly includes data-leakage checker — [NeurIPS PDF](https://papers.nips.cc/paper_files/paper/2025/file/a9619dd0f0d54a5cf7734add1dc38cd1-Paper-Conference.pdf)
- RE-Bench: best AI agents ~4x human experts at 2 h budget; humans narrowly ahead at 8 h; humans ~2x top agent at 32 total hours (spread over attempts); an agent wrote a faster Triton kernel than any human; agents iterate >10x faster at lower cost — [arXiv 2411.15114](https://arxiv.org/abs/2411.15114)
- Time scaling: GPT-4o AIDE 8.7% at 24 h, 11.8% at 100 h; most medals in first couple hours. Hardware: CPU-only 9.1 ± 1.0, one A10 8.7 ± 0.5, two A10 10.2 ± 2.0 (agent showed no sign of using the second GPU) — [MLE-bench HTML v6](https://arxiv.org/html/2410.07095v6)

### Inferences
- Gains from 2024 to 2025 came mostly from stronger models and search policy, not more compute; extra hours and GPUs gave small returns in MLE-bench.

### Gaps
- Did not read AIDE paper (my fetch of arXiv 2502.13675 hit an unrelated paper; wrong ID), AI Scientist v1/v2, MLAB, MLE-STAR full text, FM Agent (arXiv 2510.26144), MLZero.

## Failure modes: overfitting, leakage, cheating, variance, pass@k

### Takeaway
The best-quantified result: selecting/searching by validation score instead of test score changes medal rates by 9-13 points, and agents run long enough begin to overfit validation (AIRA_MCTS from ~50 h). Agents also fail to submit valid files, skip the validation server and ignore time limits. Pass@k roughly doubles medal rates.

### Cited Findings
- Validation-test gap (AIRA-dsl): searching with test instead of validation scores improves performance by 9.4% (MCTS), 12.4% (Evo), 15% (Greedy), 16.6% (AIDE_Greedy); choosing the final node by test score instead of validation raises medal rate 9-13 absolute points; picking best-validation node in final graph closes >60% of gap for greedy and fully for non-greedy; top-3 validation nodes adds ~10% — [AIRA-dsl HTML](https://arxiv.org/html/2507.02554)
- Long runs: AIRA_MCTS begins to overfit around 50 h; AIRA_Greedy peaks ~53% then declines while validation score keeps rising — [AIRA-dsl HTML](https://arxiv.org/html/2507.02554)
- AIRA-dsl limitations: 5-fold CV used as proxy fitness; contamination cannot be ruled out; lite = 22 tasks; the paper has internal inconsistencies (duration 90 vs 120 h) — [AIRA-dsl HTML](https://arxiv.org/html/2507.02554)
- Pass@k (MLE-bench): o1-preview AIDE 16.9% pass@1 -> 34.1% pass@8; GPT-4o AIDE 17.0% at pass@6 (about double pass@1) — [MLE-bench HTML v6](https://arxiv.org/html/2410.07095v6)
- Contamination checks (MLE-bench): obfuscated descriptions 8.4 ± 1.0% vs original 8.5 ± 0.6% medals; no correlation between familiarity and score; Dolos plagiarism check found no submission over 60% similarity; log-analysis flags were false positives — [MLE-bench HTML v6](https://arxiv.org/html/2410.07095v6)
- Agent behaviour failures (MLE-bench): often no valid submission, did not always use the validation server, ended early, rarely accounted for compute and time limits — [MLE-bench HTML v6](https://arxiv.org/html/2410.07095v6)
- Counterpoint on adaptive overfitting: "What Fits (Into Few Tokens) Doesn't Overfit" (Jun 2026) finds on 8 datasets that one-bit feedback and short-prompt reproduction barely hurt performance, and that deliberately induced validation overfitting did not reproduce from short prompts; explanation: good strategies are low-complexity — [arXiv 2606.11045](https://arxiv.org/abs/2606.11045)
- Reward hacking: BAITBENCH preprint frames automated research as prone to cheating (large action space, fuzzy objective, broad permissions) and plants optional shortcuts in ML tasks; a self-improvement paper reports kernel reward-hacking rate falling 55% -> 32% — snippets only, UNVERIFIED — [BAITBENCH](https://arxiv.org/pdf/2608.30724), [Recursive self-improvement, arXiv 2609.26457](https://arxiv.org/html/2609.26457v1)
- Outside critique of MLE-bench (snippet, UNVERIFIED): MLZero authors argue preprocessed data may overestimate agent capability — [MLZero arXiv 2505.13941](https://arxiv.org/pdf/2505.13941)
- Earlier evidence that public Kaggle leaderboards show little adaptive overfitting (Roelofs et al. 2019, >100 competitions) — via search summary only, UNVERIFIED

### Inferences
- The AIRA gap (tens of percent from validation vs test selection) directly justifies a sealed test set the agent never queries; the 2606.11045 counterpoint suggests overfitting risk is bounded when feedback is coarse, but this was not tested on segmentation.
- Run-to-run variance is real: MLE-bench reports ± on 3-36 seeds; Claude 3.5 Sonnet's ± 1.8 on 3 seeds vs GPT-4o ± 0.5 on 36 shows how few seeds hide noise.

### Gaps
- No documented case study found of an agent tampering with the evaluator in MLE-bench-style runs; METR RE-Bench cheating details not confirmed (abstract silent). Toledo et al. gap analysis was found only as the AIRA paper itself.

## Agents vs simple baselines at equal compute

### Takeaway
Two 2026 studies find classical HPO matches or beats LLM agents inside a fixed search space; agents are only competitive when they can edit code. I found no study comparing a full MLE-bench-style agent against a plain sweep or AutoML at equal compute.

### Cited Findings
- Ferreira et al. (Mar 2026), autoresearch/nanochat: 9 methods, 24 GPU-hours each, 3 seeds, 5-min trials on one H200, 14 hyperparameters. Mean best val_bpb (27B Qwen3.5, lower is better): Centaur (hybrid) 0.9763, TPE 0.9768, SMAC 0.9778, CMA-ES 0.9785, agent editing code 0.9814, random search 0.9873, agent over 14 HPs 0.9904; start ~0.991. Authors conclude classical beats LLM in fixed space, avoiding OOM mattered more than diversity; text/tables inconsistent and some gaps within noise. LLM inference time excluded from budget — [arXiv 2603.24647](https://arxiv.org/html/2603.24647v2), [repo](https://github.com/ferreirafabio/autoresearch-automl)
- Budget-matched tabular study (2026): LLM advisor with 12 evals vs random/TPE/GP-BO at 40 on 8 PMLB datasets x 5 seeds. The advisor's own proposals add +0.40 pp CV but -0.01 pp on held-out test (p=0.92); its early lead came from a default seed config; with same seed, random search ties at 5 evals and leads by 0.37 pp at 12; authors advise seeding classical search with a default — [arXiv 2606.21641](https://arxiv.org/html/2606.21641)
- A search summary reports a code-modifying-agent paper where the random-search baseline gave valid output in only 10 of 18 runs, i.e. baselines can be implemented weakly (UNVERIFIED) — [arXiv 2511.00802](https://arxiv.org/pdf/2511.00802)
- Contrary snippet: LLM-guided search AP 0.985 vs 0.965 for random at N=50 (arXiv 2603.15916, UNVERIFIED) — [arXiv](https://arxiv.org/pdf/2603.15916)
- Classical anchor: random search matches or beats manual/grid search (Bergstra & Bengio 2012) — [JMLR](https://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf)

### Inferences
- A fixed sweep as honest control is well supported for hyperparameter-type gains; the agent's plausible edge is structural code changes, so the control should be matched in GPU-hours and the agent's LLM time/cost reported separately.

### Gaps
- No equal-compute comparison of tree-search ML agents vs AutoML (AutoGluon etc.) found. Only one small-model LM task and one tabular task cover the agent-vs-HPO question; none on segmentation.

## Computer vision / segmentation agents and label-error agents

### Takeaway
There are early segmentation-agent systems (Camyla, mAIstro, M3Builder, skill-based baselines) but their evaluations are self-run and lightly audited. Label-error agents exist (Adjudicator) but only on text data and a small subset.

### Cited Findings
- Camyla (Apr 2026): autonomous research for medical image segmentation; CamylaBench 31 datasets from 2025 publications; beats strongest per-dataset baseline (14 architectures incl. nnU-Net) on 22/31 datasets in one run, 18/31 in another, 24/31 union, same training budgets; 28 days on 8-GPU cluster over two runs; >2,700 implementations, 40 manuscripts. Abstract gives no failure rates or limitations — [arXiv 2604.10696](https://arxiv.org/abs/2604.10696)
- Skill-based coding agents for medical imaging (Aug 2026): one workflow, four public challenges; 6th of 15 on both PUMA tracks, 31st of 125 on MILK10k; reports "consistently improves validation performance" — no test numbers on abstract page — [arXiv 2608.23336](https://arxiv.org/abs/2608.23336)
- M3Builder: 94.29% model-building success across 14 radiology datasets (snippet; success = building a model, not quality) — [arXiv 2502.20301](https://arxiv.org/pdf/2502.20301); mAIstro, nnU-Net agent — [arXiv 2505.03785](https://arxiv.org/pdf/2505.03785)
- Auto-nnU-Net (non-LLM AutoML): better segmentation on 6 of 10 MSD datasets — [PMLR](https://proceedings.mlr.press/v293/becktepe25a.html)
- Adjudicator (Dec 2025): council of LLM agents over a knowledge graph for noisy-label correction; 0.99 F1 vs 0.48 single LLM and 0.59 non-KG council on a 1,000-item AlleNoise subset; self-reported, text/product data — [arXiv 2512.13704](https://arxiv.org/html/2512.13704)
- Nahum et al.: LLM-judge ensemble finds substantial label errors in TRUE/SummEval, correcting them shifts reported model performance up — [arXiv 2410.18889](https://papers.cool/arxiv/2410.18889)

### Inferences
- Camyla-style wins over strong baselines at "identical training budgets" are the closest analogue to the planned factory, but nnU-Net-vs-agent comparisons need a sealed test protocol to be trusted; I could not confirm Camyla's protocol.

### Gaps
- No agent benchmark for label-error detection in image/segmentation data found. No X-ray or industrial-inspection agent paper found. Camyla, 2608.23336, M3Builder full text not read (abstract/snippet only).
