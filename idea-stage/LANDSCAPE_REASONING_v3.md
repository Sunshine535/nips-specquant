# Reasoning-Model Inference Landscape (v3)

**Scope.** Efficient inference for long-CoT reasoning models (o1 / R1 / QwQ / Qwen3.5 / Gemini-Thinking / Claude thinking). Bias toward Oct 2025 -> Apr 2026. Focus on finding NeurIPS-2026-worthy white space at the intersection of reasoning structure and inference pipeline.
**Date.** 2026-04-19.

---

## 1. Recent high-signal papers

### 1.1 Speculative decoding for reasoning

- **SCoT (arXiv:2504.19095, 2025).** Small-large collab at CoT segment level; 1.6-2.3x on R1/QwQ.
- **Lookahead Reasoning (arXiv:2506.19830, 2025).** Two-layer parallelism (step + token). Step acceptance >=50% because steps need *semantic* match, not exact tokens; 1.4->2.1x.
- **SpecGuard (arXiv:2604.15244, 2026).** Step-level verify via attention grounding + logprob; +3.6% acc, -11% latency vs RSD.
- **RSD — Reward-Guided SD (arXiv:2501.19324, 2025).** PRM gates target invocation; up to 4.4x FLOPs.
- **Acceptance Dynamics Across Cognitive Domains (arXiv:2604.14682, 2026).** Task type predicts acceptance more than tree depth; implies domain-aware tree design.
- **SparseSpec / PillarAttn (arXiv:2512.01278, 2025).** Self-spec with sparse attention draft; 2.13x; memory-bound view.
- **EASD (arXiv:2512.23765, 2025).** Entropy-gated SD to stop reasoning error propagation.
- **Variational SD (arXiv:2602.05774, 2026).** Draft trained for sequence-level acceptance, not token likelihood.
- **TALON (arXiv:2601.07353, 2026).** Draft tree shaped by confidence.
- **Adaptive Drafter (arXiv:2511.16665, 2025).** SD for long-tail rollouts in reasoning RL.

### 1.2 Reasoning-specific KV compression

- **R-KV (arXiv:2505.24133, 2025).** Redundancy-aware; 10% cache ~= 100% perf.
- **ThinKV (arXiv:2510.01290, 2025).** Thought-adaptive per thought type; <5% cache, 5.8x.
- **RLKV (arXiv:2510.08525, 2025).** RL-discovered reasoning-critical heads; 20-50% reduction.
- **Lethe (arXiv:2511.06029, 2025).** Layer/time-adaptive pruning.
- **SideQuest (arXiv:2602.22603, 2026).** KV for long-horizon agentic reasoning.
- **Hold Onto That Thought (arXiv:2512.12008, 2025).** Reasoning models retain *more* critical tokens than instruct; aggressive eviction paradoxically *lengthens* CoT.
- **MemShare (arXiv:2507.21433, 2025).** Similar reasoning steps -> reusable KV blocks; +84% throughput.
- **Think Clearly (arXiv:2507.08806, 2025).** Step-aware hierarchical eviction.
- **DeltaKV (arXiv:2602.08005, 2026).** Residual to retrieved history; 29% size.
- **TriAttention (arXiv:2604.04921, 2026).** Trig compression for long reasoning.

### 1.3 Adaptive thinking / early exit

- **DEER (arXiv:2504.15895, 2025).** Early exit at "Wait" tokens on high trial-answer confidence.
- **CoDE-Stop (arXiv:2604.04930, 2026); ROM (arXiv:2603.22016, 2026); RPDI-EE (arXiv:2603.14251, 2026); DTSR (arXiv:2604.06787, 2026); LYNX (arXiv:2512.05325, 2025).** Variants of confidence/deviation/sufficiency-based exit.
- **NoWait (arXiv:2506.08343, 2025).** Suppress Wait/Hmm tokens; 27-51% shorter CoT, no acc loss.
- **EGB (arXiv:2503.21961, 2025); EAGER (arXiv:2510.11170, 2025); ENTRA (arXiv:2601.07123, 2026); CEEH (arXiv:2602.22642, 2026).** Entropy-gated branching / redundancy avoidance.
- **Reasoning on a Budget survey (arXiv:2507.02076, 2025); Efficiency->Adaptivity (arXiv:2511.10788, 2025).**

### 1.4 Latent / implicit reasoning

- **Coconut (arXiv:2412.06769, 2025).** Continuous-thought BFS in latent space.
- **ALiCoT (arXiv:2601.21576, 2026); AdaAnchor (arXiv:2603.15051, 2026).** Aligned/halted latent refinement; 54x / 92% reductions.
- **DLCM (arXiv:2512.24617, 2025).** Reasoning in compressed concept space.
- Surveys: arXiv:2505.16782; formal analysis arXiv:2509.25239 (latent admits more parallelism).

### 1.5 Mechanistic understanding

- **Thought Anchors (arXiv:2506.19143, 2025).** Planning/uncertainty sentences carry disproportionate causal weight; dedicated heads route to them.
- **Mechanistic CoT (arXiv:2402.18312, 2024).** Mid-layer rift; answer heads late, relational heads early.
- **Internal states before "wait" (arXiv:2510.04128, 2025); Why-do-models-loop (arXiv:2512.12895, 2025).**

### 1.6 Test-time compute scaling

- **Scaling up TTC w/ Latent Reasoning** NeurIPS'25 spotlight (recurrent depth).
- **AB-MCTS (arXiv:2503.04412, NeurIPS'25); Art of Scaling TTC (arXiv:2512.02008, 2025).** No universal TTC strategy across 30B tokens.
- **Optimal Self-Consistency (arXiv:2511.12309, 2025).** Adaptive per-question sampling.
- **SSA (arXiv:2506.09014, 2025).** LM aggregator beats majority vote by 8% pass@5.

### 1.7 Cross-cutting systems

- **Log-Linear Attention (ICLR'26, arXiv:2506.04761); SSE-H (arXiv:2507.16577, 2025).** Reasoning with O(log n) state.
- **KVFlow (arXiv:2507.07400, 2025).** Workflow-aware prefix cache.
- **ENGRAM-R (arXiv:2511.12987, 2025).** Retrieval + fact cards; -85% input tokens.

---

## 2. Saturated areas (avoid pure incremental work)

1. **Generic CoT length reduction** by entropy/confidence thresholding -- DEER, CoDE-Stop, NoWait, EAGER, ROM, RPDI-EE, LYNX, DTSR all overlap.
2. **Token-level KV eviction on reasoning** with heuristic importance scoring -- R-KV, ThinKV, Lethe, Think Clearly, H2O/SnapKV-family now heavily populated.
3. **Generic tree-SD with larger drafts / bigger trees.** Eagle-family and descendants crowd this.
4. **"Reasoning is redundant, compress it"** narratives without a structural insight.
5. **Step-level SD for reasoning** -- Lookahead + SCoT + SpecGuard already cover the obvious version.

## 3. White spaces (specific gaps)

**W1. Acceptance-*critical* vs attention-important vs perplexity-important tokens (the user's own finding).** Prior work only shows: (a) attention heads differ in reasoning-criticality (RLKV), (b) some tokens are heavy hitters (Hold Onto That Thought). No prior work has an **acceptance-rate-centric** importance signal that is shown divergent from attention/PPL -- this is a genuinely novel axis.

**W2. Structural phase of CoT x inference policy.** Reasoning traces have distinct phases (plan -> derive -> verify -> backtrack -> conclude). Existing work treats CoT as one homogeneous stream. No paper couples phase type to draft-length, tree shape, KV precision, or early-exit threshold. Thought Anchors identifies phases but does not act on them in the inference pipeline.

**W3. Backtrack / self-correction aware caching.** When a model says "Wait, that's wrong", the prior branch's KV is still kept. No one has studied *discard-on-backtrack* vs *retain-for-contrast* policies and their effect on downstream acceptance/accuracy.

**W4. SD with reasoning-branch anticipation.** Drafts collapse when the target backtracks. A draft that *predicts* the branching/wait event and proposes two futures (continue vs reset) is unexplored.

**W5. Verifier (PRM) x draft co-design.** RSD gates with PRM but does not feed PRM signal into the draft itself. Using PRM gradient to steer draft sampling (similar to classifier-free guidance but for reasoning correctness) is open.

**W6. Cross-sample KV sharing across self-consistency runs.** Self-consistency + best-of-N run k parallel CoTs; only common prefix is shared. Semantic sub-trajectory equivalences (MemShare touches this) are under-exploited -- e.g. merge two trajectories once they have converged on the same intermediate result.

**W7. Latent reasoning x speculation.** Coconut/AdaAnchor compress CoT into latent; but no one uses latent CoT *as the drafter* for explicit CoT generation.

**W8. Acceptance-rate x adaptivity feedback loop.** Online-estimate the target's current step-level acceptance, and use it to re-allocate draft length, tree width, KV precision dynamically. Present methods are static.

**W9. Structural anti-patterns.** Loops, repeated self-doubt (Wait Wait Wait paper) -- no inference-time method that detects structural pathology and *short-circuits* it rather than just length-capping.

**W10. MoE expert activation in reasoning phases.** SERE reroutes for batches, but no one has studied whether planning/verification/execution phases activate distinct expert subsets that can be pre-fetched.

## 4. Concurrent-work threats

- **DeepSeek** (R1 lineage + V4 efficiency focus, 40% memory / 1.8x speedup). Strong on RL+inference-coupled work; likely to publish reasoning-specific KV work.
- **Anthropic** (Claude thinking budget). Internal work on adaptive thinking allocation; public research footprint smaller.
- **Google DeepMind** (Gemini Deep Think). Active on latent/recurrent reasoning; NeurIPS'25 spotlight already.
- **Qwen (Alibaba)** -- heavy on Qwen3 / Qwen3.5 + native MTP head; likely to publish MTP-SD x reasoning work.
- **OpenAI** o-series -- closed, but research papers from their team on SD and verifier-guided decoding continue.
- **Meta FAIR** -- Coconut lineage; latent reasoning push.
- **MIT / Princeton (Tri Dao et al.)** -- Lookahead Reasoning, Speculative Speculative Decoding (arXiv:2603.03251); strong SD pipeline.
- **UCSD / Berkeley** -- R-KV, many KV-compression papers.

## 5. Five candidate idea seeds (reasoning-inference intersection)

**S1. AcceptSpec v3 -- Phase-conditioned acceptance-critical caching.**
Build on the user's finding that acceptance-critical tokens differ from attention-important. Add the structural dimension: each reasoning phase (plan/derive/verify/backtrack) has a *different* acceptance-critical subset. Train a tiny phase classifier on hidden states (free from Thought Anchors signals) and allocate KV precision per (phase, token) pair. Baselines: R-KV, ThinKV, RLKV. Novelty: *joint* phase x acceptance-criticality, not just one.

**S2. Backtrack-aware KV policy with contrastive retention.**
When the model emits "Wait"/"but", do not evict the failed branch; instead *quantize it harshly and retain as negative context*. Study effect on acceptance rate and on final accuracy (hypothesis: contrastive anti-memory reduces repeated mistakes -> increases acceptance downstream). No prior work does this -- NoWait/loop-analysis papers just remove or detect, none use.

**S3. Branch-predicting speculative draft (BPSD).**
Train the drafter to emit a *branching indicator* alongside tokens: continue vs restart. When branch-prob is high, target verifies two candidate futures in parallel (cheap because memory-bound). Extends Lookahead Reasoning, which is step-parallel but not branch-parallel. Targets R1-Distill-Qwen, Qwen3.5-MTP self-speculation. Key metric: accepted-length under high-branching regimes (AIME / HARP).

**S4. PRM-guided draft sampling (PRM-steered MTP head).**
Instead of using PRM only to gate acceptance (RSD), fine-tune/prompt the MTP draft head with a PRM-derived reward shaping term so the draft distribution itself is biased toward reasoning-correct continuations. The draft-target KL shrinks on reasoning-critical steps, raising acceptance. Novel because existing PRM work is verifier-side; this makes it *generator-side*.

**S5. Latent-thought drafter for explicit-CoT target.**
Use a Coconut-style latent-reasoning small model to produce a compressed latent plan, decode it into token-level draft, and have the explicit-CoT target verify. Exploits that the hard work (planning) is parallelizable in latent space and the easy work (surface form) is easily verified. No prior SD method uses a fundamentally *different reasoning modality* for draft vs target.

### Ranking for the user's context

Given the existing AcceptSpec infrastructure and Qwen3.5 MTP codebase, **S1 is a direct extension** that raises the work from "an observation" to "an observation + a structural mechanism". **S3 and S4** are natural next steps that reuse the speculative-decoding stack. **S5** is highest-risk highest-reward. **S2** is cheap to pilot and strongly differentiated.

---

## Sources
- [A Survey on KV Cache Management (arXiv:2412.19442)](https://arxiv.org/abs/2412.19442)
- [R-KV (arXiv:2505.24133)](https://arxiv.org/abs/2505.24133)
- [ThinKV (arXiv:2510.01290)](https://arxiv.org/abs/2510.01290)
- [RLKV (arXiv:2510.08525)](https://arxiv.org/abs/2510.08525)
- [Lethe (arXiv:2511.06029)](https://arxiv.org/abs/2511.06029)
- [SideQuest (arXiv:2602.22603)](https://arxiv.org/abs/2602.22603)
- [Hold Onto That Thought (arXiv:2512.12008)](https://arxiv.org/abs/2512.12008)
- [MemShare (arXiv:2507.21433)](https://arxiv.org/abs/2507.21433)
- [Think Clearly (arXiv:2507.08806)](https://arxiv.org/abs/2507.08806)
- [DeltaKV (arXiv:2602.08005)](https://arxiv.org/abs/2602.08005)
- [SmallKV (arXiv:2508.02751)](https://arxiv.org/abs/2508.02751)
- [SCoT (arXiv:2504.19095)](https://arxiv.org/abs/2504.19095)
- [Lookahead Reasoning (arXiv:2506.19830)](https://arxiv.org/abs/2506.19830)
- [SpecGuard (arXiv:2604.15244)](https://arxiv.org/abs/2604.15244)
- [RSD (arXiv:2501.19324)](https://arxiv.org/abs/2501.19324)
- [Acceptance Dynamics (arXiv:2604.14682)](https://arxiv.org/abs/2604.14682)
- [SparseSpec (arXiv:2512.01278)](https://arxiv.org/abs/2512.01278)
- [EASD (arXiv:2512.23765)](https://arxiv.org/abs/2512.23765)
- [DEER (arXiv:2504.15895)](https://arxiv.org/abs/2504.15895)
- [CoDE-Stop (arXiv:2604.04930)](https://arxiv.org/abs/2604.04930)
- [ROM (arXiv:2603.22016)](https://arxiv.org/abs/2603.22016)
- [NoWait (arXiv:2506.08343)](https://arxiv.org/abs/2506.08343)
- [EGB (arXiv:2503.21961)](https://arxiv.org/abs/2503.21961)
- [EAGER (arXiv:2510.11170)](https://arxiv.org/abs/2510.11170)
- [Coconut (arXiv:2412.06769)](https://arxiv.org/abs/2412.06769)
- [AdaAnchor (arXiv:2603.15051)](https://arxiv.org/abs/2603.15051)
- [Thought Anchors (arXiv:2506.19143)](https://arxiv.org/abs/2506.19143)
- [Mechanistic CoT (arXiv:2402.18312)](https://arxiv.org/abs/2402.18312)
- [Wait Why Loop (arXiv:2512.12895)](https://arxiv.org/abs/2512.12895)
- [Internal states before wait (arXiv:2510.04128)](https://arxiv.org/abs/2510.04128)
- [AB-MCTS (arXiv:2503.04412)](https://arxiv.org/abs/2503.04412)
- [Optimal Self-Consistency (arXiv:2511.12309)](https://arxiv.org/abs/2511.12309)
- [SSA (arXiv:2506.09014)](https://arxiv.org/abs/2506.09014)
- [Log-Linear Attention (arXiv:2506.04761)](https://arxiv.org/abs/2506.04761)
- [SSE-H (arXiv:2507.16577)](https://arxiv.org/abs/2507.16577)
- [KVFlow (arXiv:2507.07400)](https://arxiv.org/abs/2507.07400)
- [Reasoning on a Budget (arXiv:2507.02076)](https://arxiv.org/abs/2507.02076)
- [Between Under/Overthinking (arXiv:2505.00127)](https://arxiv.org/abs/2505.00127)
- [Adaptive Drafter (arXiv:2511.16665)](https://arxiv.org/abs/2511.16665)
- [TALON (arXiv:2601.07353)](https://arxiv.org/abs/2601.07353)
- [Variational SD (arXiv:2602.05774)](https://arxiv.org/abs/2602.05774)
- [DeepSeek-R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)
