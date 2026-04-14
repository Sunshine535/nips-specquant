# Proof Skeleton: AcceptSpec Mathematical Claims

**Generated**: 2026-04-14
**Scope**: All mathematical claims in code + proposal documents
**Difficulty**: nightmare | **Effort**: beast

---

## 1. Dependency DAG

```
P1 (TV Bound)
  ├── uses: L1 (Hadamard Involution)
  ├── uses: L2 (Scalar Quantization Error Bound)
  └── uses: L3 (Softmax Lipschitz Continuity)

L1 (Hadamard Involution: H·H^T = I)
  └── uses: D1 (Normalized WHT butterfly)

L2 (Scalar Quantization Error Bound)
  └── uses: D2 (Per-block min/max uniform quantization)

L4 (Compressed-Domain Attention Equivalence)
  ├── uses: L1 (Hadamard Involution)
  └── uses: L5 (Attention Orthogonal Invariance)

L5 (Attention Orthogonal Invariance)
  └── uses: D3 (Softmax(Q·K^T) structure under orthogonal transforms)

L6 (Rejection Sampling Correctness)
  └── uses: D4 (Standard SD rejection sampling from Leviathan et al.)

L7 (AcceptPredictor Score)
  └── uses: D5 (Logistic regression on attention×value features)

L8 (Gini Coefficient Formula)
  └── uses: D6 (Standard Gini definition)

L9 (Cost Model Inequality)
  └── standalone algebraic claim

EP1 (Empirical Proposition: Acceptance Sensitivity Sparsity)
  └── measurement definition, not a theorem
```

**Cycle check**: NO CYCLES DETECTED.

---

## 2. Assumption Ledger

### P1: TV Bound (`turboquant_kv.py:313-335`)

| Hypothesis | Where Verified | Status |
|------------|---------------|--------|
| A1: W_o (output projection) has bounded Frobenius norm | NEVER — passed as argument | ASSUMED |
| A2: K, V have bounded entry range (range_k, range_v) | NEVER — passed as argument | ASSUMED |
| A3: V has bounded Frobenius norm (v_fnorm) | NEVER — passed as argument | ASSUMED |
| A4: Quantization error is i.i.d. uniform in [-Δ/2, Δ/2] | NOT VERIFIED — real quantization is min/max, not centered | **UNVERIFIED** |
| A5: Quantization error σ = 1/(2√3) · √d / (2^b · √B) | Implicitly from uniform distribution assumption | NEEDS DERIVATION |
| A6: Temperature > 0 | Clamped to 1e-8 in code | VERIFIED |
| A7: Softmax is Lipschitz with constant 1/temperature | ASSUMED — standard but conditions matter | NEEDS VERIFICATION |

### L1: Hadamard Involution (`turboquant_kv.py:26-80`)

| Hypothesis | Where Verified | Status |
|------------|---------------|--------|
| A8: Input dimension is power of 2 (or padded) | `_next_power_of_2` + padding in `rotate()` | VERIFIED |
| A9: Random signs are ±1 (not 0) | `randint(0,2)*2-1` generates {-1,+1} | VERIFIED |
| A10: Normalization factor 1/√d is correct for involution | Requires H·H = d·I for unnormalized → (H/√d)² = I | NEEDS VERIFICATION |

### L4: Compressed-Domain Attention (`turboquant_kv.py:243-280`)

| Hypothesis | Where Verified | Status |
|------------|---------------|--------|
| A11: Q, K, V are all rotated by the SAME Hadamard matrix | Code: Q rotated, K/V rotated separately during quantize | **CHECK NEEDED** |
| A12: Attention scale uses padded_dim, not original dim | Code uses `1/√padded_dim` | VERIFIED but SUSPICIOUS |
| A13: GQA repeat_interleave preserves correctness | Standard HF pattern | VERIFIED |

### L6: Rejection Sampling (`speculative_decode.py:557-607`)

| Hypothesis | Where Verified | Status |
|------------|---------------|--------|
| A14: Draft and target have same vocabulary size | Padded in code (lines 579-582) | VERIFIED |
| A15: Draft probabilities are > 0 for sampled tokens | Clamped to 1e-10 at line 586 | VERIFIED |
| A16: Adjusted distribution (p_t - p_d).clamp(min=0) is valid probability | Normalized at line 594 | NEEDS VERIFICATION |
| A17: Temperature is > 0 | Clamped to 1e-8 | VERIFIED |
| A18: Bonus token sampling at all-accept is correct | Uses verify_logits[:, gamma-1, :] | NEEDS VERIFICATION |

### L7: AcceptPredictor (`acceptspec.py:395-477`)

| Hypothesis | Where Verified | Status |
|------------|---------------|--------|
| A19: Draft attention weights are valid probabilities (sum to 1) | ASSUMED — depends on model output | UNVERIFIED |
| A20: LBFGS convergence to global optimum for logistic regression | Logistic regression is convex → LBFGS converges | VERIFIED (theory) |
| A21: Softmax(w) head weights is appropriate parameterization | Design choice, not correctness requirement | N/A |

---

## 3. Typed Symbol Table

```
Symbol          Type                          Depends on           Notes
─────────────────────────────────────────────────────────────────────────────
α               scalar ∈ [0,1]                KV, U, γ             acceptance rate = n_accepted/γ
S_accept(i)     scalar ∈ [0,1]                α_full, α_perturbed  per-token acceptance sensitivity
γ               int ≥ 1                       -                    draft length (speculation depth)
U_j             scalar ∈ [0,1]                -                    coupled uniform random variable for position j
KV_full         tensor cache                  model, prefix        full-precision KV cache
KV_{i→2bit}     tensor cache                  KV_full, i           KV with token i quantized to 2-bit
p_target(x|ctx) prob. distribution            target model, KV     target model next-token distribution
p_draft(x|ctx)  prob. distribution            draft model          draft model next-token distribution
H               matrix ∈ ℝ^{d×d}             d, signs             normalized signed Hadamard: H = diag(s)·WHT/√d
W_o             matrix ∈ ℝ^{d_model×d_head}  model                output projection matrix
τ               scalar > 0                    -                    temperature parameter
σ_quant         scalar > 0                    d, b, B              quantization noise std dev
d               int                           model config         head dimension
b               int ∈ {2,3,4,8}              config               quantization bit-width
B               int                           config               block size for per-block quantization
score(i)        scalar ≥ 0                    w_h, a_h, v_i        AcceptPredictor criticality score
w_h             scalar ∈ [0,1], Σ=1           calibration          per-head weight (softmax parameterized)
a_h(q,k_i)      scalar ∈ [0,1]               draft attention      attention weight from head h to token i
||v_i||_2       scalar ≥ 0                    value vectors        L2 norm of value vector for token i
G               scalar ∈ [0,1]                sensitivities        Gini coefficient of sensitivity distribution
```

---

## 4. Canonical Quantified Statements

### P1: TV Bound
```
∀ target model M with output projection W_o,
∀ KV cache (K,V) with entry ranges range_k, range_v,
∀ quantization with bits b, block_size B,
∀ temperature τ > 0:

  TV(p_M(·|KV), p_M(·|Q̃(KV))) ≤ ||W_o||_F / τ · (range_v · σ + ||V||_F · range_k · σ / τ)

  where σ = (1/(2√3)) · √d / (2^b · √B)
  and Q̃ denotes Hadamard-rotated scalar quantization.
```
**ISSUE**: σ formula has UNCLEAR derivation. The 1/(2√3) comes from uniform distribution std dev, but √d/√B dependence needs explicit justification.

### L1: Hadamard Involution
```
∀ x ∈ ℝ^d where d is power of 2,
∀ sign vector s ∈ {-1,+1}^d:

  H_s^{-1}(H_s(x)) = x

  where H_s(x) = diag(s) · WHT(x) / √d  [forward]
        H_s^{-1}(y) = WHT(y) · diag(s) / √d  [inverse]
```
**ISSUE**: Code applies signs AFTER WHT in inverse (line 58-59), but forward applies signs BEFORE WHT (line 50-51). Must verify H_s^{-1} ∘ H_s = I under this convention.

### L4: Compressed-Domain Attention
```
∀ query Q ∈ ℝ^{1×d}, keys K ∈ ℝ^{n×d}, values V ∈ ℝ^{n×d}:

  Attn(Q, K̃, Ṽ) ≈ H^{-1}(Attn(H(Q), H(K̃_rot), H(Ṽ_rot)))

  where K̃_rot, Ṽ_rot are quantized in rotated space.
```
**ISSUE**: The equivalence claim requires that softmax(Q·K^T) = softmax((HQ)·(HK)^T) which holds iff H^T·H = I. But the Hadamard is applied with DIFFERENT random signs for Q vs K in the code — need to check whether the SAME `self.rotation` object is used.

### L6: Rejection Sampling
```
∀ draft distribution p_d, target distribution p_t over vocabulary V,
∀ draft token x ~ p_d:

  The output distribution of the rejection sampling procedure equals p_t.

  Specifically:
    - Accept x with prob min(1, p_t(x)/p_d(x)) → contributes p_t(x)∧p_d(x) / p_d(x) · p_d(x) = min(p_t(x), p_d(x))
    - On reject, sample from (p_t - p_d)_+ / ||(p_t - p_d)_+||_1
    - Combined: exactly p_t
```

### L8: Gini Coefficient
```
∀ values v_1,...,v_n ≥ 0 with Σv_i > 0:

  G = (2·Σ_{i=1}^{n} i·v_{(i)}) / (n·Σv_i) - (n+1)/n

  where v_{(1)} ≤ ... ≤ v_{(n)} is the sorted order.
```

---

## 5. Micro-Claim Inventory

### MC-1: WHT butterfly is self-inverse (up to scaling)
**Context**: `_raw_wht` applied twice yields d·x
**Goal**: WHT(WHT(x)) = d·x for x ∈ ℝ^d (d = power of 2)
**Rule**: Walsh-Hadamard matrix H satisfies H² = dI (well-known)
**Side-conditions**: d is power of 2 ✓

### MC-2: Normalized Hadamard with signs is involutory
**Context**: H_s = diag(s)·WHT/√d, H_s^{-1} = WHT·diag(s)/√d
**Goal**: H_s^{-1}(H_s(x)) = x
**Derivation needed**: WHT(diag(s)·WHT(diag(s)·x)/√d)·diag(s)/√d = ?
- = (1/d)·diag(s)·WHT(diag(s)·WHT(diag(s)·x))
- Need: WHT(diag(s)·WHT(diag(s)·x)) = d·x
- This requires: (diag(s)·WHT)² = d·I, i.e., diag(s)·WHT·diag(s)·WHT = d·I
- This is NOT the same as WHT² = d·I unless diag(s) commutes with WHT.
- **CRITICAL**: diag(s) does NOT commute with WHT in general!

### MC-3: TV bound derivation from quantization noise
**Context**: P1 claims TV ≤ f(W_o, ranges, σ, τ)
**Goal**: Derive the bound from first principles
**Rule**: Pinsker's inequality + softmax Lipschitz + quantization error model
**Side-conditions**: quantization error distribution, independence across coordinates
**STATUS**: FULL DERIVATION MISSING — only the final formula is given in code

### MC-4: Scalar quantizer error is bounded by Δ/2
**Context**: Per-block min-max uniform quantization with (2^b - 1) levels
**Goal**: |x - Q(x)| ≤ (max-min) / (2·(2^b - 1))
**Rule**: Rounding to nearest grid point
**Side-conditions**: x ∈ [min, max] ✓ (clamped in code)

### MC-5: Rejection sampling produces target distribution
**Context**: `_rejection_sample` in speculative_decode.py:557-607
**Goal**: Output distribution = p_target
**Rule**: Standard SD proof (Leviathan et al., 2023, Theorem 1)
**Side-conditions**: 
  - p_draft(x) > 0 for all x with p_target(x) > 0: ✓ (clamped 1e-10)
  - Adjusted distribution is valid: (p_t - p_d)_+ normalized to sum 1: ✓ (line 594-595)
  - Bonus token at all-accept uses correct logits: NEEDS CHECK

### MC-6: Compressed-domain attention equivalence
**Context**: `compressed_attention` in turboquant_kv.py:243-280
**Goal**: output = Attn(Q, dequant(K), dequant(V))
**Rule**: Orthogonal invariance of dot-product attention
**Side-conditions**:
  - SAME rotation applied to Q, K, V: Uses `self.rotation` → ✓ same object
  - Output inverse-rotated: ✓ (line 279)
  - Scale factor uses padded_dim: line 263 — **SUSPICIOUS**: should be head_dim?

### MC-7: AcceptPredictor logistic regression is well-specified
**Context**: `AcceptPredictor.fit()` in acceptspec.py:419-457
**Goal**: LBFGS finds optimal head weights w for binary classification
**Rule**: Logistic regression is convex → unique global minimum
**Side-conditions**: 
  - Features X = attn^T * vnorm: shape [kv, heads] — CORRECT
  - Logits = (X * softmax(w)).sum(dim=1): This is NOT standard logistic regression
  - **ISSUE**: softmax(w) constrains weights to simplex — this makes the objective NON-CONVEX in w

### MC-8: Gini coefficient formula matches standard definition
**Context**: `_gini_coefficient` in acceptspec.py:360-370
**Goal**: Implementation matches G = (2Σi·v_{(i)})/(n·Σv) - (n+1)/n
**Rule**: Standard formula
**Side-conditions**: values ≥ 0 — NOT CHECKED (sensitivity values are abs, so ≥ 0 ✓)

### MC-9: Cost model inequality
**Context**: FINAL_PROPOSAL.md:68-74
**Goal**: AcceptSpec wins when f_critical < (T_verify_full - T_compress)/(T_verify_full · (1 - C_compressed/C_full))
**Derivation**: T_acceptspec < T_naive_composition when compression savings exceed overhead
**Side-conditions**: All T > 0, C_compressed < C_full

---

## 6. Limit-Order Map

| Statement | Limit | Uniformity | Location |
|-----------|-------|------------|----------|
| TV(p, p̃) ≤ O(1/2^b) | as b→∞ | For fixed d, B, τ, ranges | turboquant_kv.py:313-335 |
| σ_quant = O(√d / (2^b·√B)) | exact, not asymptotic | For all d, b, B | turboquant_kv.py:329 |
| Top-20% → >80% sensitivity | empirical claim | Over GSM8K test set | FINAL_PROPOSAL.md:34 |
| Spearman ρ < 0.7 | empirical claim | Over GSM8K test set | EXPERIMENT_PLAN.md:14 |

---

## 7. Critical Issues Identified (Pre-Review)

### ISSUE-A: MC-2 — Hadamard involution with random signs may be BROKEN
The forward transform applies signs then WHT: H_s(x) = WHT(diag(s)·x)/√d
The inverse applies WHT then signs: H_s^{-1}(y) = diag(s)·WHT(y)/√d

For involution: H_s^{-1}(H_s(x)) = diag(s)·WHT(WHT(diag(s)·x)/√d)/√d
= diag(s)·WHT(WHT(diag(s)·x))/d = diag(s)·d·diag(s)·x/d = diag(s²)·x = x ✓

Wait — this DOES work because s ∈ {±1} so s² = 1. Let me re-verify carefully:
- H_s(x) = WHT(s⊙x)/√d
- H_s^{-1}(y) = s⊙WHT(y)/√d
- H_s^{-1}(H_s(x)) = s⊙WHT(WHT(s⊙x)/√d)/√d = s⊙(1/d)·WHT(WHT(s⊙x))
- Since WHT(WHT(z)) = d·z: = s⊙(1/d)·d·(s⊙x) = s⊙s⊙x = x ✓ (since s²=1)

**RESOLVED**: The involution IS correct. Key: WHT is its own inverse up to scaling d.

### ISSUE-B: MC-6 — Attention scale uses padded_dim instead of head_dim
At turboquant_kv.py:263: `scale = 1.0 / math.sqrt(padded_dim)`
But standard attention uses `1/√head_dim`. If head_dim ≠ padded_dim (not power of 2), the scale is WRONG.

### ISSUE-C: MC-3 — TV bound σ formula derivation is MISSING
The code asserts σ = 1/(2√3) · √d / (2^b · √B) but provides no derivation.

### ISSUE-D: MC-7 — AcceptPredictor uses softmax(w) making optimization non-convex
Standard logistic regression has convex loss. But logits = (X·softmax(w)).sum(dim=1) introduces softmax parameterization, which makes the landscape non-convex in w.

### ISSUE-E: P1 TV Bound — quantization error model assumes uniform distribution
Real min-max quantization has bounded error but NOT uniform distribution. The variance formula σ² = Δ²/12 only holds for uniform quantization noise, which is an approximation.
