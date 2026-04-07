# Round 1 Review (GPT-5.4 nightmare)

## Scores
| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 8 |
| Method Specificity | 6 |
| Contribution Quality | 7 |
| Frontier Leverage | 8 |
| Feasibility | 5 |
| Validation Focus | 6 |
| Venue Readiness | 6 |
| **Average** | **6.6** |

## Key Criticisms
1. Method underspecified: no exact predictor, no per-layer/head/token implementation detail
2. Proposition 1 is hand-wavy — attention bound ignores residual paths, MLP mixing, multi-layer effects
3. Scope too broad for 150 GPU-hours: too many models × tasks × baselines
4. "acceptance-critical" sparsity may not hold — if diffuse, whole premise collapses
5. Claims not mapped 1:1 to experiments

## Required Fixes
1. Specify exact predictor algorithm and implementation
2. Downgrade to empirical proposition or prove rigorously
3. Cut to 1 model family, 3 tasks
4. Oracle study FIRST to validate sparsity assumption
5. Claim-to-experiment table
