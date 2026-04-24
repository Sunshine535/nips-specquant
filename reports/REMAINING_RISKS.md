# Remaining Risks

| Risk | Severity | Mitigation | Status |
|------|----------|------------|--------|
| P0 oracle kv_len bug | High | Fix in oracle_sensitivity.py before any new oracle measurements | DOCUMENTED, NOT FIXED |
| P0 core_comparison MTP path | High | Fix target-as-draft before any policy comparison | DOCUMENTED, NOT FIXED |
| M3 oracle_accept = 0% accuracy | High | Wait for fp16_baseline result; if also 0%, model/budget issue | IN PROGRESS |
| Qwen3.5-9B wrong model | High | May need to pivot to Qwen3-8B (pure MHA) | DECISION PENDING |
| MARA predictor on real data | Medium | Must beat uniform on matched-support risk ranking | UNTESTED |
| Risk signal may not be predictable | Medium | If uniform beats MARA, stop MARA method | UNTESTED |
| Systems overhead | Medium | Separate quality claim from speed claim | DOCUMENTED |
| SpecAttn/QuantSpec novelty threat | High | Must cite and directly compare | ACKNOWLEDGED |
| M2 Spearman artifact | Medium | Matched-support recomputation needed | DOCUMENTED |
| Multi-seed stability | Medium | Run 3 seeds before any claim | PLANNED |
