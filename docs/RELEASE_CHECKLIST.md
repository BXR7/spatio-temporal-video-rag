# Release readiness

This branch is a research/prototype upgrade. It is not a published release and no tag is created.

- [x] Source modules extracted from notebook `%%writefile` cells
- [x] Notebook no longer the only implementation source
- [x] Dependency specification added
- [x] `.env.example` added
- [x] `.gitignore` added
- [x] Architecture documentation added
- [x] Evaluation limitations documented
- [x] Deterministic unit tests added
- [ ] Clean full installation and model-backed ingestion/search
- [ ] Independent ground-truth annotations and retrieval-quality metrics
- [ ] GPU validation on the target deployment configuration
- [ ] Truthful recorded demo/GIF
- [ ] License selected and added by the owner
- [ ] CI and integration test coverage
- [x] No generated databases, model weights, or secrets added
- [ ] Public release approval

## Recommendation

If the owner later approves a release after GPU/model validation, an honest prototype label such as `v0.1.0-research` is more appropriate than `v1.0.0`. Do not publish automatically.

## License review

No license is added on this branch. The owner should review source, library, model, and demo-media terms before choosing MIT, Apache-2.0, or another license.
