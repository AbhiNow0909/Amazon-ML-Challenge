# Amazon ML Challenge 2026 — Business Entity Resolution

The encoder-first work is on branch `gpt-approach`.

- [Claude continuation handoff](CLAUDE_HANDOFF.md): live-state checks, architecture,
  checkpoint locations, safe resumption and remaining submission work.
- [Pipeline README](code/business_entity_resolution/README.md): environment and run commands.
- [Versioned log snapshot](handoff/encoder/logs/): logs captured at handoff time.
- [Operations guide](logs/README.md): live logs and detached job helpers.

Frozen clean Source1 entity-level holdout macro F0.5: **0.9665124399102869**.
Historical baseline: **0.965614**. Target: **0.980000**, not reached.
Full-test inference is in progress; consult live disk state before acting.
