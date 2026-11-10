# BioCache adapter for HiPPO

This package is only a **proxy** over the repository's recurrent cell state. It
does not turn HiPPO into a retrieval system and does not claim a complete VLM
integration.

`HiPPOPersistentState` explicitly exposes:

- hidden state `h`
- polynomial `memory`
- `time_step`

`HiPPOBioCacheAdapter.from_cell_state(cell, state)` wraps the existing
`(h, memory, time_step)` triple without importing PyTorch. When a caller passes
an observation, `update` may delegate to the wrapped cell's existing
`forward`; `query` only returns an injected query result, the cached cell
output, or `h`.

The common API is `reset`, `update`, `query`, `persistent_state`,
`static_extra_state`, and `temporary_state`. Recursive byte accounting, timing,
byte/time budgets, manifest schema, and delayed-recall/VNBench-feature JSONL
schemas are included.

No-model validation:

```bash
python -m biocache_adapt --dry-run --pretty
python -m biocache_adapt --help
```
