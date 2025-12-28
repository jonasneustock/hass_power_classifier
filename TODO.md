# TODOs / Performance Optimizations

- Reduce poller log volume: remove per-diff/trigger logs or lower to debug to avoid I/O overhead on hot loops.
- Slim dashboard payloads: keep recent diff caches small; move chart data to lightweight APIs and lazy-load via JS; narrow event limits further. *(recent diff caches capped to 100, events limit 30; per-sensor diffs computed without DB fetches)*
- Tighten cleanup/retention: lower `SAMPLE_RETENTION_SECONDS` and `UNLABELED_TTL_SECONDS` defaults; add manual “truncate/cleanup” action; ensure cleanup runs often.
- Optimize DB access:
  - Verify SQLite indexes via `EXPLAIN`; add selective column queries for dashboard/segments.
  - Further reduce segments query columns/limits; consider async/lazy segments API for UI.
  - Long term: migrate to MongoDB per plan.
- HA/MQTT polling: consider parallel/async HA reads, longer poll interval on slow HA, and short backoff on repeated failures.
- Template rendering: rely on pagination for segments (done) and consider JS-driven paginated lists to avoid large server-side renders.
- Training: limit training set size (downsample) for large labeled sets; cache models and skip retraining when unchanged.
- Cleanup for published power checks: ensure TTL check isn’t too frequent; avoid heavy computations in poller loop.
