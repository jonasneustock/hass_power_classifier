# TODOs / Refactors

- [x] Split main.py into routers and an app factory; reduce globals.
- [x] Refactor poller (separate thresholding, hints, publishing).
- [x] Tidy training manager pieces.
- [x] Extract model tuning logic and normalize train/predict.
- [x] DataStore cleanup/transactions.
- [x] Templates/JS partials & chart isolation.
- [x] Config validation improvements.
- [x] API/auth scaffolding.
- [x] Tests for new modules and flows.

Remaining ideas:
- Add auth/token guard for API endpoints.
- Add CSV export/import for segments.
- Increase coverage for poller/training/context.

## Planned migration: SQLite → MongoDB
- **Data model & schema**: Map existing tables to collections:
  - `appliances` → `appliances` (fields: name, power/status entities, activity_sensors, learning flags, power stats, timestamps; unique index on name).
  - `segments` → `segments` (stats, labels/predictions, flank, timestamps; indexes on start_ts, label_appliance, candidate, label_phase).
  - `samples`/`sensor_samples` → `samples` (ts, value, sensor; consider time-series or capped collection; indexes on ts and sensor+ts).
  - `model_metrics` → `model_metrics` (ts, classifier/regression metrics; index on ts).
- **Abstraction layer**: Introduce a datastore protocol; keep method signatures (add_sample, list_segments, etc.). Implement `MongoStore` alongside existing SQLite store; selectable via config (`DB_BACKEND=sqlite|mongo`).
- **Config**: Add envs `DB_BACKEND`, `MONGO_URI`, `MONGO_DB`, optional auth/SSL. Keep SQLite as default for now.
- **Mongo implementation**:
  - Use pymongo with bulk inserts for samples.
  - Enforce appliance uniqueness; updates for labels/predictions/power stats/current_power.
  - Replace SQL deletes/updates with update/aggregate pipelines; add indexes on init.
- **Migration script**:
  - One-off tool to read from SQLite (`DataStore`) and write to Mongo (`MongoStore`), chunked for samples to avoid memory issues.
  - Idempotent option; optional wipe target flag; dry-run mode to count docs.
  - Include a CLI entry (e.g., `python -m app.migrate_sqlite_to_mongo`) and docs on running it before switching `DB_BACKEND`.
- **Tests**:
  - Use `mongomock` or test Mongo container to validate method parity with SQLite.
  - Benchmark critical paths: segment listing/filtering, sample window queries, label updates.
  - (Optional) dual-write mode in tests to compare outputs.
- **Deployment**:
  - Add Mongo service to docker-compose for dev with volume for persistence.
  - Document migration steps in README; switch `DB_BACKEND` to mongo in staging first.
  - Monitor index usage (`explain`) on hot queries after migration.
