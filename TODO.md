# TODOs / Refactors

- [ ] Split main.py into routers (dashboard/segments/appliances/models/logs/api) and an app factory; reduce globals.
  - Create FastAPI app in a factory; register routers from separate modules; inject services.
- [ ] Refactor poller: separate sensor polling, adaptive thresholding, segment creation, activity/learning hints, and power publishing.
  - Make HA/MQTT interactions injectable; add unit tests for thresholds and hints.
- [ ] Tidy training manager: isolate eligibility checks, metric persistence, scheduler into smaller helpers.
  - Make power stats computation pure; decouple metric history I/O.
- [ ] Extract model tuning logic: centralize hyperparameter grids/utilities for classifier/regression; normalize train/predict interfaces.
  - Keep feature extraction consistent and reusable.
- [ ] DataStore cleanup: group SQL by concern, add transaction helpers, consider a small repository layer.
  - Centralize schema evolution/migrations in one place.
- [ ] Templates/JS: move repeated UI pieces into partials; isolate chart rendering; align data shapes between backend and frontend.
  - Consider a shared schema/typing for chart data.
- [ ] Configuration: move env parsing to dataclasses/pydantic with validation by domain (segmentation/training/MQTT/scheduler).
  - Fail fast on invalid config; simplify defaults.
- [ ] API/Auth: add token guard for automation endpoints; split `/api` routes into their own router with consistent responses.
- [ ] Tests: add fakes for HA/MQTT/poller; cover adaptive thresholds, learning hints, and new routing/config validation.
  - Raise coverage on new modules after refactor.
