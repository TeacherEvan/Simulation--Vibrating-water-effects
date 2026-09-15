# TODO — Vibrating Water Effects Simulation

## Objectives

| ID   | Objective | Status | Evidence |
|------|-----------|--------|----------|
| OBJ-001 | Fix `VisualEffectsManager.__init__` crash under test mocks (FilterManager raises "Could not find appropriate DisplayRegion to filter") | [x] | `tests/unit/test_visual_effects.py` 54/54 pass |
| OBJ-002 | Add `_StubTexture`, `_StubFilterManager`, `_StubNodePath` fallback classes for headless test mode | [x] | `_init_filter_manager` try/except -> stub path exercised by 30+ tests |
| OBJ-003 | Full pytest suite green (159 passed, 4 skipped, 0 failed) | [x] | `pytest -q` exit 0; live re-run 2026-09-15 |
| OBJ-004 | Commit fix with scoped diff (source + TODO only) | [x] | `git diff --stat` shows 2 files |
| OBJ-005 | Verify no secrets/credentials exposed in diff | [x] | rg scan clean |
| OBJ-006 | Confirm `.gitignore` already covers `docs/.scratch-audit/` | [x] | `.gitignore:56` already present |

## Definition of Done
- [x] All unit/integration/system tests pass
- [x] No regressions in physics, data logging, or system tests
- [x] Diff scoped to `visual_effects_manager.py` + `TODO.md`
- [x] No new dependencies, no secrets, no config changes
