# Copilot Instructions

## Project Context

This workspace hosts a water vibration physics simulation with SPH (Taichi GPU + NumPy CPU validation), Panda3D rendering, and Dear PyGui controls. Prioritize modularity, performance, and scientific traceability.

## Coding Standards

- Keep modules small and single-purpose.
- Prefer pure functions for physics kernels and utilities.
- Use type hints for public APIs.
- Avoid reformatting unrelated code.

## Performance Guidance

- Treat GPU kernels as the performance-critical path.
- Avoid unnecessary host-device transfers.
- Minimize per-frame allocations.

## Data & Logging

- All simulation events and metrics must go through the logging layer.
- Prefer HDF5 for large arrays and SQLite for queryable events.
- Keep logging toggleable by category.

## Testing & Validation

- Preserve CPU validation mode for correctness checks.
- Add comparisons for energy conservation and stability.

## Documentation

- Update inline docstrings for any new public module.
- Keep configuration values centralized.

## Safety

- Clamp unrealistic values when required and log any clamping.
