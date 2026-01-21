# Copilot Instructions

## Project Context

This workspace hosts a **Vibrating Water** physics simulation. It uses a **hybrid architecture**:

- **Physics Core**: Adaptive SPH simulation switching between **Taichi (GPU)** for performance (10k particles) and **NumPy (CPU)** for validation (~1k particles).
- **Rendering**: **Panda3D** engine with custom GLSL shaders for water surface, refraction, and particle effects.
- **UI/Control**: **Dear PyGui** for real-time parameter manipulation and graphs.
- **Data**: Hybrid logging using **HDF5** (time-series data) and **SQLite** (event logs), with **zstandard** compression.

## Architecture & Code Organization

The codebase follows a modular structure in `src/`:

- `src/physics/`: Core solvers (`TaichiSPHSolver`, `NumpySPHSolver`).
- `src/rendering/`: Panda3D integration, `shaders/` directory, and visual effects managers.
- `src/gui/`: UI components and plots built with Dear PyGui.
- `src/data/`: Data logging, `AdaptiveLODController`, and storage handlers.
- `src/validation/`: Tools for comparing GPU vs CPU results.

## Critical Developer Workflows

- **Taichi Initialization**: Always configure memory explicitly for Windows compatibility.
  ```python
  ti.init(arch=ti.gpu, device_memory_GB=4) # Required for >1GB usage on Windows
  ```
- **Virtual Environment**: Use the provided `Dockerfile` for referencing system deps, but develop locally in a `.venv` (untracked).
- **Validation**: When modifying physics kernels, run the CPU validator to ensure energy conservation hasn't regressed.
- **Visual Effects**: Shader updates go in `src/rendering/shaders/`. Test performance impact on the A10G target (or local equivalent) before merging.

## Coding Standards

- **Performance First**: In `src/physics`, minimize host-device transfers. Use `@ti.kernel` for compute-heavy loops.
- **Type Safety**: Use type hints for all public interfaces.
- **Modularity**: Keep rendering logic decoupled from physics data structures—use intermediate buffers/arrays for state transfer.
- **Documentation**: Inline docstrings are required for all math-heavy functions (explain the physics/maths, not just code).

## Data & Logging Patterns

- **Events vs Data**: Log discrete events (LOD switches, errors) to SQLite. Log continuous data (position, velocity) to HDF5.
- **LOD Switching**: Implement hysteresis to avoid rapid toggling between CPU/GPU modes.
- **Safety**: Clamp inputs to physical reality (e.g., speed of light limit) and log warnings if exceeded.

## Testing

- Integration tests should verify the "Pause/Reset" functionality to ensure no state leaks.
- Unit tests for physics should compare `NumpySPHSolver` single-step output against analytical solutions where possible.
- System tests should exercise end-to-end simulation paths, performance benchmarks, and stress scenarios.
- If Panda3D is unavailable, use the stub classes and feature flags in `visual_effects_manager.py` to keep tests runnable.
