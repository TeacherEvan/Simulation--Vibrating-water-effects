# System Requirements for Python-Based SPH Fluid Simulation

## Technology Stack Overview
- **Taichi Lang** - GPU-accelerated computing (10k particles)
- **Panda3D** - 3D rendering with shaders
- **Dear PyGui** - GUI framework
- **NumPy/SciPy** - CPU computations
- **HDF5 + SQLite** - Data storage
- **Jupyter Notebooks** - Analysis

---

## 1. Minimum Hardware Requirements

| Component | Minimum Specification |
|-----------|----------------------|
| **CPU** | Quad-core processor (Intel Core i5-6500 / AMD Ryzen 3 2200G or equivalent) |
| **RAM** | 8 GB DDR4 |
| **GPU** | NVIDIA GTX 750 Ti (2GB VRAM) or equivalent with OpenGL 4.3+ support |
| **Storage** | 256 GB SSD with 50 GB free space |
| **Display** | 1920×1080 resolution |

### Minimum Configuration Notes
- CPU fallback mode (1k particles) will be functional on integrated graphics
- Real-time 60 FPS may not be achievable; expect 20-30 FPS at minimum specs
- SSD strongly recommended for data logging performance

---

## 2. Recommended Hardware Requirements

| Component | Recommended Specification |
|-----------|--------------------------|
| **CPU** | 8-core processor (Intel Core i7-10700 / AMD Ryzen 7 3700X or better) |
| **RAM** | 32 GB DDR4-3200 or faster |
| **GPU** | NVIDIA RTX 3060 (8GB VRAM) or RTX 4060 / AMD RX 6700 XT |
| **Storage** | 512 GB NVMe SSD with 100+ GB free space |
| **Display** | 2560×1440 resolution or higher |

### Recommended Configuration Notes
- Enables full 10k particle simulation at 60 FPS
- Supports complex water shaders with refraction effects
- Adequate headroom for extended simulation sessions
- NVMe SSD ensures smooth real-time data logging

---

## 3. GPU Requirements

### 3.1 Supported GPU Backends (Taichi Lang)

| Backend | Requirements | Notes |
|---------|-------------|-------|
| **CUDA** | NVIDIA GPU with Compute Capability 5.0+ | Primary recommended backend |
| **Vulkan** | Vulkan 1.0+ compatible GPU | Cross-platform alternative |
| **OpenGL** | OpenGL 4.3+ | Fallback option, wider compatibility |
| **Metal** | macOS with Apple Silicon or AMD GPU | macOS only |
| **CPU** | x64 or ARM64 processor | Universal fallback |

### 3.2 CUDA Version Requirements

| Component | Version |
|-----------|---------|
| **CUDA Toolkit** | 10.0 or higher (11.x+ recommended) |
| **cuDNN** | Not required |
| **NVIDIA Driver** | 450.80+ for CUDA 11.x |

### 3.3 OpenGL/Vulkan Version Requirements

| API | Minimum Version | Recommended |
|-----|-----------------|-------------|
| **OpenGL** | 4.3 (required by Taichi) | 4.6 |
| **Vulkan** | 1.0 | 1.2+ |
| **OpenGL ES** | 3.0 (mobile/embedded) | 3.2 |

### 3.4 VRAM Requirements

| Workload | Minimum VRAM | Recommended VRAM |
|----------|--------------|------------------|
| 1k particles (CPU fallback) | N/A (system RAM) | N/A |
| 10k particles (GPU) | 2 GB | 4 GB |
| 10k particles + complex shaders | 4 GB | 6-8 GB |
| Extended simulations with data caching | 6 GB | 8+ GB |

### 3.5 GPU Memory Configuration (Taichi)

On Windows, Taichi pre-allocates only 1 GB GPU memory by default. For 10k particle simulations, configure:

```python
import taichi as ti

# Option 1: Allocate 90% of GPU memory
ti.init(arch=ti.cuda, device_memory_fraction=0.9)

# Option 2: Specify exact allocation (4 GB)
ti.init(arch=ti.cuda, device_memory_GB=4)
```

Or set environment variables:
```
TI_DEVICE_MEMORY_FRACTION=0.9
TI_DEVICE_MEMORY_GB=4
```

### 3.6 NVIDIA GPU Compatibility Notes

| GPU Generation | Compute Capability | Status |
|----------------|-------------------|--------|
| Maxwell (GTX 900 series) | 5.0-5.2 | ✅ Supported |
| Pascal (GTX 10 series) | 6.0-6.1 | ✅ Fully Supported |
| Turing (RTX 20 series) | 7.5 | ✅ Fully Supported |
| Ampere (RTX 30 series) | 8.6 | ✅ Recommended |
| Ada Lovelace (RTX 40 series) | 8.9 | ✅ Optimal |

**Note:** Pre-Pascal GPUs have limited Unified Memory support. If encountering CUDA errors on older GPUs:
```bash
export TI_USE_UNIFIED_MEMORY=0
```

---

## 4. Operating System Compatibility

| OS | Version | Support Level |
|----|---------|---------------|
| **Windows** | 10 (64-bit) or later | ✅ Full Support |
| **Windows** | 11 (64-bit) | ✅ Full Support |
| **Ubuntu** | 18.04 LTS or later | ✅ Full Support |
| **Debian** | 10 or later | ✅ Supported |
| **Fedora** | 32 or later | ✅ Supported |
| **Arch Linux** | Rolling release | ✅ Supported |
| **macOS** | 10.15 (Catalina) or later | ⚠️ Partial (Metal backend only) |
| **macOS** | Apple Silicon (M1/M2/M3) | ⚠️ Partial (Metal backend only) |

### OS-Specific Notes

**Windows:**
- Requires [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe)
- Hardware-accelerated OpenGL driver required (not GDI Generic)

**Linux:**
- libstdc++6 with CXXABI 1.3.11+ required
- For Ubuntu 16.04, install updated libstdc++6:
  ```bash
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
  sudo apt-get update
  sudo apt-get install libstdc++6
  ```

**macOS:**
- CUDA not available; use Metal backend
- Some Panda3D features may have reduced performance

---

## 5. Python Version Requirements

### 5.1 Python Version Compatibility Matrix

| Library | Python 3.8 | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 |
|---------|:----------:|:----------:|:-----------:|:-----------:|:-----------:|
| **Taichi Lang** | ✅ | ✅ | ✅ | ⚠️ | ❌ |
| **Panda3D** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Dear PyGui** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **NumPy 2.0+** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **SciPy 1.8+** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **h5py** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Jupyter** | ✅ | ✅ | ✅ | ✅ | ✅ |

### 5.2 Recommended Python Version

**Python 3.10 (64-bit)** - Best compatibility across all stack components

### 5.3 Critical Requirements
- **64-bit Python required** (32-bit not supported)
- Taichi Lang currently supports Python 3.7-3.10 (check for updates)
- NumPy 2.0 requires Python 3.9+

### 5.4 Virtual Environment Setup

```bash
# Create virtual environment
python -m venv sph_simulation

# Activate (Windows)
sph_simulation\Scripts\activate

# Activate (Linux/macOS)
source sph_simulation/bin/activate

# Install dependencies
pip install taichi panda3d dearpygui numpy scipy h5py jupyterlab
```

---

## 6. Disk Space Estimates for Simulation Data

### 6.1 Per-Simulation Storage

| Data Type | Size per Frame | Size per Hour (60 FPS) |
|-----------|---------------|------------------------|
| Particle positions (10k, float32) | ~120 KB | ~26 GB |
| Particle velocities (10k, float32) | ~120 KB | ~26 GB |
| Particle densities (10k, float32) | ~40 KB | ~8.6 GB |
| Full state snapshot (10k particles) | ~400 KB | ~86 GB |
| Compressed HDF5 (gzip level 4) | ~100 KB | ~21 GB |

### 6.2 Recommended Storage Allocation

| Use Case | Minimum Space | Recommended Space |
|----------|--------------|-------------------|
| Development/testing | 10 GB | 25 GB |
| Short simulations (< 5 min) | 25 GB | 50 GB |
| Extended research sessions | 100 GB | 250 GB |
| Long-term data archival | 500 GB | 1+ TB |

### 6.3 SQLite Database Estimates

| Database Content | Estimated Size |
|------------------|----------------|
| Simulation metadata (1000 runs) | ~50 MB |
| Experiment configurations | ~10 MB |
| Analysis results cache | ~500 MB |

### 6.4 Storage Performance Requirements

| Storage Type | Real-time Logging | Recommended |
|--------------|-------------------|-------------|
| HDD (7200 RPM) | ⚠️ May drop frames | ❌ |
| SATA SSD | ✅ Adequate | ⚠️ |
| NVMe SSD | ✅ Optimal | ✅ |

**Tip:** For real-time data logging at 60 FPS, sustained write speed of 25+ MB/s required.

---

## 7. Known Compatibility Issues and Limitations

### 7.1 Taichi Lang Limitations

| Issue | Description | Workaround |
|-------|-------------|------------|
| Windows CUDA memory | Default 1 GB pre-allocation | Use `device_memory_GB` parameter |
| Pre-Pascal CUDA | Limited Unified Memory | Set `TI_USE_UNIFIED_MEMORY=0` |
| OpenGL version | Requires 4.3+ | Set `TI_ENABLE_OPENGL=0` if issues |
| macOS CUDA | Not supported | Use Metal backend |
| Multi-GPU | Limited support | Use single GPU configuration |

### 7.2 Panda3D Limitations

| Issue | Description | Workaround |
|-------|-------------|------------|
| GDI Generic driver | Software rendering only | Install hardware OpenGL drivers |
| Shader compatibility | Some shaders need OpenGL 3.2+ | Use GLSL shaders, not Cg |
| Discrete GPU selection | May use integrated GPU | Set `prefer_discrete_gpu: True` |

### 7.3 Dear PyGui Limitations

| Issue | Description | Workaround |
|-------|-------------|------------|
| 32-bit Python | Not supported | Use 64-bit Python |
| Remote desktop | Limited GPU acceleration | Use local display or VNC |
| Multi-window | Limited multi-monitor support | Configure primary viewport |

### 7.4 Cross-Library Issues

| Issue | Description | Solution |
|-------|-------------|----------|
| NumPy version conflicts | Taichi may require specific NumPy | Pin NumPy to compatible version |
| GPU context sharing | Taichi + Panda3D GPU conflicts | Initialize in correct order |
| Memory pressure | Combined GPU memory usage | Monitor with `nvidia-smi` |

### 7.5 Performance Considerations

| Scenario | Impact | Mitigation |
|----------|--------|------------|
| Running on laptop | Thermal throttling | Use cooling pad, limit to 30 FPS |
| Integrated graphics | No GPU acceleration | Use CPU backend, 1k particles |
| Low VRAM | Out of memory errors | Reduce particle count or resolution |
| Slow storage | Frame drops during logging | Buffer data, write asynchronously |

---

## 8. Quick Reference: Minimum vs Recommended

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 (64-bit) | 3.10 (64-bit) |
| CPU | 4 cores, 3.0 GHz | 8 cores, 3.5 GHz+ |
| RAM | 8 GB | 32 GB |
| GPU | GTX 750 Ti (2 GB) | RTX 3060 (8 GB) |
| CUDA | 10.0+ | 11.x+ |
| OpenGL | 4.3 | 4.6 |
| Storage | 256 GB SSD | 512 GB NVMe |
| OS | Windows 10 / Ubuntu 18.04 | Windows 11 / Ubuntu 22.04 |

---

## 9. Installation Verification Script

Save and run this script to verify your system meets requirements:

```python
#!/usr/bin/env python3
"""System requirements verification for SPH simulation stack."""

import sys
import platform

def check_python():
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    print(f"Python Arch: {platform.architecture()[0]}")
    
    if version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    if platform.architecture()[0] != "64bit":
        print("❌ 64-bit Python required")
        return False
    print("✅ Python version OK")
    return True

def check_taichi():
    try:
        import taichi as ti
        print(f"Taichi Version: {ti.__version__}")
        
        # Try GPU backends
        for arch in [ti.cuda, ti.vulkan, ti.opengl, ti.cpu]:
            try:
                ti.init(arch=arch, log_level=ti.ERROR)
                print(f"✅ Taichi backend: {arch}")
                ti.reset()
                break
            except:
                continue
        return True
    except ImportError:
        print("❌ Taichi not installed")
        return False

def check_panda3d():
    try:
        import panda3d
        print(f"✅ Panda3D installed")
        return True
    except ImportError:
        print("❌ Panda3D not installed")
        return False

def check_dearpygui():
    try:
        import dearpygui.dearpygui as dpg
        print(f"✅ Dear PyGui installed")
        return True
    except ImportError:
        print("❌ Dear PyGui not installed")
        return False

def check_numpy_scipy():
    try:
        import numpy as np
        import scipy
        print(f"NumPy Version: {np.__version__}")
        print(f"SciPy Version: {scipy.__version__}")
        print("✅ NumPy/SciPy OK")
        return True
    except ImportError as e:
        print(f"❌ Missing: {e}")
        return False

def check_h5py():
    try:
        import h5py
        print(f"h5py Version: {h5py.__version__}")
        print("✅ h5py OK")
        return True
    except ImportError:
        print("❌ h5py not installed")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SPH Simulation Stack - System Check")
    print("=" * 50)
    
    results = [
        check_python(),
        check_numpy_scipy(),
        check_taichi(),
        check_panda3d(),
        check_dearpygui(),
        check_h5py(),
    ]
    
    print("=" * 50)
    if all(results):
        print("✅ All checks passed!")
    else:
        print("⚠️ Some checks failed. Review requirements.")
```

---

*Document generated: January 2026*  
*Based on: Taichi Lang v1.6.x, Panda3D 1.10.16, Dear PyGui 1.x*
