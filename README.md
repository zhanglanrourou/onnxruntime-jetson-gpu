# onnxruntime-gpu for Jetson Orin (ARM64)

Prebuilt onnxruntime v1.26.0 with **CUDA 12.6 + cuDNN 9.3** for NVIDIA Jetson Orin NX/Nano (ARM64, JetPack 6.x).

[中文文档 (Chinese README)](README_CN.md)

## Why?

Official Immich ML GPU images are **x86_64 only** — no ARM64 CUDA support. PyPI's `onnxruntime-gpu` wheels are also x86_64 only. Jetson users stuck on CPU inference.

This project provides custom-compiled onnxruntime CUDA provider `.so` files that drop into the Immich ML container to enable GPU-accelerated inference (face recognition, CLIP semantic search, object detection).

## Quick Start

### 1. Download

Download `onnxruntime-jetson-gpu-v1.26.0.tar.gz` from [Releases](https://github.com/zhanglanrourou/onnxruntime-jetson-gpu/releases) and extract:

```
release/
├── mount-so/
│   ├── onnxruntime_pybind11_state.so      (27 MB)
│   ├── libonnxruntime_providers_cuda.so   (72 MB)
│   └── libonnxruntime_providers_shared.so  (8 KB)
├── deploy.sh                              # one-shot deploy script
└── deploy-snippet.yml                     # docker-compose snippet
```

### 2. Deploy

**Option A: One-shot script**

```bash
cd release && chmod +x deploy.sh && ./deploy.sh
```

**Option B: Docker Compose**

Add to `immich-machine-learning` service in your docker-compose.yml:

```yaml
immich-machine-learning:
  devices:
    - nvidia.com/gpu=all
  volumes:
    # Core: custom-built onnxruntime GPU .so files
    - ./release/mount-so/onnxruntime_pybind11_state.so:/opt/venv/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-311-aarch64-linux-gnu.so:ro
    - ./release/mount-so/libonnxruntime_providers_cuda.so:/opt/venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so:ro
    - ./release/mount-so/libonnxruntime_providers_shared.so:/opt/venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so:ro
    # CUDA runtime
    - /usr/local/cuda/lib64:/usr/local/cuda/lib64:ro
    # cuDNN 9.x runtime libraries
    - /usr/lib/aarch64-linux-gnu/libcudnn.so.9:/usr/lib/aarch64-linux-gnu/libcudnn.so.9:ro
    - /usr/lib/aarch64-linux-gnu/libcudnn_adv.so.9:/usr/lib/aarch64-linux-gnu/libcudnn_adv.so.9:ro
    - /usr/lib/aarch64-linux-gnu/libcudnn_cnn.so.9:/usr/lib/aarch64-linux-gnu/libcudnn_cnn.so.9:ro
    - /usr/lib/aarch64-linux-gnu/libcudnn_ops.so.9:/usr/lib/aarch64-linux-gnu/libcudnn_ops.so.9:ro
    - /usr/lib/aarch64-linux-gnu/libcudnn_engines_precompiled.so.9:/usr/lib/aarch64-linux-gnu/libcudnn_engines_precompiled.so.9:ro
    - /usr/lib/aarch64-linux-gnu/libcudnn_engines_runtime_compiled.so.9:/usr/lib/aarch64-linux-gnu/libcudnn_engines_runtime_compiled.so.9:ro
    - /usr/lib/aarch64-linux-gnu/libcudnn_heuristic.so.9:/usr/lib/aarch64-linux-gnu/libcudnn_heuristic.so.9:ro
    - /usr/lib/aarch64-linux-gnu/libcudnn_graph.so.9:/usr/lib/aarch64-linux-gnu/libcudnn_graph.so.9:ro
  environment:
    LD_LIBRARY_PATH: /opt/venv/lib/python3.11/site-packages/onnxruntime/capi:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu
    HF_ENDPOINT: https://hf-mirror.com  # Use mirror if huggingface.co is blocked
```

Restart:

```bash
docker compose up -d --force-recreate immich-machine-learning
```

### 3. Verify

```bash
docker exec immich_machine_learning python3 -c "
import onnxruntime
print('Version:  ', onnxruntime.__version__)
print('Providers:', onnxruntime.get_available_providers())
print('Device:   ', onnxruntime.get_device())
"
```

Expected output:

```
Version:   1.26.0
Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Device:    GPU
```

## Build from Source

### Prerequisites

- Jetson Orin NX/Nano, JetPack 6.x (L4T R36+)
- GCC 12 (`apt install gcc-12 g++-12`)
- CUDA 12.6 (JetPack default)
- cuDNN 9.x (JetPack default)
- Python 3.11 (`apt install python3.11-dev`)
- cmake 3.28+ (`pip install cmake`)
- Ninja (`apt install ninja-build`)

### Build Steps

```bash
# 1. Get onnxruntime source (via Immich)
git clone https://github.com/immich-app/immich.git --branch v3.0.3

# 2. Apply patches
cp src-mod/include/onnxruntime/core/common/inlined_containers.h \
   immich/machine-learning/onnxruntime/onnxruntime/core/common/
cp src-mod/onnxruntime/core/providers/tensorrt/tensorrt_execution_provider.cc \
   immich/machine-learning/onnxruntime/onnxruntime/core/providers/tensorrt/

# 3. Disable Flash Attention .cu files (OOM on 16GB Jetson)
cd immich/machine-learning/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention/
for f in *.cu; do mv "$f" "${f}.disabled"; done

# 4. Build
cd immich/machine-learning/onnxruntime
./build.sh ~/immich/machine-learning/onnxruntime

# 5. Output
ls onnxruntime/build/Linux/Release/cmake/onnxruntime/capi/*.so
ls onnxruntime/build/Linux/Release/cmake/libonnxruntime_providers_*.so
```

Build time: ~30 min on Jetson Orin NX 16GB (`-j8`), ~1147 targets.

## Issues Solved

| Problem | Solution |
|---------|----------|
| `absl::flat_hash_map` template errors (GCC 12 + C++20) | `-DDISABLE_ABSEIL` — use `std::unordered_map` with transparent hash |
| `InlinedVector<bool>` → `span<bool>` conversion | Change to `InlinedVector<char>` (TensorRT provider only) |
| `std::hash<char[N]>` not defined | `InlinedHash` struct with array decay + `const char*` overload |
| Flash Attention OOM (48 `.cu` files, 15GB RAM) | `-Donnxruntime_USE_FLASH_ATTENTION=OFF` |
| cutlass `[[nodiscard]]` void function warnings | `-Xcompiler -Wno-error=attributes` |
| CLIP model download blocked (China) | `HF_ENDPOINT=https://hf-mirror.com` |
| cuDNN 9.x not loading in container | Mount all 8 `libcudnn_*.so.9` + set `LD_LIBRARY_PATH` |

### Key CMake Flags

```bash
-DCMAKE_C_COMPILER=/usr/bin/gcc-12
-DCMAKE_CXX_COMPILER=/usr/bin/g++-12
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_CUDA_ARCHITECTURES=87
-Donnxruntime_USE_CUDA=ON
-Donnxruntime_USE_TENSORRT=OFF
-Donnxruntime_ENABLE_PYTHON=ON
-Donnxruntime_BUILD_UNIT_TESTS=OFF
-Donnxruntime_USE_FLASH_ATTENTION=OFF
-Donnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF
-Donnxruntime_USE_FP8_KV_CACHE=OFF
-DCMAKE_CXX_FLAGS="-Wno-error=restrict -DDISABLE_ABSEIL"
-DCMAKE_CUDA_FLAGS="-ccbin /usr/bin/g++-12 -Xcompiler -DDISABLE_ABSEIL -Xcompiler -Wno-error=attributes"
```

### Docker GPU Setup (JetPack 6.x)

JetPack 6.x uses NVIDIA CDI (Container Device Interface) instead of `--runtime=nvidia`:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
docker run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi
```

## Project Structure

```
onnxruntime-jetson-gpu/
├── README.md              # English docs (this file)
├── README_CN.md           # Chinese docs
├── build.sh               # Build script
├── patches/               # Source patches
├── src-mod/               # Modified source files
└── release/               # Release packaging
    ├── mount-so/          # Compiled .so files
    ├── deploy.sh          # One-shot deploy script
    └── deploy-snippet.yml # docker-compose template
```

## Credits

Based on onnxruntime v1.26.0 from [Microsoft/onnxruntime](https://github.com/microsoft/onnxruntime).
Built for [Immich](https://github.com/immich-app/immich) v3.0.3 on Jetson Orin NX 16GB, JetPack 6.2.2 / L4T R36.5.0.
