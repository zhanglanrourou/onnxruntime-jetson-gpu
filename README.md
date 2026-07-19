# onnxruntime-gpu for Jetson Orin (ARM64)

Prebuilt onnxruntime v1.26.0 with **CUDA 12.6 + cuDNN 9.3** for NVIDIA Jetson Orin NX/Nano (ARM64, L4T/JetPack 6.x).

## Quick Start

Download the `.so` files from [Releases]() and mount into Immich ML container:

```yaml
# docker-compose.yml additions for immich-machine-learning:
devices:
  - nvidia.com/gpu=all
volumes:
  - ./onnxruntime_pybind11_state.cpython-311-aarch64-linux-gnu.so:/opt/venv/lib/python3.11/site-packages/onnxruntime/capi/...
  - ./libonnxruntime_providers_cuda.so:/opt/venv/lib/python3.11/site-packages/onnxruntime/capi/...
  - /usr/local/cuda/lib64:/usr/local/cuda/lib64:ro
  - /usr/lib/aarch64-linux-gnu/libcudnn.so.9:/usr/lib/aarch64-linux-gnu/libcudnn.so.9:ro
environment:
  LD_LIBRARY_PATH: /usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu
```

## Build from Source

### Why Custom Build?

Official Immich ML GPU images are **x86_64 only** — no ARM64 CUDA support. Stock `onnxruntime-gpu` wheels on PyPI are also x86_64 only. Jetson pre-built wheels from dusty-nv/jetson-containers are Python 3.10 only (Immich ML uses 3.11).

### Issues Solved

| Problem | Solution |
|---------|----------|
| `absl::flat_hash_map` template errors (GCC 12 + C++20) | `DISABLE_ABSEIL` — use `std::unordered_map` with transparent hash |
| `InlinedVector<bool>` → `span<bool>` conversion | Change to `InlinedVector<char>` (TensorRT provider only) |
| `std::hash<char[N]>` not defined | `InlinedHash` struct with array decay + `const char*` overload |
| Flash Attention OOM (48 `.cu` files, 15GB RAM) | `-Donnxruntime_USE_FLASH_ATTENTION=OFF` |
| cutlass `nodiscard` void function warnings | `-Xcompiler -Wno-error=attributes` |
| hf-mirror.com for CLIP model download | `wget https://hf-mirror.com/immich-app/ViT-B-32__openai/...` |

### Prerequisites

- Jetson Orin NX/Nano, JetPack 6.x (L4T R36+)
- GCC 12 (`apt install gcc-12 g++-12`)
- CUDA 12.6 (JetPack default)
- cuDNN 9.3 (JetPack default)
- Python 3.11 (`apt install python3.11-dev`)
- cmake 3.28+ (`pip install cmake`)
- Ninja (`apt install ninja-build`)

### Build Steps

```bash
git clone https://github.com/immich-app/immich.git --branch v3.0.3
# Apply source patches from src-mod/
# Disable flash_attention .cu files (OOM on 16GB)
find onnxruntime/contrib_ops/cuda/bert/flash_attention/ -name "*.cu" -exec mv {} {}.disabled \;
# Run build
./build.sh ~/onnxruntime
```

### Modifications

See `src-mod/` for modified source files:
- `inlined_containers.h` — `InlinedHash` transparent hash for DISABLE_ABSEIL path
- `tensorrt_execution_provider.cc` — `bool` → `char` vector fix

## Deploy to Immich

```bash
# 1. Copy .so files
cp onnxruntime/build/Linux/Release/cmake/onnxruntime/capi/*.so /mnt/data/immich/

# 2. Download CLIP model
sudo wget -O /mnt/data/docker/volumes/immich_model-cache/_data/clip/ViT-B-32__openai/visual/model.onnx \
  https://hf-mirror.com/immich-app/ViT-B-32__openai/resolve/main/visual/model.onnx

# 3. Update docker-compose.yml (see example above)

# 4. Restart
cd /mnt/data/immich && docker compose up -d --force-recreate immich-machine-learning

# 5. Verify
docker exec immich_machine_learning python3 -c "
import onnxruntime; print(onnxruntime.get_available_providers())
"
# Expected: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## Credits

Based on onnxruntime v1.26.0 from Microsoft/onnxruntime.
Built on Jetson Orin NX 16GB, JetPack 6.2.2 / L4T R36.5.0.
