# Build Configuration v17 (Final Working)

## cmake flags
```
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

## Build targets
- Total: 1147 (v17, without flash attention)
- Time: ~30 min on Jetson Orin NX 16GB (-j8)
- Output: 3 .so files (72MB + 27MB + 8KB)

## Tested
- Host: Python 3.11.15 → `onnxruntime.get_available_providers()` = ['CUDAExecutionProvider', 'CPUExecutionProvider']
- Container: Immich ML v3.0.3 → CLIP ViT-B-32 GPU inference confirmed
