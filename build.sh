#!/bin/bash
# Build onnxruntime with CUDA for Jetson Orin (ARM64, L4T, JetPack 6.x)
# Prerequisites: GCC 12, CUDA 12.6, cuDNN 9.x, Python 3.11, cmake 3.28+

set -e
ONNX_ROOT="${1:-$HOME/onnxruntime}"
BUILD_DIR="${ONNX_ROOT}/build/Linux/Release"

# Apply source modifications before building:
# 1. Copy src-mod/ files into onnxruntime source tree
# 2. Disable 48 flash_attention .cu files (OOM on 16GB Jetson)
#    cd onnxruntime/contrib_ops/cuda/bert/flash_attention/
#    for f in *.cu; do mv "$f" "${f}.disabled"; done

cmake "${ONNX_ROOT}" \
  -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -Donnxruntime_USE_CUDA=ON \
  -Donnxruntime_USE_TENSORRT=OFF \
  -Donnxruntime_ENABLE_PYTHON=ON \
  -Donnxruntime_BUILD_UNIT_TESTS=OFF \
  -Donnxruntime_USE_FLASH_ATTENTION=OFF \
  -Donnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF \
  -Donnxruntime_USE_FP8_KV_CACHE=OFF \
  -DCMAKE_CXX_FLAGS="-Wno-error=restrict -DDISABLE_ABSEIL" \
  -DCMAKE_CUDA_FLAGS="-ccbin /usr/bin/g++-12 -Xcompiler -DDISABLE_ABSEIL -Xcompiler -Wno-error=attributes" \
  -B "${BUILD_DIR}"

cmake --build "${BUILD_DIR}" --parallel 8

echo "Build complete. Copy .so files to capi:"
cp -v "${BUILD_DIR}/cmake/libonnxruntime_providers_cuda.so" \
      "${BUILD_DIR}/cmake/onnxruntime/capi/"
