#!/bin/bash
# onnxruntime-gpu for Jetson Orin — 一键部署到 Immich ML 容器
# One-shot deploy: replace onnxruntime .so files inside running Immich ML container
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SO_DIR="${SCRIPT_DIR}/mount-so"

echo "=== onnxruntime-jetson-gpu deploy ==="
echo "Target container: immich_machine_learning"
echo ""

# Verify container exists
if ! docker ps --format '{{.Names}}' | grep -q '^immich_machine_learning$'; then
    echo "ERROR: immich_machine_learning container not running"
    exit 1
fi

# Verify .so files
for so in onnxruntime_pybind11_state.so libonnxruntime_providers_cuda.so libonnxruntime_providers_shared.so; do
    if [ ! -f "${SO_DIR}/${so}" ]; then
        echo "ERROR: ${SO_DIR}/${so} not found"
        exit 1
    fi
done

TARGET_DIR="/opt/venv/lib/python3.11/site-packages/onnxruntime/capi"

echo "Copying .so files into container..."
docker cp "${SO_DIR}/onnxruntime_pybind11_state.so" \
    immich_machine_learning:"${TARGET_DIR}/onnxruntime_pybind11_state.cpython-311-aarch64-linux-gnu.so"
docker cp "${SO_DIR}/libonnxruntime_providers_cuda.so" \
    immich_machine_learning:"${TARGET_DIR}/libonnxruntime_providers_cuda.so"
docker cp "${SO_DIR}/libonnxruntime_providers_shared.so" \
    immich_machine_learning:"${TARGET_DIR}/libonnxruntime_providers_shared.so"

echo ""
echo "Restarting container..."
docker restart immich_machine_learning

echo ""
echo "Waiting for healthy..."
sleep 5

echo ""
echo "Verifying GPU availability..."
docker exec immich_machine_learning python3 -c "
import onnxruntime
print('Version:  ', onnxruntime.__version__)
print('Providers:', onnxruntime.get_available_providers())
print('Device:   ', onnxruntime.get_device())
"

echo ""
echo "=== Deploy complete ==="
