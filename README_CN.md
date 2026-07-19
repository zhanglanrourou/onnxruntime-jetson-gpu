# onnxruntime-gpu for Jetson Orin (ARM64)

为 NVIDIA Jetson Orin NX/Nano 预编译的 onnxruntime v1.26.0，**CUDA 12.6 + cuDNN 9.3**，用于 Immich ML GPU 推理。

## 为什么需要这个？

Immich 官方的 ML GPU 镜像只支持 **x86_64**，没有 ARM64 CUDA 版本。PyPI 上的 `onnxruntime-gpu` 也只提供 x86_64 的 wheel。Jetson 用户如果直接用官方镜像，所有 AI 推理（人脸识别、CLIP 语义搜索、物体检测）都只能跑 CPU，速度极慢。

这个项目提供了在 Jetson Orin 上自行编译的 onnxruntime CUDA provider `.so` 文件，替换进 Immich ML 容器后即可启用 GPU 推理。

## 适用环境

| 组件 | 版本 |
|------|------|
| 设备 | Jetson Orin NX / Nano (ARM64) |
| 系统 | JetPack 6.x (L4T R36+) |
| CUDA | 12.6 (JetPack 自带) |
| cuDNN | 9.3 (JetPack 自带) |
| Python | 3.11 (Immich ML 容器内) |
| Immich | v3.0.3 |
| onnxruntime | v1.26.0 |

## 快速开始

### 1. 下载 Release

从 [Releases](https://github.com/zhanglanrourou/onnxruntime-jetson-gpu/releases) 下载 `onnxruntime-jetson-gpu-v1.26.0.tar.gz`，解压后目录结构：

```
release/
├── mount-so/
│   ├── onnxruntime_pybind11_state.so      (27 MB)
│   ├── libonnxruntime_providers_cuda.so   (72 MB)
│   └── libonnxruntime_providers_shared.so  (8 KB)
├── deploy.sh                              # 一键部署脚本
└── deploy-snippet.yml                     # docker-compose 部署片段
```

### 2. 部署

**方法A：一键脚本（推荐）**

```bash
cd release
chmod +x deploy.sh
./deploy.sh
```

**方法B：手动 docker-compose 部署**

在 immich 的 `docker-compose.yml` 中，给 `immich-machine-learning` 服务添加以下配置：

```yaml
immich-machine-learning:
  devices:
    - nvidia.com/gpu=all
  volumes:
    # 核心：替换 onnxruntime 的 .so 文件
    - ./release/mount-so/onnxruntime_pybind11_state.so:/opt/venv/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-311-aarch64-linux-gnu.so:ro
    - ./release/mount-so/libonnxruntime_providers_cuda.so:/opt/venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so:ro
    - ./release/mount-so/libonnxruntime_providers_shared.so:/opt/venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so:ro
    # CUDA runtime
    - /usr/local/cuda/lib64:/usr/local/cuda/lib64:ro
    # cuDNN 9.x
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
    HF_ENDPOINT: https://hf-mirror.com  # 国内用户加速模型下载
```

然后重启容器：

```bash
cd /mnt/data/immich
docker compose up -d --force-recreate immich-machine-learning
```

### 3. 验证

```bash
docker exec immich_machine_learning python3 -c "
import onnxruntime
print('Version:  ', onnxruntime.__version__)
print('Providers:', onnxruntime.get_available_providers())
print('Device:   ', onnxruntime.get_device())
"
```

期望输出：

```
Version:   1.26.0
Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Device:    GPU
```

看到 `CUDAExecutionProvider` 就说明 GPU 推理已启用。

## 从源码构建

### 前置依赖

```bash
# 基础编译工具
sudo apt install -y gcc-12 g++-12 ninja-build python3.11-dev

# cmake 3.28+（pip 安装）
pip install cmake

# Docker GPU CDI 配置
sudo nvidia-ctk cdi generate
```

### 构建步骤

```bash
# 1. 获取源码
git clone https://github.com/immich-app/immich.git --branch v3.0.3
# Immich v3.0.3 包含 onnxruntime v1.26.0 源码

# 2. 应用源码补丁
cp src-mod/include/onnxruntime/core/common/inlined_containers.h \
   immich/machine-learning/onnxruntime/onnxruntime/core/common/
cp src-mod/onnxruntime/core/providers/tensorrt/tensorrt_execution_provider.cc \
   immich/machine-learning/onnxruntime/onnxruntime/core/providers/tensorrt/

# 3. 禁用 Flash Attention（48 个 .cu 文件，16GB Jetson 编译 OOM）
cd immich/machine-learning/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention/
for f in *.cu; do mv "$f" "${f}.disabled"; done

# 4. 编译
cd immich/machine-learning/onnxruntime
./build.sh ~/immich/machine-learning/onnxruntime

# 5. 产出位置
ls onnxruntime/build/Linux/Release/cmake/onnxruntime/capi/*.so
ls onnxruntime/build/Linux/Release/cmake/libonnxruntime_providers_*.so
```

编译时间约 30 分钟（Jetson Orin NX 16GB，`-j8`）。

## 踩坑记录

以下是在 Jetson Orin NX 16GB + JetPack 6.2.2 上实际编译遇到的问题和解决方案：

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `absl::flat_hash_map` 模板报错 | GCC 12 + C++20 与 Abseil 不兼容 | 添加 `-DDISABLE_ABSEIL`，用 `std::unordered_map` 替代 |
| `InlinedVector<bool>` → `span<bool>` 转换失败 | `vector<bool>` 特化导致 | 改为 `InlinedVector<char>`（仅 TensorRT provider） |
| `std::hash<char[N]>` 未定义 | GCC 对定长字符数组 hash 支持不足 | 实现 `InlinedHash` 结构体，数组退化为 `const char*` |
| Flash Attention 编译 OOM | 48 个 `.cu` 文件并行编译吃满 16GB 内存 | `-Donnxruntime_USE_FLASH_ATTENTION=OFF` 并禁用所有相关 `.cu` |
| cutlass `nodiscard` void 函数警告 | GCC 对 `[[nodiscard]]` 更严格 | `-Xcompiler -Wno-error=attributes` |
| 官方 wheel 没有 CUDA provider | PyPI 上的 onnxruntime-gpu 只有 x86_64 | 必须从源码编译，替换 `.so` |
| Immich ML 容器内无法下载 CLIP 模型 | huggingface.co 国内访问受限 | `HF_ENDPOINT=https://hf-mirror.com` |
| `libcudnn.so` 加载失败 | 容器内未挂载 cuDNN 9.x 的所有依赖库 | 挂载 8 个 cuDNN `.so.9` 文件 + 设置 `LD_LIBRARY_PATH` |

### 关键 CMake 参数

```bash
-DCMAKE_C_COMPILER=/usr/bin/gcc-12
-DCMAKE_CXX_COMPILER=/usr/bin/g++-12
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_CUDA_ARCHITECTURES=87          # Orin GPU 架构 SM 8.7
-Donnxruntime_USE_CUDA=ON
-Donnxruntime_USE_TENSORRT=OFF
-Donnxruntime_ENABLE_PYTHON=ON
-Donnxruntime_BUILD_UNIT_TESTS=OFF
-Donnxruntime_USE_FLASH_ATTENTION=OFF
-Donnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF
-Donnxruntime_USE_FP8_KV_CACHE=OFF
```

### Docker GPU CDI 配置

JetPack 6.x 使用 NVIDIA CDI (Container Device Interface) 而非旧的 `--runtime=nvidia`：

```bash
# 生成 CDI 配置
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# 检查 GPU 可见性
docker run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi
```

## 项目结构

```
onnxruntime-jetson-gpu/
├── README.md              # 英文文档
├── README_CN.md           # 中文文档（本文件）
├── build.sh               # 构建脚本
├── patches/               # 源码补丁
├── src-mod/               # 修改后的源文件
├── release/               # Release 打包目录
│   ├── mount-so/          # 编译产物 .so 文件
│   ├── deploy.sh          # 一键部署脚本
│   └── deploy-snippet.yml # docker-compose 模板
└── onnxruntime-jetson-gpu-v1.26.0.tar.gz
```

## 鸣谢

- [Microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) — ONNX Runtime 引擎
- [immich-app/immich](https://github.com/immich-app/immich) — 自托管照片管理
- [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers) — Jetson ML 基础镜像参考

---

构建于 Jetson Orin NX 16GB，JetPack 6.2.2 / L4T R36.5.0，2026-07-19。
