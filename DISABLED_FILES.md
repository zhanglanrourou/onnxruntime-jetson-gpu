# Disabled CUDA Files (48 flash_attention .cu files)

These 48 files are **LLM/BERT-specific Flash Attention kernels**.
They cause OOM (out of memory) kills during compilation on 16GB Jetson Orin.
Immich ML only uses CLIP models — Flash Attention is not needed.

Path: `onnxruntime/contrib_ops/cuda/bert/flash_attention/`

## Files disabled:
flash_fwd_hdim128_bf16_sm80.cu
flash_fwd_hdim128_fp16_sm80.cu
flash_fwd_hdim192_bf16_sm80.cu
flash_fwd_hdim192_fp16_sm80.cu
flash_fwd_hdim256_bf16_sm80.cu
flash_fwd_hdim256_fp16_sm80.cu
flash_fwd_hdim32_bf16_sm80.cu
flash_fwd_hdim32_fp16_sm80.cu
flash_fwd_hdim64_bf16_sm80.cu
flash_fwd_hdim64_fp16_sm80.cu
flash_fwd_hdim96_bf16_sm80.cu
flash_fwd_hdim96_fp16_sm80.cu
flash_fwd_hdim128_bf16_causal_sm80.cu
... (36 more causal/split variants)
flash_fwd_split_hdim96_fp16_sm80.cu

## Controlled by cmake flag:
-Donnxruntime_USE_FLASH_ATTENTION=OFF
