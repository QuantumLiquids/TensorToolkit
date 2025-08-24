# 12 GPU 设置与加速

## 依赖
- CUDA 11.0+（含 cuBLAS、cuSOLVER），cuTENSOR 建议安装
- 驱动与工具链版本需与硬件匹配

## 构建
```bash
cmake .. -DQLTEN_USE_GPU=ON \
         -DCUTENSOR_ROOT=/path/to/cutensor  # 若需 cuTENSOR
make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
```

## 使用要点
- 目前 GPU 主要用于矩阵/分解后端（cuBLAS/cuSOLVER）与张量算子（cuTENSOR）。
- 大规模任务建议优先确保数据布局与批量规模，减少 Host-Device 往返。

## 常见问题
- 找不到 CUDA/库：核对环境变量与 `CMAKE_PREFIX_PATH`。
- 性能不升反降：检查是否受 PCIe 传输或小批量调用影响。

