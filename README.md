- # Agri-DINOv3

  面向**农业病虫害识别**的视觉基础模型，基于 [DINOv3 ViT-B/16](https://github.com/facebookresearch/dinov3) 在农业场景下微调/训练。本仓库提供：

  * **评测**：在 12 个农业下游分类 benchmark 上做统一线性探测（linear probe）
  * **推理**：对单张图片或图片文件夹提取视觉特征（768-d CLS embedding）

  ## 性能对比

  在统一线性探测协议下，与现有视觉基础模型在 14 个农业下游数据集上的 **Micro-F1** 对比（越高越好，上限 100%）。**Ours** 为本仓库 Agri-DINOv3 模型。

  | 序号 | 数据集             | AgriCLIP | BioCLIP | DINOv2 | DINOv3  |  ViT   |  OpenCLIP  |  **Ours**   |
  | :--: | ------------------ | :------: | :-----: | :----: | :-----: | :----: | :--------: | :---------: |
  |  1   | plantdoc           |  50.03%  | 68.08%  | 66.34% | 66.53%  | 66.80% |   63.56%   | **70.59%**  |
  |  2   | agrivision4        |  90.96%  | 95.61%  | 95.13% | 95.28%  | 96.20% |   92.96%   | **97.63%**  |
  |  3   | corn\_leaf         |  91.00%  | 94.65%  | 94.05% | 94.74%  | 94.87% | **95.11%** |   94.88%    |
  |  4   | seasonal\_corn     |  57.95%  | 92.95%  | 91.14% | 91.26%  | 91.30% |   86.29%   | **96.19%**  |
  |  5   | cucurbit           |  85.51%  | 97.44%  | 97.05% | 97.64%  | 96.64% |   92.17%   | **98.09%**  |
  |  6   | manalagi\_apple    |  94.96%  | 98.81%  | 99.41% | 100.00% | 99.70% |   98.81%   | **100.00%** |
  |  7   | rice\_leaf         |  82.93%  | 95.70%  | 94.68% | 95.63%  | 95.22% |   93.10%   | **96.81%**  |
  |  8   | multicrop          |  25.36%  | 77.78%  | 75.95% | 78.37%  | 72.75% |   57.06%   | **84.31%**  |
  |  9   | tcp                |  43.78%  | 70.00%  | 69.90% | 71.19%  | 70.26% |   62.44%   | **75.60%**  |
  |  10  | crops\_leafs       |  83.88%  | 95.87%  | 95.97% | 97.04%  | 95.29% |   93.16%   | **98.11%**  |
  |  11  | corn\_pests\_early |  79.49%  | 94.06%  | 95.03% | 95.32%  | 94.73% |   92.93%   | **95.70%**  |
  |  12  | tom24              |  56.10%  | 81.73%  | 83.14% | 85.57%  | 83.49% |   77.64%   | **88.21%**  |

  > Micro-F1 衡量样本级分类性能；在 12 个数据集上，Ours 在绝大多数 benchmark 上优于 AgriCLIP、BioCLIP、DINOv2、DINOv3、ViT 与 OpenCLIP 等基线。

  ## 环境准备

  ```bash
  pip install -r requirements.txt
  
  # DINOv3 模型结构（必需）
  git clone https://github.com/facebookresearch/dinov3.git
  export DINOV3_REPO=/path/to/dinov3
  
  # 评测所需数据集（推理不需要）
  export AGRI_ROOT=/path/to/Agri-dataset
  ```

  进入仓库目录：

  ```bash
  cd Agri-dino-v3-260621
  export PYTHONPATH="${PWD}:${PYTHONPATH}"
  ```

  ## 推理

  对单张图片或整个文件夹提取特征，输出 `.npz`（默认）或 `.json`：

  ```bash
  # 单张图片
  CHECKPOINT=./ckpt/agri-dinov3.pth \
  INPUT=/path/to/leaf.jpg \
  OUTPUT=results/leaf_features.npz \
  bash run_infer_single.sh
  
  # 整个文件夹
  CHECKPOINT=./ckpt/agri-dinov3.pth \
  INPUT=/path/to/images/ \
  OUTPUT=results/batch_features.npz \
  bash run_infer_single.sh
  ```

  等价 Python 命令：

  ```bash
  python infer_features.py \
    --input /path/to/image_or_folder \
    --checkpoint ./ckpt/agri-dinov3.pth \
    --output results/features.npz \
    --batch-size 32
  ```

  输出文件包含：

  * `paths`：图片路径列表
  * `features`：形状 `[N, 768]` 的 L2 归一化 CLS 特征

  ## 评测

  在 14 个农业病虫害 / 作物病害数据集上评测 checkpoint（30% train / 70% test，seed=42）：

  `plantdoc`, `agrivision4`, `corn_leaf`, `seasonal_corn`, `cucurbit`, `manalagi_apple`, `rice_leaf`, `multicrop`, `tcp`, `crops_leafs`, `corn_pests_early`, `tom24`, `wcs_cucumber`, `wcs_tomato`

  ### 全量 benchmark

  ```bash
  CHECKPOINT=./ckpt/agri-dinov3.pth \
  CUDA_VISIBLE_DEVICES=0 \
  bash run_dino_cls_single.sh
  ```

  ### 单个数据集快速验证

  ```bash
  python run_eval.py \
    --models dinov3_vitb16 \
    --datasets plantdoc \
    --checkpoint ./ckpt/agri-dinov3.pth \
    --batch-size 64 \
    --workers 8 \
    --use-cache
  ```

  ### 汇总结果为 CSV

  ```bash
  python scripts/summarize_results.py \
    --metrics-path results/metrics_dinov3_cls.jsonl \
    --output results/summary_dinov3_cls.csv
  ```

  ### 常用环境变量

  | 变量            | 说明                                              |
  | --------------- | ------------------------------------------------- |
  | `CHECKPOINT`    | 模型权重 `.pth`（必填）                           |
  | `DATASETS`      | `all` 或逗号分隔数据集名，如 `plantdoc,corn_leaf` |
  | `BATCH_SIZE`    | 特征提取 batch size，默认 `64`                    |
  | `USE_CACHE`     | `1` 复用 `cache/` 中已提取特征，默认开启          |
  | `REFRESH_CACHE` | `1` 强制重新提取特征                              |

  评测结果写入：

  * `results/metrics_dinov3_cls.jsonl` — 每个数据集的 acc / macro\_f1
  * `results/summary_dinov3_cls.csv` — accuracy 透视表
  * `results/summary_macro_f1.csv` — macro F1 透视表

  ## 评测协议

  * 冻结 backbone，提取 `[CLS]` 特征（L2 归一化）
  * 在训练特征上拟合 `LogisticRegression`，测试集报告 Top-1 Acc 与 Macro F1
  * 固定划分见 `data/splits/{dataset}_seed42.json`

  ## 目录结构

  ```
  ├── run_infer_single.sh        # 推理入口
  ├── infer_features.py          # 特征提取
  ├── run_dino_cls_single.sh     # 评测入口
  ├── run_eval.py                # benchmark 主脚本
  ├── backbones/dinov3.py        # 模型加载（支持 DirectFinetune 等 checkpoint 格式）
  ├── configs/models.yaml
  ├── data/splits/               # 固定 train/test 划分
  └── scripts/summarize_results.py
  ```

  ## 引用

  * DINOv3: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
