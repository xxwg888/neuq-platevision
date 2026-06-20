# 车牌定位分割与识别算法的设计与实现

面向“车牌定位分割与识别算法的设计与实现”课程设计，做成论文式实验项目：多数据集、多颜色类型、传统方法与深度方法多方对比、可复现实验与高质量可视化。**全部使用真实公开数据集，不含任何合成/GAN 数据。**

## 研究目标

1. 实现蓝牌、绿牌的定位、透视校正、字符分割与识别（课程必做）。
2. 扩展支持黄牌（单/双层）、白牌（警/特种）等复杂类型（课程加分项）。
3. 传统图像处理方法（模板匹配 / HOG+KNN）与轻量深度学习方法（YOLO + CRNN-CTC）多方对比。
4. 统计整牌准确率、字符准确率、Precision/Recall/F1、mAP、编辑距离、运行速度；做分颜色、分省份、错误案例分析。

## 技术路线（双链路）

### 主力深度链路（最终准确率以此为准）

```
整车图 ──► YOLOv8n-pose ──► [检测框 + 颜色类别 + 4角点] ──► 透视校正(warp) ──► CRNN-CTC ──► 整牌字符串
```

- **YOLOv8n-pose**：把“关键点”能力改造为检测车牌 4 个角点（左上/右上/右下/左下），一个模型同时输出检测框、6 类颜色（blue/green/yellow_single/yellow_double/white/black）与角点。
- **透视校正**：用 4 角点 `cv2.getPerspectiveTransform` 把倾斜畸变的车牌拉正为标准矩形，这是斜拍车牌 OCR 高准确率的关键前提。
- **CRNN-CTC**：CNN + 双向 GRU + CTC，对校正后整牌做端到端识别，贪心解码（blank=0）。

### 传统 baseline 链路（满足“字符分割”要求 + 对比实验）

```
车牌裁剪图 ──► 灰度+Otsu二值化+边框裁剪+形态学去噪 ──► 投影法强制切分 N 字 ──► 单字符识别(模板匹配 / HOG+KNN)
```

两条链路共用**同一套字符分割流程**，仅替换识别器，保证对比公平。

## 关键规格速查

| 项 | 值 |
|----|----|
| OCR 输入尺寸 | 3 × 48 × 160 (C×H×W)，RGB |
| OCR 预处理 | BGR→RGB，resize W=160 H=48，`(x/255-0.5)/0.5` |
| OCR 解码 | CTC greedy decode，blank index = 0 |
| 字符表大小 | 76（1 blank + 33 省份 + 26 字母 + 10 数字 + 6 特殊） |
| 校正裁剪尺寸 | warp 到 320×96 存储，载入时 resize 到 160×48 |
| YOLO 输入 | 整车图 imgsz=640 |
| YOLO-pose 输出 | box + 6类颜色 + 4角点(kpt_shape=[4,3], flip_idx=[1,0,3,2]) |

字符表（`src/plate_course/chars.py`）：

- 省份：`京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新港澳`
- 字母：`A-Z`；数字：`0-9`；特殊：`警 学 挂 领 使 临`

## 数据集

| 数据集 | 角色 | 贡献 |
|--------|------|------|
| CCPD2019 | 主训练 | 蓝牌主体；含一定省份多样性（非纯皖 A） |
| CCPD-Green | 主训练 | 绿牌（新能源 8 位） |
| CRPD_all (single/double/multi) | 主训练 | 黄牌单/双层、白牌、复杂道路场景、**全国省份覆盖** |

### 省份感知重划分（关键修复）

CRPD 官方 train/val/test 按**省份做域泛化划分**：云/桂/宁等省几乎只出现在 val/test（如 桂 865 张中 862 张在 test），直接继承会导致这些省训练样本近乎为零、识别失效。

`scripts/crpd_extract_province.py` 对 CRPD 全池**按省份重新做 image-disjoint 70/15/15 划分**（每图整体归一个 split，零图像泄漏），并对各省限幅、稀有类型全保留，使每个省在 train 都有真实代表。修复后云/桂/宁等省整牌准确率从 ~0.02–0.43 恢复到 >0.96（全部真实数据）。

数据稀缺到无法学习的省份（如浙在 CRPD 仅 82 张）由 `build_ocr_manifest.py --min-prov-total` 阈值自动剔除，避免拉低指标；本项目实际阈值下无省份被剔除（最少省份仍 ≥89 个唯一真实训练样本）。

## 复现命令（环境：`conda activate wry`）

### 1. 数据准备

```bash
# CCPD2019 蓝牌 / CCPD-Green 绿牌子集
python scripts/prepare_ccpd_subset.py       --output-dir data/processed/ccpd2019_subset ...
python scripts/prepare_ccpd_green_subset.py  --output-dir data/processed/ccpd_green_subset ...

# CRPD 省份感知抽取（透视校正裁剪 + image-disjoint 重划分）
python scripts/crpd_extract_province.py --cap-prov 400 \
    --output-dir /var/tmp/plate_data_cxj/processed/crpd_province

# 合并为 5 色 OCR manifest（类型过采样 + 省份平衡 + 稀缺省剔除）
python scripts/build_ocr_manifest.py        # 产出 data/processed/ocr_5color/manifests/{train,val,test}.jsonl
```

### 2. YOLOv8n-pose 训练（检测+颜色+角点）

```bash
python scripts/prepare_yolo_pose_dataset.py --output-dir datasets/yolo_pose_5color --manifests ...
python scripts/train_yolo_pose.py --data datasets/yolo_pose_5color/plate_pose.yaml \
    --output-dir outputs/yolo_pose_5color --epochs 100 --imgsz 640 --batch 32 --device 1 --export-onnx
# 产出: outputs/yolo_pose/best.pt(+.onnx)
```

### 3. OCR 训练 + 导出（CRNN-CTC，AMP + 数据增强 + 省份平衡）

```bash
PYTHONPATH=src python scripts/train_crnn_lite.py \
    --train-manifest data/processed/ocr_5color/manifests/train.jsonl \
    --val-manifest   data/processed/ocr_5color/manifests/val.jsonl \
    --output-dir outputs/models/ocr_5color_prov \
    --model crnn_lite --hidden-size 256 --num-layers 2 \
    --epochs 120 --batch-size 256 --lr 1e-3 --num-workers 12 --amp --augment --device cuda

cp outputs/models/ocr_5color_prov/best.pt outputs/models/ocr_best.pt
python scripts/export_onnx.py --checkpoint outputs/models/ocr_best.pt --output outputs/models/ocr_best.onnx
```

### 4. 传统 baseline（对比实验）

```bash
# 模板匹配（NCC，每类均值字形模板）
python scripts/template_baseline.py \
    --train-manifest data/processed/ocr_5color/manifests/train.jsonl \
    --test-manifest  data/processed/ocr_5color/manifests/test.jsonl \
    --output-json outputs/metrics/template_baseline.json

# HOG + KNN
python scripts/knn_baseline.py \
    --train-manifest data/processed/ocr_5color/manifests/train.jsonl \
    --test-manifest  data/processed/ocr_5color/manifests/test.jsonl \
    --output-json outputs/metrics/knn_baseline.json --save-model outputs/models/knn/knn_baseline.xml
```

### 5. 评估 / 推理 / 可视化

```bash
# OCR 综合评估（总体/分类型/分省份/泄漏检查）
python scripts/eval_ocr_full.py --ocr-checkpoint outputs/models/ocr_best.pt

# 端到端全链路（检测 P/R/F1 + 端到端识别）
python scripts/evaluate.py --mode pipeline \
    --yolo-model outputs/yolo_pose/best.pt --ocr-checkpoint outputs/models/ocr_best.pt \
    --test-manifest data/processed/ocr_5color/manifests/test.jsonl \
    --output-json outputs/metrics/eval_pipeline_5color_pose.json

# 单图推理
python scripts/infer.py path/to/car.jpg \
    --yolo-model outputs/yolo_pose/best.pt --ocr-checkpoint outputs/models/ocr_best.pt --device cuda

# 成功/失败案例图 + 指标汇总表
python scripts/make_case_figures.py
python scripts/build_metrics_table.py        # 产出 outputs/metrics/summary_table.md
```

## 结果概览（真实数据测试集，N=2977）

| 方法 | 整牌准确率 | 字符准确率 |
|---|---|---|
| 经典模板匹配 (NCC) | 0.265 | 0.628 |
| HOG + KNN | 0.516 | 0.803 |
| **深度 CRNN-CTC（本文）** | **0.923** | **0.985** |

- 分颜色（识别）：蓝 0.918 / 绿 0.990 / 白 0.938 / 黄 0.937（均 ≥0.92）。
- 颜色分类（检测器多分类头）：整体 0.995，蓝 0.996 / 绿 1.000 / 黄单 0.985 / 黄双 1.000 / 白 1.000。
- 端到端全链路：检测 P 0.99 / R 0.996 / F1 0.99，端到端整牌 0.921、字符 0.985（5 色重训检测后 Recall 由 0.81 升至 0.996，非蓝牌漏检基本消除）。
- 诚实性：完全未见车牌子集整牌仍达 0.897，指标未被实例重叠显著抬高。

完整指标见 `outputs/metrics/summary_table.md`；案例图见 `outputs/figures/{success,failure}_cases.png`。

## 交付物清单

| # | 交付物 | 路径 |
|---|---|---|
| 1 | 数据集解析脚本 | `scripts/prepare_*` / `crpd_extract_province.py` / `build_ocr_manifest.py` |
| 2 | YOLO-pose 训练脚本 | `scripts/train_yolo_pose.py` |
| 3 | OCR 训练脚本 | `scripts/train_crnn_lite.py` |
| 4 | 传统 baseline 脚本 | `scripts/template_baseline.py` / `knn_baseline.py` |
| 5–6 | YOLO-pose 权重/ONNX | `outputs/yolo_pose/best.pt` / `best.onnx` |
| 7–8 | OCR 权重/ONNX | `outputs/models/ocr_best.pt` / `ocr_best.onnx` |
| 9 | 单图推理 | `scripts/infer.py` |
| 10 | 批量评估 | `scripts/evaluate.py` / `eval_ocr_full.py` |
| 11 | 指标结果表 | `outputs/metrics/summary_table.md` + `*.json` |
| 12–13 | 成功/失败案例图 | `outputs/figures/{success,failure}_cases.png` |
| 14 | 说明文档 | `README.md` + `docs/09_server_pipeline.md` + `report/experiment_notes.md` |

## 目录结构

```text
.
├── data/processed/ocr_5color/manifests/   # 最终 5 色 OCR 划分
├── docs/                                  # 方案/数据集/实验设计/服务器链路
├── outputs/
│   ├── models/  ocr_best.pt|onnx, knn/, template/
│   ├── yolo_pose/  best.pt|onnx           # 检测+颜色+角点主模型
│   ├── metrics/  summary_table.md, *.json
│   └── figures/  success_cases.png, failure_cases.png
├── scripts/                               # 数据/训练/推理/评估/可视化
└── src/plate_course/                      # chars/model/dataset/metrics/postprocess/runtime
```
