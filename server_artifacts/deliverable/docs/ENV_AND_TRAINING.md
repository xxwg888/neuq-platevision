# 环境、训练超参与原始训练记录

## 1. 运行环境(服务器实测)

| 组件 | 版本 |
|---|---|
| OS | Linux,4× RTX 4090(24 GB) |
| Python | 3.10.4 |
| PyTorch | 2.4.0 + cu121 |
| torchvision | 0.19.0 + cu121 |
| CUDA | 12.1 |
| ultralytics | 8.4.71 |
| opencv-python | 4.13.0 |
| onnx / onnxruntime | 1.22.0 / 1.23.2 |
| numpy / pillow | 2.2.6 / 12.0.0 |

完整依赖见 `deliverable/requirements.txt`;CPU 推理见 `deliverable/requirements-cpu-inference.txt`。

## 2. 技术路线确认(重要)

本项目为**自训练**的 **YOLOv8n-pose(检测+颜色+4角点) + CRNN-CTC(自训练 OCR)** 双模型路线,
**不是 HyperLPR3、不是任何第三方现成车牌库**。证据:

- 训练脚本:`scripts/train_yolo_pose.py`、`scripts/train_crnn_lite.py`(均在本包内);
- 自训练权重:`outputs/yolo_pose/best.pt`、`outputs/models/ocr_best.pt`;
- 原始训练记录(下方第 4 节)含逐 epoch 曲线、超参、seed,可完全复现;
- 模型结构:`src/plate_course/model.py`(CRNN = CNN + 双向 GRU + CTC,2,560,236 参数)。
- **全程仅用真实公开数据(CCPD2019 + CCPD-Green + CRPD),无任何合成/GAN 数据。**

## 3. 训练超参

### 3.1 检测 YOLOv8n-pose

| 超参 | 值 |
|---|---|
| 预训练 | yolov8n-pose.pt |
| epochs | 100 |
| batch | 32 |
| imgsz | 640 |
| lr0 | 0.01 |
| patience | 30 |
| seed | 2026 |
| device | 单卡(GPU 1) |
| kpt_shape / flip_idx | [4,3] / [1,0,3,2] |
| 类别 | 6(blue/green/yellow_single/yellow_double/white/black) |

最终验证集:Precision 0.9155 / Recall 0.8796 / box mAP50 **0.9448** / box mAP50-95 0.8020 / 角点 pose mAP50 **0.9450**。

### 3.2 识别 CRNN-CTC

| 超参 | 值 |
|---|---|
| 模型 | crnn_lite(CNN + BiGRU×2 + CTC) |
| 参数量 | 2,560,236 |
| epochs | 120 |
| batch_size | 256 |
| lr | 1e-3(余弦衰减) |
| weight_decay | 1e-4 |
| hidden_size / num_layers | 256 / 2 |
| 输入 | 3×48×160 RGB,(x/255−0.5)/0.5 |
| AMP / 数据增强 | 开 / 开 |
| seed | 2026 |
| device | cuda |
| num_classes | 76(1 blank + 33 省 + 26 字母 + 10 数字 + 6 特殊) |

最终验证集:plate_accuracy 0.9268 / character_accuracy 0.9848 / avg_edit_distance 0.106。

## 4. 原始训练记录文件(均在本包 `training_records/`)

| 文件 | 内容 |
|---|---|
| `yolo_pose/results.csv` | YOLO 逐 epoch 全指标(loss/P/R/mAP) |
| `yolo_pose/results.png` | YOLO 训练曲线总图 |
| `yolo_pose/Box{P,R,F1,PR}_curve.png` | 检测 P/R/F1/PR 曲线 |
| `yolo_pose/Pose{P,R,F1,PR}_curve.png` | 角点 P/R/F1/PR 曲线 |
| `yolo_pose/confusion_matrix*.png` | 颜色类别混淆矩阵 |
| `yolo_pose/args.yaml` | YOLO 完整训练参数快照 |
| `yolo_pose/val_batch*_pred.jpg` | 验证集预测可视化 |
| `crnn/train_log.csv` | CRNN 逐 epoch(train/val 的 loss、plate_acc、char_acc、edit_dist、fps) |
| `crnn/model_config.json` | CRNN 结构 + 字符表 + 归一化/解码 + 全部训练 args |
| `crnn/val_examples.json` | CRNN 验证集预测样例 |

> CRNN 没有自带 loss 曲线图,可用 `train_log.csv` 直接画(列名见文件首行)。
> 如需,运行:`python -c "import pandas..."` 或用 Excel 打开 `train_log.csv` 绘图。
