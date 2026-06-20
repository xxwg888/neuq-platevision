# 车牌定位、分割与识别 — 交付包说明

> 全部基于**真实公开数据集**(CCPD2019 + CCPD-Green + CRPD),**不含任何合成/GAN 数据**。
> 技术路线为**自训练** YOLOv8n-pose(检测+颜色+4角点) + CRNN-CTC(自训练 OCR),**非 HyperLPR3、非任何第三方现成库**。

## 0. 最终指标(真实测试集 N=2977,服务器实测可复现)

| 指标 | 数值 |
|---|---|
| 端到端整牌准确率 | **0.9206** |
| 端到端字符准确率 | **0.9852** |
| 检测 P / R / F1(IoU≥0.5) | 0.9855 / 0.9959 / 0.9907 |
| 颜色分类整体准确率 | 0.9952 |
| 检测 box mAP50 / 角点 pose mAP50 | 0.9438 / 0.9450 |
| 三方对比(整牌) | 模板 0.265 / KNN 0.516 / **深度 0.923** |

一条命令复现 0.9206 / 0.9852 见 `docs/REPRODUCE.md`。

## 1. 目录结构

```
deliverable/
├── PACKAGE_README.md              # 本文件
├── README.md                      # 项目主说明(技术路线/数据/复现)
├── charset.txt                    # 76 类字符表,第0行=<blank>,行号=类别索引
├── requirements.txt               # 完整环境(训练+评估+推理)
├── requirements-cpu-inference.txt # 仅 CPU/ONNX 推理依赖
├── data_stats.json / data_stats.md# train/val/test × 颜色/来源/省份 数量统计
├── test_image_list.csv            # N=2977 测试集逐条清单(id/车牌号/颜色/来源/文件名)
├── docs/
│   ├── REPRODUCE.md               # 复现命令 + 预期输出
│   ├── ONNX_IO.md                 # ONNX 输入输出格式 + CTC解码 + CPU 推理
│   └── ENV_AND_TRAINING.md        # 版本/超参/seed + 训练记录索引 + 路线确认
├── scripts/                       # 全部代码:数据/训练/推理/评估/可视化
├── src/plate_course/              # 核心库:chars/model/dataset/metrics/postprocess/runtime
├── configs/                       # YOLO/CRNN 训练配置快照 + plate_pose.yaml
├── models/
│   ├── yolo_pose/ best.pt, best.onnx
│   └── ocr/       ocr_best.pt, ocr_best.onnx
├── metrics/                       # 全部原始 JSON + summary_table.md
├── figures/
│   ├── success_cases.png, failure_cases.png
│   └── ppt/  pipeline_diagram / perspective_demo / segmentation_demo / crnn_training_curves
├── training_records/
│   ├── yolo_pose/ results.csv, 各曲线png, 混淆矩阵, args.yaml, val预测图
│   └── crnn/      train_log.csv, model_config.json, val_examples.json
└── report/        车牌识别完整报告.md, experiment_notes.md
```

> 注:`models/` 与 `metrics/` 同时也按原工程路径放在 `outputs/` 下,使文档里的命令开箱即用。

## 2. 各交付物对应你的清单

| 你要的 | 位置 |
|---|---|
| 1. 推理代码(单图→JSON+可视化) | `scripts/infer.py`(加 `--vis-dir` 出可视化);CPU 版 `scripts/onnx_infer_demo.py` |
| 2. 评估代码 | `scripts/evaluate.py` / `eval_ocr_full.py` / `color_eval.py` |
| 3. YOLO-pose 权重 | `models/yolo_pose/best.pt` + `best.onnx` |
| 4. OCR 权重 | `models/ocr/ocr_best.pt` + `ocr_best.onnx` |
| 5. 字符表(含 blank 与顺序) | `charset.txt`(第0行 `<blank>`) |
| 6. 测试集 manifest + 数量统计 | `data/.../test.jsonl`(工程内)、`test_image_list.csv`、`data_stats.md` |
| 7. metrics 全部 JSON + 汇总表 | `metrics/*.json` + `metrics/summary_table.md` |
| 8. 成功/失败案例图 | `figures/success_cases.png` / `failure_cases.png` |
| 9. requirements | `requirements.txt` / `requirements-cpu-inference.txt` |
| 10. 复现命令 | `docs/REPRODUCE.md` |
| 完整代码目录 | `scripts/ src/ configs/` |
| 原始训练记录 | `training_records/`(YOLO results.csv + CRNN train_log.csv + 全部曲线) |
| 数据统计 | `data_stats.md` / `test_image_list.csv` |
| PPT 素材 | `figures/ppt/` + `training_records/yolo_pose/*.png` + `metrics/summary_table.md` |

## 3. 单图推理(最常用)

GPU/.pt:
```bash
python scripts/infer.py car.jpg \
    --yolo-model models/yolo_pose/best.pt \
    --ocr-checkpoint models/ocr/ocr_best.pt \
    --device cuda --vis-dir vis_out
# stdout 打印 JSON;vis_out/car_vis.png 为可视化(框+角点+识别文字)
```

CPU/ONNX(本地前后端,无需 GPU/torch):
```bash
pip install -r requirements-cpu-inference.txt
python scripts/onnx_infer_demo.py car.jpg \
    --pose-onnx models/yolo_pose/best.onnx \
    --ocr-onnx  models/ocr/ocr_best.onnx \
    --charset   charset.txt --vis-out vis.png
```

## 4. ONNX 问题速答(详见 docs/ONNX_IO.md)

- **YOLO ONNX**:输入 `images [1,3,640,640]` float(RGB,/255);输出 `output0 [1,22,8400]`,22 = 4框 + 6颜色 + 4角点×3,需转置+阈值+NMS+缩放。
- **OCR ONNX 输入 shape**:`[batch,3,48,160]`,单图即 **1×3×48×160**(确认)。
- **OCR CTC 解码**:输出 `[1,40,76]`,每步 argmax → 合并连续重复 → 去 blank(索引0) → 查 `charset.txt`。
- **CPU onnxruntime**:✅ 两模型均实测 `CPUExecutionProvider` 端到端可跑(见 onnx_infer_demo.py)。

## 5. 测试集图片(单独打包,需 scp)

`test_image_list.csv` 列出全部 2977 条。完整测试图片(场景图+裁剪图,约 1.1 GB)单独打成
`车牌测试集_2977.tar.gz`,含 `images/`(场景)、`crops/`(车牌裁剪)与 `test_local.jsonl`(相对路径,
可直接在本地跑 infer/evaluate)。scp 方式见交付说明。
