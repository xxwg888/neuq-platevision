# 端到端指标复现

## 一条命令复现报告中的端到端指标

```bash
conda activate wry        # 或 pip install -r requirements.txt
cd 车牌课程设计

python scripts/evaluate.py --mode pipeline \
    --yolo-model    outputs/yolo_pose/best.pt \
    --ocr-checkpoint outputs/models/ocr_best.pt \
    --test-manifest data/processed/ocr_5color/manifests/test.jsonl \
    --output-json   outputs/metrics/eval_pipeline_5color_pose.json \
    --device cuda
```

## 预期输出(服务器实测,2026-06-20)

```
detection: {'TP': 2922, 'FP': 43, 'FN': 12,
            'precision': 0.9855, 'recall': 0.9959, 'F1': 0.9907, 'iou_threshold': 0.5}
ocr:       {'plate_accuracy': 0.920602, 'character_accuracy': 0.985160,
            'avg_edit_distance': 0.104381}
```

- **端到端整牌准确率 = 0.9206**
- **端到端字符准确率 = 0.9852**
- 运行时间 ~72 秒(单卡 RTX 4090),N = 2977 真实测试样本。

> 该命令对每张测试场景图跑 YOLOv8n-pose 检测 → 4 角点透视校正 → CRNN-CTC 识别,
> 按 IoU≥0.5 匹配 GT 框统计检测 P/R/F1,并对匹配上的牌统计整牌/字符准确率。

## 其它指标的复现

```bash
# OCR 综合(总体/分颜色/分省份/泄漏检查)-> outputs/metrics/ocr_final.json
python scripts/eval_ocr_full.py --ocr-checkpoint outputs/models/ocr_best.pt

# 颜色分类混淆矩阵 -> outputs/metrics/color_eval.json (整体 0.9952)
python scripts/color_eval.py --yolo-model outputs/yolo_pose/best.pt \
    --test-manifest data/processed/ocr_5color/manifests/test.jsonl \
    --output-json outputs/metrics/color_eval.json

# 传统 baseline 对比 -> template_baseline.json / knn_baseline.json
python scripts/template_baseline.py --train-manifest .../train.jsonl --test-manifest .../test.jsonl \
    --output-json outputs/metrics/template_baseline.json
python scripts/knn_baseline.py     --train-manifest .../train.jsonl --test-manifest .../test.jsonl \
    --output-json outputs/metrics/knn_baseline.json

# 汇总成 Markdown 表 -> outputs/metrics/summary_table.md
python scripts/build_metrics_table.py
```

## CPU / 本地无 GPU 复现(用 ONNX)

端到端单图(强制 CPU):
```bash
pip install -r deliverable/requirements-cpu-inference.txt
python scripts/onnx_infer_demo.py test.jpg \
    --pose-onnx outputs/yolo_pose/best.onnx \
    --ocr-onnx  outputs/models/ocr_best.onnx \
    --charset   deliverable/charset.txt --vis-out out.png
```
