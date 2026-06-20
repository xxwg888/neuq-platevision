# ONNX 输入/输出格式与 CPU 推理说明

本文件回答关于两个 ONNX 模型的全部问题。两者均已在 `onnxruntime` **CPU** 上实测可跑
(见 `scripts/onnx_infer_demo.py`,强制 `CPUExecutionProvider`,端到端输出正确车牌号)。

---

## 1. YOLOv8n-pose ONNX(`outputs/yolo_pose/best.onnx`)

| | 名称 | 形状 | 类型 |
|---|---|---|---|
| 输入 | `images` | `[1, 3, 640, 640]` | float32 |
| 输出 | `output0` | `[1, 22, 8400]` | float32 |

**输入预处理**:BGR→RGB,resize 到 640×640,像素值 `/255.0`(范围 0~1),HWC→CHW,加 batch 维。

**输出解码**(通道在前,需转置为 `[8400, 22]`):

- `22 = 4 + 6 + 12`
  - `[:4]` → 检测框 `cx, cy, w, h`(在 640 空间,中心点格式)
  - `[4:10]` → 6 类颜色分数 `blue / green / yellow_single / yellow_double / white / black`
  - `[10:22]` → 4 个角点 `(x, y, conf) × 4`(左上/右上/右下/左下,640 空间)
- `8400` → anchor 预测数量

**后处理**:每行取颜色分数最大值作为置信度与类别 → 置信度阈值过滤(默认 0.25)→ `cv2.dnn.NMSBoxes`(IoU 0.45)→ 把框与角点按 `原图尺寸/640` 缩放回原图。角点用于 `cv2.getPerspectiveTransform` 透视校正。

> 注意:`scripts/infer.py` 的 ONNX 分支当前仅解码 框+颜色(不含角点),会退化为正框裁剪;
> **完整角点解码(透视校正)请用 `scripts/onnx_infer_demo.py`**,或用 `.pt` 模型走 ultralytics(自动带角点)。

---

## 2. CRNN-CTC OCR ONNX(`outputs/models/ocr_best.onnx`)

| | 名称 | 形状 | 类型 |
|---|---|---|---|
| 输入 | `input` | `[batch, 3, 48, 160]` | float32 |
| 输出 | `logits` | `[batch, 40, 76]` | float32 |

**输入 shape 是不是 1×3×48×160?** —— 是。batch 维是动态的,单图即 `[1, 3, 48, 160]`,
H=48、W=160、3 通道 RGB。

**输入预处理**:BGR→RGB,resize 到 **160×48**(W×H),像素 `(x/255 − 0.5) / 0.5`(归一化到 -1~1),HWC→CHW,加 batch 维。

**输出 CTC 解码**(`logits` 形状 `[1, 40, 76]`,40 个时间步,76 类):

```python
seq = logits[0].argmax(-1)        # 每个时间步取最大类别 -> 长度40的索引序列
out, prev = [], -1
for idx in seq:
    if idx != prev and idx != 0:  # 1) 合并连续重复  2) 丢弃 blank(索引0)
        out.append(charset[idx])
    prev = idx
plate = "".join(out)
```

- `charset` 即 `charset.txt`,**第 0 行是 `<blank>`**,行号 = 类别索引。
- 76 类 = 1 blank + 33 省份 + 26 字母 + 10 数字 + 6 特殊(警学挂领使临)。

---

## 3. 能不能在 CPU onnxruntime 上跑?

**能。** 两个模型都用 `providers=["CPUExecutionProvider"]` 实测加载并端到端推理成功:

```bash
pip install -r requirements-cpu-inference.txt    # 仅 onnxruntime/opencv/numpy/pillow,无需 GPU/torch
python scripts/onnx_infer_demo.py  test_image.jpg \
    --pose-onnx outputs/yolo_pose/best.onnx \
    --ocr-onnx  outputs/models/ocr_best.onnx \
    --charset   deliverable/charset.txt \
    --vis-out   out_vis.png
```

实测输出(CPU):
```json
{ "backend": "onnxruntime-CPU",
  "plates": [ { "plate_type": "blue", "plate_number": "皖A990H5",
                "box": [155.3, 402.3, 210.7, 84.6], "corners": [[158.5,402.0], ...] } ] }
```

> CPU 单图延迟约几十~一百毫秒级(取决于 CPU),适合本地前后端逐图调用与老师现场抽查。
> 若本地有 GPU 且装了 onnxruntime-gpu,把 providers 换成 `CUDAExecutionProvider` 即可加速。
