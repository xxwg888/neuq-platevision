# 传统 baseline(模板匹配 NCC + HOG+KNN)— 本地单图推理说明

与深度 CRNN-CTC 对照的两套传统方法,供前后端统一展示三套路线。**纯 CPU**,
单图接口为 `scripts/traditional_infer_demo.py`(非批量评测脚本)。

---

## 0. 三套方法的核心区别

| | 流程 | 是否手工切字 | 分类器 | 测试集表现(整牌/字符) |
|---|---|---|---|---|
| **CRNN-CTC**(深度) | 整块透视校正车牌 → CNN+BiGRU+CTC | **否**(端到端,CTC 对齐) | 神经网络 | **0.923 / 0.985** |
| **HOG+KNN**(传统) | 二值化 → 投影分割 → HOG → KNN | 是 | KNN(k=5) | 0.516 / 0.803 |
| **模板 NCC**(传统) | 二值化 → 投影分割 → 模板相似度 | 是 | 最近模板(余弦) | 0.265 / 0.628 |

关键差异:CRNN **不手工切字**,把整块校正车牌喂给网络,CTC 解决字符对齐;
两套传统方法都依赖**显式字符分割**,分割误差会逐字传播,且对形近字(0/Q、5/S)、
汉字多笔画区分弱——这正是引入深度方法的动机,也是失败案例的主要来源。

---

## 1. 模板匹配 NCC 完整流程

1. **二值化**(`binarize`):灰度 → 裁剪边框(上下 10%、左右 2%)→ Otsu → 反相为白字 → 形态学开运算去铆钉/噪点。
2. **字符分割**(`segment_runs`,投影法强制切到 N 字):平滑列投影 → 阈值取字符段 → 段过多合并最近邻、过少分裂最宽段。
3. **模板匹配**:每个字符块 resize 到 40×24 → 展平、零均值、单位化为 NCC 向量 → 与各类模板做点积(=归一化互相关)→ 取最大相似度的类别。

模板文件 `models/template/templates.npz`:
- 字段 **`classes`**(67,) 与 **`templates`**(67×960 float32),**行一一对应**。
- `templates[i]` 是第 `classes[i]` 类的均值字形 NCC 向量(40×24=960 维)。
- `classes` 按 `CHAR_TO_IDX`(传统 71 字符集:31省 + 24字母去I/O + 10数字 + 6特殊,**无 blank**)顺序的子集;训练中无干净样本的 港澳挂领使临 未包含,故 67 类。
- ⚠️ 这是**传统字符集**,与 CRNN 的 `charset.txt`(76,含 blank)**不同**;NCC/KNN 不要用 charset.txt。predict 时直接用 `classes[argmax]` 取字符,行对齐即可,无需外部映射。

---

## 2. HOG+KNN 完整流程

1. **二值化**:同上(共用 `binarize`)。
2. **字符分割**:同上(共用 `segment_runs`)。
3. **HOG 特征**:每个字符块 resize 到 24×40 → `cv2.HOGDescriptor`(win 24×40,block 16×16,stride 8×8,cell 8×8,9 bins)→ 288 维。
4. **KNN 分类**:`cv2.ml.KNearest`,k=5,`findNearest` 多数投票。

KNN 文件 `models/knn/knn_baseline.xml`(294MB,单独下载 `knn_baseline.xml.gz` 解压):存了全部训练字符的 HOG 特征(KNN 为惰性分类器)。
置信度 = k 个近邻中投票给胜出类的比例;另给 `knn_neighbor_dist`(平均距离)。

---

## 3. 单图推理命令(前后端接口)

```bash
python scripts/traditional_infer_demo.py \
    --image xxx.jpg \
    --mode both \
    --yolo-onnx models/yolo_pose/best.onnx \
    --templates models/template/templates.npz \
    --knn-model models/knn/knn_baseline.xml \
    --vis-dir   outputs/demo_vis
```

- `--mode {ncc,knn,both}`:选择跑哪套/两套都跑。
- `--input-type {auto,car,plate}`:`auto`(默认)= 给了 `--yolo-onnx` 就先检测,检测框占图 <85% 视为整车图并透视校正,否则当作已校正车牌;`car` 强制检测;`plate` 跳过检测。
- `--n-chars N`:指定字符数(默认:绿牌 8,其余 7;crop 模式未知类型则自由切分)。可显式传入更稳。
- `--k`(默认 5)、`--conf`(检测阈值 0.25)、`--plate-width/height`(默认 160×48)。

### 输出 JSON(stdout)

```jsonc
{
  "input_path": "...",
  "is_car_image": true,            // 是否判定为整车图
  "used_detector": true,           // 是否跑了 YOLO-pose
  "plate_type": "blue",            // 检测得到的颜色(crop 模式为 null)
  "bbox": [x, y, w, h],            // 原图车牌框(仅检测时)
  "corners": [[x1,y1],...],        // 4 角点(仅检测时)
  "n_chars_used": 7,
  "ncc_text": "皖A99QHS",
  "knn_text": "皖A990H5",
  "per_char": [
    {"index":0,"bbox":[x,y,w,h],   // 字符在校正车牌坐标系的框
     "ncc":"皖","ncc_similarity":0.71,
     "knn":"皖","knn_confidence":0.6,"knn_neighbor_dist":1.42}, ...
  ],
  "vis_paths": {
    "detected": "..._detected.png",          // 整车图 + 框 + 角点(仅检测时)
    "plate_crop": "..._plate_crop.png",       // 校正后车牌
    "binary": "..._binary.png",               // 二值化图
    "segmented": "..._segmented.png",         // 字符分割框
    "classification_vis": "..._classification.png"  // 逐字符 NCC/KNN 对比
  }
}
```

整车图与车牌 crop 两种输入都支持:整车图自动检测+校正;crop 用 `--input-type plate` 跳过检测。

---

## 4. 批量评测命令(复现报告指标)

```bash
# 模板 NCC(整牌 0.265 / 字符 0.628)
python scripts/template_baseline.py \
    --train-manifest data/processed/ocr_5color/manifests/train.jsonl \
    --test-manifest  data/processed/ocr_5color/manifests/test.jsonl \
    --save-templates models/template/templates.npz \
    --output-json    outputs/metrics/template_baseline.json

# HOG+KNN(整牌 0.516 / 字符 0.803)
python scripts/knn_baseline.py \
    --train-manifest data/processed/ocr_5color/manifests/train.jsonl \
    --test-manifest  data/processed/ocr_5color/manifests/test.jsonl \
    --save-model     models/knn/knn_baseline.xml --k 5 \
    --output-json    outputs/metrics/knn_baseline.json
```

---

## 5. 本地 CPU 依赖

```
opencv-python   # cv2:HOG、KNN、二值化、imdecode/imencode
numpy
matplotlib      # 逐字符分类可视化(中文)
pillow          # CJK 文字标注
# 整车图模式额外需要:onnxruntime(跑 YOLO-pose 透视校正)
```
**不需要 torch / ultralytics。** 中文可视化字体已打包 `fonts/NotoSansCJK-Regular.ttc`,demo 自动加载。

## 6. Windows 中文路径

demo 的图像读写使用 `cv2.imdecode(np.fromfile(...))` / `cv2.imencode(...).tofile(...)`,
可直接读写中文路径,无需 ASCII cache 规避。
