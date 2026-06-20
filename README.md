# NEUQ PlateVision Lab

车牌定位、分割与识别课程设计项目。项目面向蓝牌、绿牌、黄牌、白牌等真实场景车牌，完成从车牌检测、角点定位、透视校正、字符识别到前后端可视化演示的完整流程。

## 技术路线

- 检测与定位：自训练 YOLOv8n-pose，同时输出车牌框、四角点和颜色类别。
- 识别模型：自训练 CRNN-CTC，字符集包含 `<blank>`、省份汉字、字母和数字。
- 本地推理：YOLO ONNX + OCR ONNX，支持 CPU onnxruntime 推理。
- 可视化系统：React + TypeScript + Vite 前端，FastAPI 后端。
- 实验评估：提供端到端准确率、字符准确率、检测 Precision/Recall/F1、颜色分类结果和成功/失败案例。

## 项目结构

```text
.
├── backend/                 # FastAPI 推理服务
├── frontend/                # React/Vite 可视化界面
├── data/                    # 本地上传、输出和评估目录
├── plan/                    # 设计计划、接口约定和报告素材
├── server_artifacts/        # 训练代码、指标、文档和模型交付物
└── README.md
```

## 功能

- 单图识别：上传车辆图片，输出车牌号、检测框、置信度和中间结果。
- 批量实验：对测试集或样例集进行批量评估，生成统计指标。
- 中间过程展示：定位结果、车牌裁剪、二值化、字符分割。
- 多 provider 推理：自训练 ONNX、OpenCV baseline、远程服务器接口。
- 课程报告支撑：保留训练记录、指标表、复现实验命令和 PPT 素材。

## 快速启动

后端：

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

前端：

```powershell
cd frontend
npm install
npm run dev
```

浏览器打开：

```text
http://127.0.0.1:5173
```

## 模型与数据

训练交付物位于 `server_artifacts/deliverable/`。其中包含训练脚本、模型结构、指标文件、复现说明和 ONNX I/O 文档。

出于仓库体积和数据合规考虑，完整测试集、展示样例、模型权重和中间输出默认不提交到 Git。需要本地演示时，将以下文件放回对应目录：

- `server_artifacts/deliverable/models/yolo_pose/best.onnx`
- `server_artifacts/deliverable/models/ocr/ocr_best.onnx`
- `server_artifacts/deliverable/charset.txt`
- `server_artifacts/testset_2977/`

## 实验结果

端到端测试集规模为 2977 张。自训练 YOLOv8n-pose + CRNN-CTC 管线达到：

- 端到端整牌准确率：0.9206
- 字符准确率：0.9852
- 检测 Precision：0.9855
- 检测 Recall：0.9959
- 检测 F1：0.9907

详细指标见 `server_artifacts/deliverable/metrics/` 和 `server_artifacts/deliverable/docs/REPRODUCE.md`。

## 验证命令

后端测试：

```powershell
backend\.venv\Scripts\python -m unittest discover -s backend\tests -v
```

前端构建：

```powershell
cd frontend
npm run build
```

## 说明

本项目用于东北大学秦皇岛分校数字图像处理课程设计。仓库保留可复现代码、系统实现和实验结果，数据集文件和模型权重按课程演示环境单独管理。
