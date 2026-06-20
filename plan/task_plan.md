# Task Plan: 车牌识别研究演示平台

## Goal
在桌面创建一个可运行的前后端分离车牌识别平台，本地可用 OpenCV baseline 演示，后续可接服务器模型/API。

## Phases
- [x] Phase 1: 创建项目目录和计划文件
- [x] Phase 2: 实现 FastAPI 后端接口与 provider 架构
- [x] Phase 3: 实现 React 前端研究工作台
- [x] Phase 4: 补充批量评测、服务器接入和报告素材
- [ ] Phase 5: 运行验证并记录结果

## Decisions Made
- 项目目录: `C:\Users\23898\Desktop\车牌识别课程设计`
- 技术路线: React + TypeScript + Vite, FastAPI + OpenCV
- 推理模式: `opencv_baseline`, `local_model`, `remote_server`
- 本地职责: UI、接口、baseline、小模型推理；训练放服务器

## Errors Encountered
- `New-Item -LiteralPath` 在当前 PowerShell 中不可用，已改用 `New-Item -Path`。
- PowerShell 阻止 `npm.ps1`，已改用 `npm.cmd`。
- pip 在沙箱内网络下载超时，已授权后完成安装。
- 初次测试发现根目录运行时缺少 `backend/app` 路径，已在测试中补充路径。
- TypeScript 最新版本对旧 `moduleResolution` 报错，已改用 `Bundler`。
- 前端缺少 React/Node 类型包，已安装 `@types/react`、`@types/react-dom`、`@types/node`。

## Status
**Currently in Phase 5** - 后端单元测试与前端生产构建已通过，正在进行真实服务和页面验证。

