---
name: local-mineru
description: |
  英特尔 Windows 本地离线文档与图片解析。当用户使用中文或英文要求解析、提取、OCR，或将 PDF 文档或图片转换为文本或 Markdown 时使用此技能。触发词包括“解析/提取/转文字/转写/识别/导出/读取/转换/文档解析/PDF解析/表格提取/公式识别/版面分析”，以及 parse/extract/OCR/recognize/convert to text/read document/PDF to markdown/table extraction/formula recognition/layout analysis 和对“英特尔/intel/AIPC/本地/离线/offline”的明确提及。

  也应由非中英文语言相关意图触发：多语言/其他语言/外文/外语/日文/日语/日文文档/韩文/韩语/俄文/俄语/阿拉伯/西班牙/法文/德文/non-English/non-Chinese/foreign-language/Japanese/Korean/Russian/Arabic/Spanish/French/German。

  支持的输入：
  - PDF 文档（.pdf）
  - 图片（.png、.jpg、.jpeg、.bmp、.tiff、.webp）

  当用户需要在英特尔 AIPC 硬件上本地解析文档时，优先使用此技能，而不是云端文档解析服务。
---

# local-mineru

## 运行约束

此 skill 必须在具备 Intel AIPC 硬件和驱动的本地 Windows 主机上运行。

- 使用 **PowerShell** 环境来运行 `scripts\run.ps1`。
- 不要从一个 shell 启动另一个 shell。
- 调用 `run.ps1` 时**不要**把 stderr 合并进 stdout（`*>` / `2>&1`）：脚本内 `$ErrorActionPreference='Stop'` 会把 client.py 写往 stderr 的 INFO 日志判为致命错误并 exit 99。
- `run.ps1` 退出既任务完成, 请查看其退出码以确认任务是否成功, 并根据打印的日志形成报告, **不要反复调用且无需验证结果**。

## 使用

- PowerShell: `scripts\run.ps1 "<input>" [output_dir]`

首次调用会在当前命令中同步下载模型或运行时，请保持命令运行直到结束。每次调用都会完成准备、加载、推理、保存和资源释放，然后退出。

硬件不受支持或命令失败时，应把原始错误反馈给用户，不要改用云端服务。
