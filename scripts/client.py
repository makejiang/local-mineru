# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Single-process MinerU document parser."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _normalize_log_path(value: str | None) -> Path:
    if not value:
        value = f"~/.openvino/log/mineru-client-py-{time.strftime('%Y%m%d-%H%M%S')}.log"
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _print_parse_reply(reply: dict) -> int:
    if not reply.get("ok") or not reply.get("success"):
        print(f"❌ 文档解析失败:\n{reply.get('error', reply.get('error_code', '<unknown error>'))}", file=sys.stderr)
        return 1
    markdown = reply.get("markdown", "")
    if markdown:
        print(markdown)
    print(f"\n✅ 解析完成！共 {reply.get('pages', 0)} 页")
    print(f"解析耗时: {float(reply.get('parse_time_s', 0.0)):.1f}s")
    print(f"Markdown: {reply.get('md_file', '')}")
    return 0


def _input_files(input_path: str) -> tuple[list[Path], str]:
    resolved = Path(input_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"路径不存在: {resolved}")
    if resolved.is_file():
        return [resolved], str(resolved.parent / "output")
    files = sorted(path for path in resolved.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise ValueError(f"目录中未找到可解析的文件: {resolved}")
    return files, str(resolved / "output")


def run_request(input_path: str, output_dir: str = "", log_path: Path | None = None) -> int:
    runtime = None
    try:
        files, default_output = _input_files(input_path)
        target_output = output_dir or default_output
        Path(target_output).mkdir(parents=True, exist_ok=True)

        import runtime as runtime_module

        runtime = runtime_module.Runtime(log_path=log_path)
        runtime.initialize()
        if runtime.state != runtime_module.STATE_RUNNING:
            print(runtime.init_error or f"runtime initialization failed: {runtime.state}", file=sys.stderr)
            return 1
        result = 0
        for index, path in enumerate(files, 1):
            if len(files) > 1:
                print(f"[{index}/{len(files)}] 解析: {path.name}")
            reply = runtime.dispatch({"op": "parse", "input_path": str(path), "output_dir": target_output})
            result = _print_parse_reply(reply) or result
        return result
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"❌ 文档解析失败:\n{exc}", file=sys.stderr)
        return 1
    finally:
        if runtime is not None:
            runtime.pipeline = None
        gc.collect()


def _cmd_request(input_path: str, output_dir: str, log_path: Path | None = None) -> int:
    return run_request(input_path, output_dir, log_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="local-mineru single-process client")
    parser.add_argument("-i", "--input", required=True, help="PDF/image or directory")
    parser.add_argument("-o", "--output", default="", help="output directory")
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)
    return run_request(args.input, args.output, _normalize_log_path(args.log))


if __name__ == "__main__":
    raise SystemExit(main())
