"""项目常量定义：支持的设备、精度、默认参数等。"""
from __future__ import annotations

from typing import Tuple


SUPPORTED_DEVICES: set[str] = {"CPU", "GPU", "NPU"}
SUPPORTED_PRECISIONS: set[str] = {"fp16", "int4"}
NPU_REQUIRED_PRECISION: str = "int4"

DEFAULT_LAYOUT_IMAGE_SIZE: Tuple[int, int] = (1036, 1036)
DEFAULT_MIN_IMAGE_EDGE: int = 8
DEFAULT_MAX_IMAGE_EDGE_RATIO: float = 4.0

# VLM 模型目录名称（位于 model_dir 下）
VLM_MODEL_BASENAME = "MinerU2.5-Pro-2605-1.2B"
VLM_MODEL_INT4_SUFFIX = "-int4"
# 内容提取（正文/表格/公式/图片）单块生成的 token 上限。
# 表格与跨页表格合并可能很长，2048 偏紧；同时该值原本会被
# build_generation_config 的缓存键冲突覆盖，实际生效的是版面检测的预算，
# 故与 DEFAULT_LAYOUT_MAX_NEW_TOKENS 取齐，两种预算不再互相打架。
DEFAULT_MAX_NEW_TOKENS: int = 4096

# 版面检测的输出是一串区块框（<|box_start|>…<|box_end|>），实测约 22 token/框。
# 1024 只够写出约 46 个框；高密度版面（双栏栅格、含数十个编号条目的清单页）
# 会在生成到一半时撞上上限被硬截断，末尾整块版面静默丢失且不报错。
# 4096 可覆盖约 185 个框，足以应付常规单页。
DEFAULT_LAYOUT_MAX_NEW_TOKENS: int = 4096

# 截断自动重试（方案 D）时的预算上限。检测到某条输出因 finish_reason=LENGTH
# 被截断，就以 2 倍递增预算只重跑那一条；但不超过此值，避免在病态页面上
# 无界放大耗时与显存占用。
DEFAULT_MAX_NEW_TOKENS_CEILING: int = 8192

NPU_MAX_PROMPT_LEN: int = 4096
NPU_TIMEOUT_SECONDS: int = 600
DEFAULT_PDF_DPI: int = 150
DEFAULT_WEBUI_PORT: int = 7878

MINERU_OUTPUT_TOKENS: Tuple[str, ...] = (
    "<|box_start|>",
    "<|box_end|>",
    "<|ref_start|>",
    "<|ref_end|>",
    "<|rotate_up|>",
    "<|rotate_right|>",
    "<|rotate_down|>",
    "<|rotate_left|>",
    "<nl>",
    "<fcel>",
    "<ecel>",
    "<lcel>",
    "<ucel>",
    "<xcel>",
    "<ched>",
    "<|md_start|>",
    "<|md_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|paratext|>",
    "<|txt_contd|>",
)

IMAGE_EXTS: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXT: str = ".pdf"

SENTENCE_END_CHARS: set[str] = {"。", ".", "！", "!", "？", "?", "…", "」", "』", "）", ")"}
