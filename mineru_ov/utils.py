"""公共工具函数：图像处理、文件操作等。

此模块集中管理项目中重复使用的工具函数，确保代码一致性。
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
import openvino as ov
from PIL import Image
from loguru import logger


ImageInput = Union[Image.Image, str, Path, bytes]


def crop_image_from_bbox(
    page_img: Image.Image,
    bbox: list[float] | tuple[float, float, float, float],
    margin: int = 0,
) -> Image.Image | None:
    """根据归一化坐标裁剪图像区域。

    Parameters
    ----------
    page_img  : PIL Image 对象
    bbox      : 归一化坐标 [x1, y1, x2, y2]，值为 0.0-1.0
    margin    : 裁剪边距（像素）

    Returns
    -------
    PIL.Image | None: 裁剪后的图像，失败时返回 None
    """
    if len(bbox) != 4:
        logger.warning("无效的 bbox 格式: {}", bbox)
        return None

    try:
        w, h = page_img.size
        x1, y1, x2, y2 = bbox
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        if x2 <= x1 or y2 <= y1:
            logger.warning("裁剪区域无效: x1={}, x2={}, y1={}, y2={}", x1, x2, y1, y2)
            return None

        return page_img.crop((x1, y1, x2, y2))
    except Exception as e:
        logger.warning("图像裁剪失败: {}", e)
        return None


def save_image(
    image: Image.Image,
    output_path: Path | str,
    format: str = "PNG",
    **kwargs,
) -> Path | None:
    """保存图像到文件。

    Parameters
    ----------
    image       : PIL Image 对象
    output_path : 输出文件路径
    format      : 图像格式（PNG, JPEG 等）
    **kwargs    : PIL save() 额外参数

    Returns
    -------
    Path | None: 成功返回保存路径，失败返回 None
    """
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(path), format=format, **kwargs)
        return path
    except Exception as e:
        logger.warning("保存图像失败 {}: {}", output_path, e)
        return None


def load_image(src: ImageInput) -> Image.Image:
    """从多种输入格式加载 PIL Image。

    Parameters
    ----------
    src : Image.Image, bytes, str, Path

    Returns
    -------
    Image.Image: PIL Image对象（不强制转换RGB，由调用方决定是否转换）
    """
    if isinstance(src, Image.Image):
        return src
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src))
    return Image.open(str(src))


_BOX_START_TOKEN = "<|box_start|>"
_BOX_END_TOKEN = "<|box_end|>"


def is_layout_output_truncated(layout_text: str) -> tuple[int, int]:
    """判断版面模型输出是否被截断。

    版面模型按阅读顺序逐个生成区块框。生成预算（layout_max_new_tokens）耗尽时，
    末尾的框会在坐标尚未写完时被硬切断，例如停在 ``<|box_start|>543 673 912``。
    这种残缺框会被 ``parse_layout_output`` 静默丢弃——它覆盖的整块版面内容随之
    消失，却不抛异常、不影响退出码，最终 Markdown 看起来一切正常。

    Returns
    -------
    tuple[int, int]: (box_start 数量, box_end 数量)；两者不等即为截断。
    """
    return layout_text.count(_BOX_START_TOKEN), layout_text.count(_BOX_END_TOKEN)


def warn_if_truncated(
    *,
    task: str,
    budget: int | None = None,
    budget_name: str = "max_new_tokens",
    page_hint: str = "",
    length_stopped: bool = False,
    unclosed_boxes: int = 0,
    hint: str = "",
) -> bool:
    """在生成输出被截断时输出 WARNING。

    截断有两条相互独立的证据，任一成立即判定为截断：

    - ``length_stopped``：引擎报告的 ``finish_reason == LENGTH``（权威信号）
    - ``unclosed_boxes``：输出里 ``<|box_start|>`` 多于 ``<|box_end|>``（结构性信号）

    两条同时成立时合并为一条告警，避免重复刷屏。

    Parameters
    ----------
    task           : 任务名，用于日志，如 "版面检测" / "内容提取"
    budget         : 当前生效的 token 预算，仅用于日志
    budget_name    : 预算对应的配置项名，如 "layout_max_new_tokens"
    page_hint      : 出错来源，如 "第 3 页"
    length_stopped : 引擎是否报告因长度上限而停止
    unclosed_boxes : 未闭合的区块框数量
    hint           : 附加的排查建议

    Returns
    -------
    bool: 是否发出了告警
    """
    if not length_stopped and unclosed_boxes <= 0:
        return False

    evidence = []
    if unclosed_boxes > 0:
        evidence.append(f"有 {unclosed_boxes} 个区块框未闭合、已被丢弃")
    if length_stopped:
        evidence.append("生成因达到 token 上限而中断")

    where = f"（{page_hint}）" if page_hint else ""
    budget_txt = (
        f"当前 {budget_name}={budget}"
        if budget is not None
        else f"请检查 {budget_name} 配置"
    )
    tail = f"；{hint}" if hint else ""

    logger.warning(
        "{}输出被截断{}：{}，这部分内容不会出现在结果中。{}，建议调大后重试{}。",
        task,
        where,
        "；".join(evidence),
        budget_txt,
        tail,
    )
    return True


def warn_if_blocks_truncated(
    flags,
    *,
    budget: int | None = None,
    budget_name: str = "max_new_tokens",
    label: str = "内容提取",
) -> bool:
    """汇总一批区块生成的截断情况，只发一条 WARNING（避免逐块刷屏）。

    Parameters
    ----------
    flags       : 每个区块对应的截断标记（布尔序列）
    budget      : 当前生效的 token 预算，仅用于日志
    budget_name : 预算对应的配置项名，如 "max_new_tokens"
    label       : 任务名，用于日志

    Returns
    -------
    bool: 是否发出了告警
    """
    bad = [i for i, flag in enumerate(flags) if flag]
    if not bad:
        return False

    shown = ", ".join(str(i) for i in bad[:8])
    more = f" 等共 {len(bad)} 处" if len(bad) > 8 else ""
    budget_txt = (
        f"当前 {budget_name}={budget}"
        if budget is not None
        else f"请检查 {budget_name} 配置"
    )

    logger.warning(
        "{}有 {} 处输出因达到 token 上限被截断（批内序号 {}{}），"
        "这些区块的内容不完整。{}，建议调大后重试。",
        label,
        len(bad),
        shown,
        more,
        budget_txt,
    )
    return True


def log_budget_escalation(
    label: str,
    n_items: int,
    old_budget: int,
    new_budget: int,
) -> None:
    """记录一次"因截断而抬高预算、只重跑被截断条目"的事件。

    这是用户可见的进度信息（走 loguru）。注意 vlm_backend 模块用的是标准库
    logging，其 INFO 级别在本 skill 中不会输出，所以这里不能借用它的 logger。
    """
    logger.info(
        "{}：{} 条输出因 token 上限被截断，预算 {} -> {}，仅重跑这 {} 条",
        label,
        n_items,
        old_budget,
        new_budget,
        n_items,
    )


def parse_layout_output_checked(
    helper,
    layout_text: str,
    *,
    page_hint: str = "",
    layout_max_new_tokens: int | None = None,
    length_stopped: bool = False,
) -> list:
    """调用 ``helper.parse_layout_output``，并在版面输出被截断时发出 WARNING。

    Parameters
    ----------
    helper                : MinerUClientHelper 实例（提供 parse_layout_output）
    layout_text           : 版面模型原始输出
    page_hint             : 出错时用于定位的来源，如 "第 3 页"
    layout_max_new_tokens : 当前生效的版面生成预算，仅用于日志
    length_stopped        : 引擎是否报告因长度上限而停止

    Returns
    -------
    list: ContentBlock 列表（与 helper.parse_layout_output 返回值一致）
    """
    n_start, n_end = is_layout_output_truncated(layout_text)
    warn_if_truncated(
        task="版面检测",
        budget=layout_max_new_tokens,
        budget_name="layout_max_new_tokens",
        page_hint=page_hint,
        length_stopped=length_stopped,
        unclosed_boxes=max(0, n_start - n_end),
        hint="版面检测实测约 22 token/框",
    )
    return helper.parse_layout_output(layout_text)


MIN_VLM_IMAGE_SIZE = 28  # Qwen2VL visual encoder 要求的最小尺寸


def pil_to_ov_tensor(image: Image.Image) -> ov.Tensor:
    """将 PIL Image 转换为 VLMPipeline 期望的 ov.Tensor (NHWC, uint8)。

    优化策略（v20）：
    - 先 convert("RGB") 确保内存连续，再用 np.asarray 零拷贝
    - 避免对非 RGB 图像使用 np.array 拷贝
    - 自动 padding 过小的图片以满足 Qwen2VL 视觉编码器最小尺寸要求

    Parameters
    ----------
    image : PIL Image 对象

    Returns
    -------
    ov.Tensor: 形状为 (1, H, W, 3) 的 NHWC 批次张量，dtype=uint8
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if h < MIN_VLM_IMAGE_SIZE or w < MIN_VLM_IMAGE_SIZE:
        new_w = max(w, MIN_VLM_IMAGE_SIZE)
        new_h = max(h, MIN_VLM_IMAGE_SIZE)
        padded = Image.new("RGB", (new_w, new_h), (255, 255, 255))
        padded.paste(image, (0, 0))
        image = padded
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return ov.Tensor(arr)


def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 88) -> str:
    """将 PIL Image 转换为 base64 编码字符串。

    Parameters
    ----------
    image   : PIL Image 对象
    format  : 图像格式（JPEG, PNG 等）
    quality : JPEG 质量（仅对 JPEG 有效）

    Returns
    -------
    str: base64 编码的图像数据（不含前缀）
    """
    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    import base64
    return base64.b64encode(buf.getvalue()).decode()


def image_to_thumbnail(
    image: Image.Image,
    max_size: tuple[int, int] = (1200, 1200),
) -> Image.Image:
    """生成图像缩略图。

    Parameters
    ----------
    image   : PIL Image 对象
    max_size: 最大尺寸 (width, height)

    Returns
    -------
    PIL.Image: 缩略图
    """
    img = image.copy()
    img.thumbnail(max_size, Image.LANCZOS)
    return img


def ensure_directory(path: Path | str, name: str = "目录") -> Path:
    """确保目录存在，不存在则创建。

    Parameters
    ----------
    path : 目录路径
    name : 目录名称（用于日志）

    Returns
    -------
    Path: 已创建的目录路径
    """
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        logger.debug("创建 {}: {}", name, p)
    return p
