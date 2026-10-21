# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO backend for MinerU 2.5 document parsing.

核心设计思路：
1. 使用 MinerUClientHelper（低层辅助类）而非向 MinerUClient 注入后端，
   彻底绕开 apply_chat_template 等接口兼容性问题。
2. PIL Image 需转为 ov.Tensor（NHWC uint8 batch），不能直接传 PIL。
3. MinerU 布局 token（<|box_start|> 等）被 HF 标记为 special=True，
   OpenVINO detokenizer 会将其过滤，导致版面解析失败。
   需在模型转换后用 patch_detokenizer_for_mineru() 重新导出 detokenizer。

参考实现：https://github.com/openvinotoolkit/openvino_notebooks/pull/3445
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from PIL import Image

from ._inference import InferenceEngine, BatchInferenceEngine, build_generation_config
from .constants import (
    DEFAULT_LAYOUT_IMAGE_SIZE,
    DEFAULT_LAYOUT_MAX_NEW_TOKENS,
    DEFAULT_MAX_IMAGE_EDGE_RATIO,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_NEW_TOKENS_CEILING,
    DEFAULT_MIN_IMAGE_EDGE,
)
from .utils import (
    ImageInput,
    load_image,
    log_budget_escalation,
    parse_layout_output_checked,
    warn_if_blocks_truncated,
)

logger = logging.getLogger(__name__)

_CPU_WORKERS = 4

# 截断自动重试的最大重试次数（不含首次）。实际次数通常更少：
# 预算每次翻倍、触到 ceiling 后不再增长即停止。
_TRUNCATION_MAX_RETRIES = 2


def _escalating_generate(
    engine,
    items: list,
    make_cfg: Callable,
    budget: int,
    *,
    ceiling: int,
    max_attempts: int,
    label: str,
) -> list[tuple[str, bool]]:
    """批量生成；对因 token 上限被截断的条目按 2 倍递增预算重跑。

    只重跑被截断的那几条，已成功的不会重复计算，因此正常情况下零额外开销
    （没有截断就直接返回）。预算每次翻倍但不超过 ``ceiling``；触顶后仍截断
    就原样返回，由调用方发出 WARNING。

    Parameters
    ----------
    engine       : 推理引擎，需实现 ``batch_generate_detailed``
    items        : list[(image, prompt, sampling_params)]
    make_cfg     : ``(sampling_params, budget) -> GenerationConfig``
    budget       : 起始 token 预算
    ceiling      : 预算上限
    max_attempts : 除首次外最多再重试几次（0 表示关闭重试）
    label        : 日志用任务名

    Returns
    -------
    list[tuple[str, bool]]: 与 items 同序的 (文本, 最终是否仍被截断)
    """
    total = len(items)
    texts: list[str] = [""] * total
    flags: list[bool] = [False] * total
    pending = list(range(total))
    cur_budget = budget

    for attempt in range(max_attempts + 1):
        if not pending:
            break

        batch = [
            (items[i][0], items[i][1], make_cfg(items[i][2], cur_budget))
            for i in pending
        ]
        results = engine.batch_generate_detailed(batch)
        for k, idx in enumerate(pending):
            texts[idx] = results[k][0]
            flags[idx] = results[k][1]
        pending = [idx for k, idx in enumerate(pending) if results[k][1]]

        if not pending or attempt >= max_attempts:
            break

        new_budget = min(cur_budget * 2, ceiling)
        if new_budget <= cur_budget:
            break  # 已到上限，重试也无意义
        log_budget_escalation(label, len(pending), cur_budget, new_budget)
        cur_budget = new_budget

    return list(zip(texts, flags))


class OVMinerUClient:
    """OpenVINO 推理客户端，实现 MinerU 2.5 两步文档解析流程。

    通过 MinerUClientHelper 处理版面检测与内容提取的 pipeline 逻辑，
    通过 openvino_genai.VLMPipeline 完成实际推理。

    Attributes
    ----------
    engine : InferenceEngine
        OpenVINO 推理引擎封装
    helper : MinerUClientHelper
        MinerU 辅助类，处理版面检测和内容提取

    用法
    ----
    ```python
    client = OVMinerUClient(model_dir="ov_models/", device="GPU")
    markdown, blocks = client.image_to_markdown(pil_image)
    ```

    Notes
    -----
    - NPU 设备默认使用延迟优化模式 (LATENCY)
    - NPU 模型编译结果会缓存到 .ov_npu_cache 目录
    - max_new_tokens 超过 npu_max_new_tokens 时会自动降低
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: str = "CPU",
        precision: str = "fp16",
        image_analysis: bool = False,
        layout_image_size: Tuple[int, int] = DEFAULT_LAYOUT_IMAGE_SIZE,
        min_image_edge: int = DEFAULT_MIN_IMAGE_EDGE,
        max_image_edge_ratio: float = DEFAULT_MAX_IMAGE_EDGE_RATIO,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        layout_max_new_tokens: int = DEFAULT_LAYOUT_MAX_NEW_TOKENS,
        retry_on_truncation: bool = True,
        max_new_tokens_ceiling: int = DEFAULT_MAX_NEW_TOKENS_CEILING,
        npu_max_new_tokens: int = 2048,
        ov_config: Optional[dict] = None,
        performance_hint: str = "LATENCY",
        enable_warmup: bool = True,
        debug: bool = False,
    ) -> None:
        """初始化 OVMinerUClient。

        Parameters
        ----------
        model_dir              : VLM 模型子目录路径（已按精度选择好）
        device                 : 推理设备，如 "CPU", "GPU", "NPU"
        precision              : 模型精度, "fp16" 或 "int4"
        image_analysis         : 是否启用图表/图片内容分析
        layout_image_size      : 版面检测输入图像尺寸 (宽, 高)
        min_image_edge         : 最小图像边长（像素）
        max_image_edge_ratio   : 最大图像边长与输入图像边长的比例上限
        max_new_tokens         : 内容提取单次推理最大新 token 数
        layout_max_new_tokens  : 版面检测最大新 token 数（约 22 token/框，过小会截断输出）
        npu_max_new_tokens     : NPU 设备专用 token 上限
        ov_config              : OpenVINO 额外配置字典
        performance_hint       : 性能模式 "LATENCY" / "THROUGHPUT"
        enable_warmup          : 是否预热推理
        debug                  : 调试模式开关
        """
        try:
            from mineru_vl_utils.mineru_client import (
                DEFAULT_PROMPTS,
                DEFAULT_SAMPLING_PARAMS,
                MinerUClientHelper,
            )
        except ImportError as e:
            raise ImportError(
                "mineru-vl-utils 未安装或版本过低，请运行: "
                "pip install 'mineru-vl-utils>=0.2.7'"
            ) from e

        self.model_dir = Path(model_dir)
        self.device = device
        self.precision = precision
        self._max_new_tokens = max_new_tokens
        self._layout_max_new_tokens = layout_max_new_tokens
        self._max_new_tokens_ceiling = max(
            max_new_tokens_ceiling, max_new_tokens, layout_max_new_tokens
        )
        self._retry_attempts = _TRUNCATION_MAX_RETRIES if retry_on_truncation else 0
        self._layout_sp = (
            DEFAULT_SAMPLING_PARAMS.get("[layout]") if DEFAULT_SAMPLING_PARAMS else None
        )

        self.engine = InferenceEngine(
            model_dir=str(self.model_dir),
            device=device,
            max_new_tokens=max_new_tokens,
            ov_config=ov_config,
            performance_hint=performance_hint,
            enable_warmup=enable_warmup,
        )

        self.helper = MinerUClientHelper(
            backend="openvino",
            prompts=DEFAULT_PROMPTS,
            sampling_params=DEFAULT_SAMPLING_PARAMS,
            layout_image_size=layout_image_size,
            min_image_edge=min_image_edge,
            max_image_edge_ratio=max_image_edge_ratio,
            simple_post_process=False,
            handle_equation_block=True,
            abandon_list=False,
            abandon_paratext=False,
            image_analysis=image_analysis,
            debug=debug,
        )
        logger.info("OVMinerUClient ready (device=%s)", device)

    def _layout_blocks(self, items: list, page_start: int = 1) -> list[list]:
        """版面检测并解析为区块列表。

        被截断的（finish_reason=LENGTH）会先自动加大预算重跑，仍截断或存在
        未闭合的框时，带上页码发出 WARNING。
        """
        texts_with_flags = _escalating_generate(
            self.engine,
            items,
            lambda _sp, budget: build_generation_config(self._layout_sp, budget),
            self._layout_max_new_tokens,
            ceiling=self._max_new_tokens_ceiling,
            max_attempts=self._retry_attempts,
            label="版面检测",
        )
        return [
            parse_layout_output_checked(
                self.helper,
                text,
                page_hint=f"第 {page_start + offset} 页",
                layout_max_new_tokens=self._layout_max_new_tokens,
                length_stopped=truncated,
            )
            for offset, (text, truncated) in enumerate(texts_with_flags)
        ]

    def _batch_predict(self, items: list, page_hint: str = "") -> list[str]:
        """内容提取；被截断的先自动加大预算重跑，仍截断则汇总告警。"""
        label = f"内容提取{'（' + page_hint + '）' if page_hint else ''}"
        texts_with_flags = _escalating_generate(
            self.engine,
            items,
            build_generation_config,
            self._max_new_tokens,
            ceiling=self._max_new_tokens_ceiling,
            max_attempts=self._retry_attempts,
            label=label,
        )
        warn_if_blocks_truncated(
            [flag for _, flag in texts_with_flags],
            budget=self._max_new_tokens,
            label=label,
        )
        return [text for text, _ in texts_with_flags]

    def two_step_extract(
        self,
        image: ImageInput,
        not_extract_list: Optional[Sequence[str]] = None,
        image_analysis: Optional[bool] = None,
        pause_check: Optional[Callable[[], bool]] = None,
    ):
        """对单页图像执行完整两步提取，返回 ExtractResult。
        
        如果 `pause_check` 在任意子步骤返回 True，则立即返回当前已提取的部分结果。
        """
        from mineru_vl_utils.structs import ExtractResult

        page = load_image(image)
        if page.mode != 'RGB':
            page = page.convert('RGB')

        t_layout = time.time()
        layout_image = self.helper.prepare_for_layout(page)
        layout_prompt = self.helper.prompts["[layout]"]
        layout_sp = self.helper.sampling_params.get("[layout]")
        blocks = self._layout_blocks(
            [(layout_image, layout_prompt, layout_sp)], page_start=1
        )[0]
        logger.info(
            "layout_detect: %.2fs, %d blocks",
            time.time() - t_layout, len(blocks)
        )

        if pause_check is not None and pause_check():
            logger.info("收到暂停信号，返回已检测版面的部分结果")
            return ExtractResult(blocks)

        t_extract = time.time()
        block_images, prompts, sps, indices = self.helper.prepare_for_extract(
            page,
            blocks,
            not_extract_list=list(not_extract_list) if not_extract_list else None,
            image_analysis=image_analysis,
        )

        from mineru_vl_utils.post_process.table_image_processor import (
            TABLE_IMAGE_TOKEN_MAP_KEY,
            replace_table_image_tokens,
        )

        infer_items = []
        infer_indices = []
        for block_image, prompt, sp, idx in zip(block_images, prompts, sps, indices):
            if pause_check is not None and pause_check():
                logger.info("收到暂停信号，停止后续区块提取")
                break
            infer_items.append((block_image, prompt, sp))
            infer_indices.append(idx)

        contents = (
            self._batch_predict(infer_items, page_hint="第 1 页") if infer_items else []
        )

        for idx, content in zip(infer_indices, contents):
            block = blocks[idx]
            if TABLE_IMAGE_TOKEN_MAP_KEY in block:
                token_map = block.get(TABLE_IMAGE_TOKEN_MAP_KEY) or {}
                if token_map:
                    content = replace_table_image_tokens(content, token_map)
            block["content"] = content

        logger.info(
            "content_extract: %.2fs (%d blocks)",
            time.time() - t_extract, len(block_images)
        )

        blocks = self.helper.post_process(blocks)
        return ExtractResult(blocks)

    def image_to_markdown(
        self,
        image: ImageInput,
        **kwargs,
    ) -> Tuple[str, object]:
        """单页图像 → (Markdown 字符串, ExtractResult)。

        Parameters
        ----------
        image : 输入图像
        **kwargs : 传递给 two_step_extract 的参数

        Returns
        -------
        Tuple[str, object]: (Markdown 内容, ExtractResult 对象)
        """
        from mineru_vl_utils.post_process import json2md
        blocks = self.two_step_extract(image, **kwargs)
        return json2md(list(blocks)), blocks

    def pdf_to_markdown(
        self,
        pdf: Union[str, Path, bytes],
        dpi: int = 200,
        progress_callback=None,
        **kwargs,
    ) -> Tuple[str, List]:
        """将 PDF 每页转换为 Markdown。

        Parameters
        ----------
        pdf              : PDF 文件路径或 bytes
        dpi              : PDF 渲染分辨率
        progress_callback: 进度回调函数 (current: int, total: int) -> None
        **kwargs         : 传递给 image_to_markdown 的参数

        Returns
        -------
        Tuple[str, List]: (拼接的 Markdown, 每页的 blocks 列表)
        """
        from .pdf_utils import load_images_from_path
        pages = list(load_images_from_path(pdf, dpi=dpi))
        md_pages: List[str] = []
        block_pages: List = []
        total = len(pages)

        for i, (_, _, page_img) in enumerate(pages):
            if progress_callback is not None:
                progress_callback(i, total)
            md, blocks = self.image_to_markdown(page_img, **kwargs)
            md_pages.append(md)
            block_pages.append(blocks)

        if progress_callback is not None:
            progress_callback(total, total)

        return "\n\n---\n\n".join(md_pages), block_pages

    def batch_process_pages(
        self,
        page_images: list[Image.Image],
        progress_callback=None,
        **kwargs,
    ) -> list[list]:
        """两阶段批量处理多页图像：全部版面检测 → 全部内容提取。

        Parameters
        ----------
        page_images : list[Image.Image]
            已渲染好的页面图像列表
        progress_callback : 可选进度回调 (current, total, message)
        **kwargs : 传递给 prepare_for_extract 的参数

        Returns
        -------
        list[list]: 每页的 ContentBlock 列表
        """
        n_pages = len(page_images)
        if n_pages == 0:
            return []

        layout_prompt = self.helper.prompts["[layout]"]
        layout_sp = self.helper.sampling_params.get("[layout]")

        if progress_callback:
            progress_callback(0, n_pages, "第一阶段：版面检测...")
        logger.info("Phase 1/2 — layout detection for {} pages", n_pages)

        layout_items = []
        for page_img in page_images:
            layout_img = self.helper.prepare_for_layout(page_img)
            layout_items.append((layout_img, layout_prompt, layout_sp))
        all_blocks = self._layout_blocks(layout_items, page_start=1)
        logger.info("Phase 1 done — {} layouts parsed", n_pages)

        if progress_callback:
            progress_callback(0, n_pages, "第二阶段：内容提取...")
        logger.info("Phase 2/2 — content extraction for {} pages", n_pages)

        all_block_items: list[tuple[int, int, Image.Image, str, object]] = []
        for page_idx, (page_img, blocks) in enumerate(zip(page_images, all_blocks)):
            block_images, prompts, sps, indices = self.helper.prepare_for_extract(
                page_img, blocks,
                not_extract_list=list(kwargs.get("not_extract_list") or []),
                image_analysis=kwargs.get("image_analysis"),
            )
            for idx, img, prompt, sp in zip(indices, block_images, prompts, sps):
                all_block_items.append((page_idx, idx, img, prompt, sp))

        if all_block_items:
            infer_items = [(img, prompt, sp) for (_, _, img, prompt, sp) in all_block_items]
            contents = self._batch_predict(infer_items)
            from mineru_vl_utils.post_process.table_image_processor import (
                TABLE_IMAGE_TOKEN_MAP_KEY, replace_table_image_tokens,
            )
            for (page_idx, block_idx, _, _, _), content in zip(all_block_items, contents):
                block = all_blocks[page_idx][block_idx]
                if TABLE_IMAGE_TOKEN_MAP_KEY in block:
                    token_map = block.get(TABLE_IMAGE_TOKEN_MAP_KEY) or {}
                    if token_map:
                        content = replace_table_image_tokens(content, token_map)
                block["content"] = content

        for page_idx, blocks in enumerate(all_blocks):
            all_blocks[page_idx] = self.helper.post_process(blocks)
            if progress_callback:
                progress_callback(page_idx + 1, n_pages, f"Page {page_idx + 1}/{n_pages}")

        if progress_callback:
            progress_callback(n_pages, n_pages, "完成")
        return all_blocks

    def unload(self) -> None:
        """释放 VLMPipeline，释放显存/内存。"""
        if self.engine:
            self.engine.unload()
            self.engine = None
        logger.info("OVMinerUClient unloaded")


class BatchMinerUClient:
    """基于 ContinuousBatchingPipeline 的高吞吐推理客户端。

    与 OVMinerUClient 接口完全兼容，可在 pipeline 中互换使用：
      - 使用 BatchInferenceEngine（ContinuousBatchingPipeline）
      - Block 提取通过 batch_generate() 批量提交
      - pdf_to_markdown 使用两阶段批处理策略

    v20 optimizations:
      - GenerationConfig cached (layout / default block)
      - CPU preprocessing parallelized via ThreadPoolExecutor
      - decode + parse merged into single parallel step
      - Small N skips thread pool (sequential is faster)
      - Warmup uses layout prompt for realistic kernel compilation
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: str = "CPU",
        precision: str = "fp16",
        image_analysis: bool = False,
        layout_image_size: Tuple[int, int] = DEFAULT_LAYOUT_IMAGE_SIZE,
        min_image_edge: int = DEFAULT_MIN_IMAGE_EDGE,
        max_image_edge_ratio: float = DEFAULT_MAX_IMAGE_EDGE_RATIO,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_concurrent: int = 8,
        layout_max_new_tokens: int = DEFAULT_LAYOUT_MAX_NEW_TOKENS,
        retry_on_truncation: bool = True,
        max_new_tokens_ceiling: int = DEFAULT_MAX_NEW_TOKENS_CEILING,
        npu_max_new_tokens: int = 2048,
        ov_config: Optional[dict] = None,
        enable_warmup: bool = True,
        debug: bool = False,
        cpu_workers: int = _CPU_WORKERS,
    ) -> None:
        try:
            from mineru_vl_utils.mineru_client import (
                DEFAULT_PROMPTS,
                DEFAULT_SAMPLING_PARAMS,
                MinerUClientHelper,
            )
        except ImportError as e:
            raise ImportError(
                "mineru-vl-utils 未安装或版本过低，请运行: "
                "pip install 'mineru-vl-utils>=0.2.7'"
            ) from e

        self.model_dir = Path(model_dir)
        self.device = device
        self._max_new_tokens = max_new_tokens
        self._layout_max_new_tokens = layout_max_new_tokens
        self._cpu_workers = cpu_workers
        self._max_new_tokens_ceiling = max(
            max_new_tokens_ceiling, max_new_tokens, layout_max_new_tokens
        )
        self._retry_attempts = _TRUNCATION_MAX_RETRIES if retry_on_truncation else 0

        layout_sp = DEFAULT_SAMPLING_PARAMS.get("[layout]") if DEFAULT_SAMPLING_PARAMS else None
        layout_prompt = DEFAULT_PROMPTS.get("[layout]", "") if DEFAULT_PROMPTS else ""

        self.engine = BatchInferenceEngine(
            model_dir=str(model_dir),
            device=device,
            max_concurrent=max_concurrent,
            max_new_tokens=max_new_tokens,
            ov_config=ov_config,
            enable_warmup=enable_warmup,
            warmup_prompt=layout_prompt,
        )

        self.helper = MinerUClientHelper(
            backend="openvino",
            prompts=DEFAULT_PROMPTS,
            sampling_params=DEFAULT_SAMPLING_PARAMS,
            layout_image_size=layout_image_size,
            min_image_edge=min_image_edge,
            max_image_edge_ratio=max_image_edge_ratio,
            simple_post_process=False,
            handle_equation_block=True,
            abandon_list=False,
            abandon_paratext=False,
            image_analysis=image_analysis,
            debug=debug,
        )

        # 版面采样参数留作按预算重建生成配置用（截断重试会抬高预算）
        self._layout_sp = layout_sp

        logger.info("BatchMinerUClient ready (device=%s)", device)

    def _batch_predict(
        self,
        items: list[tuple[Image.Image, str, object]],
        page_hint: str = "",
    ) -> list[str]:
        """批量内容提取；被截断的先自动加大预算重跑，仍截断则汇总告警。"""
        label = f"内容提取{'（' + page_hint + '）' if page_hint else ''}"
        texts_with_flags = _escalating_generate(
            self.engine,
            items,
            build_generation_config,
            self._max_new_tokens,
            ceiling=self._max_new_tokens_ceiling,
            max_attempts=self._retry_attempts,
            label=label,
        )
        warn_if_blocks_truncated(
            [flag for _, flag in texts_with_flags],
            budget=self._max_new_tokens,
            label=label,
        )
        return [text for text, _ in texts_with_flags]

    def _layout_blocks(
        self,
        items: list[tuple[Image.Image, str, object]],
        page_start: int = 1,
    ) -> list[list]:
        """批量版面检测并解析为区块列表。

        被截断的会先自动加大预算重跑，仍截断或存在未闭合的框时按页告警。
        """
        texts_with_flags = _escalating_generate(
            self.engine,
            items,
            lambda _sp, budget: build_generation_config(self._layout_sp, budget),
            self._layout_max_new_tokens,
            ceiling=self._max_new_tokens_ceiling,
            max_attempts=self._retry_attempts,
            label="版面检测",
        )
        return [
            parse_layout_output_checked(
                self.helper,
                text,
                page_hint=f"第 {page_start + offset} 页",
                layout_max_new_tokens=self._layout_max_new_tokens,
                length_stopped=truncated,
            )
            for offset, (text, truncated) in enumerate(texts_with_flags)
        ]

    def two_step_extract(
        self,
        image: ImageInput,
        not_extract_list: Optional[Sequence[str]] = None,
        image_analysis: Optional[bool] = None,
        pause_check: Optional[Callable[[], bool]] = None,
    ):
        from mineru_vl_utils.structs import ExtractResult

        page = load_image(image)
        if page.mode != 'RGB':
            page = page.convert('RGB')

        t_layout = time.time()
        layout_image = self.helper.prepare_for_layout(page)
        layout_prompt = self.helper.prompts["[layout]"]
        layout_sp = self.helper.sampling_params.get("[layout]")
        blocks = self._layout_blocks(
            [(layout_image, layout_prompt, layout_sp)], page_start=1
        )[0]
        logger.info("layout_detect: %.2fs, %d blocks", time.time() - t_layout, len(blocks))

        if pause_check is not None and pause_check():
            logger.info("收到暂停信号，返回已检测版面的部分结果")
            return ExtractResult(blocks)

        t_extract = time.time()
        block_images, prompts, sps, indices = self.helper.prepare_for_extract(
            page, blocks,
            not_extract_list=list(not_extract_list) if not_extract_list else None,
            image_analysis=image_analysis,
        )
        if not block_images:
            return ExtractResult(blocks)

        if pause_check is not None and pause_check():
            logger.info("收到暂停信号，跳过批处理提取")
            return ExtractResult(blocks)

        infer_items = list(zip(block_images, prompts, sps))
        contents = self._batch_predict(infer_items, page_hint="第 1 页")

        from mineru_vl_utils.post_process.table_image_processor import (
            TABLE_IMAGE_TOKEN_MAP_KEY, replace_table_image_tokens,
        )
        for idx, content in zip(indices, contents):
            block = blocks[idx]
            if TABLE_IMAGE_TOKEN_MAP_KEY in block:
                token_map = block.get(TABLE_IMAGE_TOKEN_MAP_KEY) or {}
                if token_map:
                    content = replace_table_image_tokens(content, token_map)
            block["content"] = content

        logger.info(
            "content_extract: %.2fs (%d blocks, batched)",
            time.time() - t_extract, len(block_images),
        )
        blocks = self.helper.post_process(blocks)
        return ExtractResult(blocks)

    def image_to_markdown(self, image: ImageInput, **kwargs) -> Tuple[str, object]:
        from mineru_vl_utils.post_process import json2md
        blocks = self.two_step_extract(image, **kwargs)
        return json2md(list(blocks)), blocks

    def batch_process_pages(
        self,
        page_images: list[Image.Image],
        progress_callback=None,
        **kwargs,
    ) -> list[list]:
        """两阶段批量处理多页图像：全部版面检测 → 全部内容提取。

        v20 optimizations applied:
          - prepare_for_layout parallelized
          - decode + parse_layout_output merged and parallelized
          - prepare_for_extract parallelized
          - post_process parallelized
          - GenerationConfig cached for layout and default block
          - Small N skips thread pool

        Parameters
        ----------
        page_images : list[Image.Image]
            已渲染好的页面图像列表
        progress_callback : 可选进度回调 (current, total, message)
        **kwargs : 传递给 prepare_for_extract 的参数
            - not_extract_list
            - image_analysis

        Returns
        -------
        list[list]: 每页的 ContentBlock 列表
        """
        n_pages = len(page_images)
        if n_pages == 0:
            return []

        if progress_callback:
            progress_callback(0, n_pages, "Phase 1/2 — 布局检测...")

        layout_prompt = self.helper.prompts["[layout]"]
        layout_sp = self.helper.sampling_params.get("[layout]")
        extract_kwargs = dict(
            not_extract_list=list(kwargs.get("not_extract_list") or []),
            image_analysis=kwargs.get("image_analysis"),
        )

        # ── Phase 1: parallel prepare_for_layout ──
        t0 = time.time()
        if n_pages <= 1:
            layout_images = [self.helper.prepare_for_layout(p) for p in page_images]
        else:
            with ThreadPoolExecutor(max_workers=min(self._cpu_workers, n_pages)) as pool:
                layout_images = list(pool.map(self.helper.prepare_for_layout, page_images))
        logger.info("Phase 1 prepared %d layout images in %.2fs", n_pages, time.time() - t0)

        # Batch-submit ALL layouts, then parse each (truncation is reported per page).
        layout_items = [(img, layout_prompt, layout_sp) for img in layout_images]
        del layout_images
        all_blocks = self._layout_blocks(layout_items, page_start=1)
        logger.info("Phase 1 done: %d layouts batched.", n_pages)

        if progress_callback:
            progress_callback(0, n_pages, "Phase 2/2 — 内容提取...")

        # ── Phase 2: parallel prepare_for_extract ──
        t0 = time.time()

        def _prep_one_page(page_idx: int):
            b_imgs, b_prompts, b_sps, b_indices = self.helper.prepare_for_extract(
                page_images[page_idx], all_blocks[page_idx], **extract_kwargs,
            )
            return page_idx, b_imgs, b_prompts, b_sps, b_indices

        if n_pages <= 1:
            prep_results = [_prep_one_page(i) for i in range(n_pages)]
        else:
            with ThreadPoolExecutor(max_workers=min(self._cpu_workers, n_pages)) as pool:
                prep_results = list(pool.map(_prep_one_page, range(n_pages)))

        all_block_items: list[tuple[int, int, Image.Image, str, object]] = []
        for page_idx, b_imgs, b_prompts, b_sps, b_indices in prep_results:
            for idx, img, prompt, sp in zip(b_indices, b_imgs, b_prompts, b_sps):
                all_block_items.append((page_idx, idx, img, prompt, sp))
        del prep_results

        n_blocks_total = len(all_block_items)
        logger.info("Phase 2 prepared %d blocks in %.2fs", n_blocks_total, time.time() - t0)

        if all_block_items:
            infer_items = [(img, prompt, sp) for (_, _, img, prompt, sp) in all_block_items]
            contents = self._batch_predict(infer_items)
            from mineru_vl_utils.post_process.table_image_processor import (
                TABLE_IMAGE_TOKEN_MAP_KEY, replace_table_image_tokens,
            )
            for (page_idx, block_idx, _, _, _), content in zip(all_block_items, contents):
                block = all_blocks[page_idx][block_idx]
                if TABLE_IMAGE_TOKEN_MAP_KEY in block:
                    token_map = block.get(TABLE_IMAGE_TOKEN_MAP_KEY) or {}
                    if token_map:
                        content = replace_table_image_tokens(content, token_map)
                block["content"] = content
        logger.info("Phase 2 done: %d blocks batched.", n_blocks_total)

        # ── Phase 3: parallel post_process ──
        if n_pages <= 2:
            result = [self.helper.post_process(b) for b in all_blocks]
        else:
            with ThreadPoolExecutor(max_workers=min(self._cpu_workers, n_pages)) as pool:
                result = list(pool.map(self.helper.post_process, all_blocks))

        for page_idx in range(n_pages):
            if progress_callback:
                progress_callback(page_idx + 1, n_pages, f"Page {page_idx + 1}/{n_pages}")
        if progress_callback:
            progress_callback(n_pages, n_pages, "完成")
        return result

    def pdf_to_markdown(
        self,
        pdf: Union[str, Path, bytes],
        dpi: int = 200,
        progress_callback=None,
        **kwargs,
    ) -> Tuple[str, List]:
        """两阶段批量 PDF 解析：所有布局检测 → 所有内容提取。"""
        from .pdf_utils import load_images_from_path
        pages = list(load_images_from_path(pdf, dpi=dpi))
        n_pages = len(pages)
        if n_pages == 0:
            return "", []

        page_images = [img for _, _, img in pages]
        all_blocks = self.batch_process_pages(
            page_images,
            progress_callback=progress_callback,
            **kwargs,
        )

        from mineru_vl_utils.post_process import json2md
        md_pages: List[str] = []
        block_pages: List = []
        for page_idx, blocks in enumerate(all_blocks):
            md_pages.append(json2md(list(blocks)))
            block_pages.append(blocks)
        return "\n\n---\n\n".join(md_pages), block_pages

    def unload(self) -> None:
        if self.engine:
            self.engine.unload()
            self.engine = None
        logger.info("BatchMinerUClient unloaded")


def create_ov_vlm_client(cfg):
    """从配置对象创建推理客户端（支持普通版和 Batch 版）。

    Parameters
    ----------
    cfg : OVConfig 配置对象

    Returns
    -------
    OVMinerUClient 或 BatchMinerUClient
    """
    use_batch = getattr(cfg, "use_batch_backend", False)
    kwargs = dict(
        model_dir=str(cfg.vlm_model_dir),
        device=cfg.device,
        precision=cfg.precision,
        image_analysis=cfg.image_analysis,
        layout_image_size=cfg.layout_image_size,
        max_new_tokens=cfg.max_new_tokens,
        layout_max_new_tokens=cfg.layout_max_new_tokens,
        retry_on_truncation=getattr(cfg, "retry_on_truncation", True),
        max_new_tokens_ceiling=getattr(
            cfg, "max_new_tokens_ceiling", DEFAULT_MAX_NEW_TOKENS_CEILING
        ),
        enable_warmup=cfg.enable_warmup,
    )
    if use_batch:
        kwargs["max_concurrent"] = getattr(cfg, "max_concurrent", 8)
        kwargs["ov_config"] = getattr(cfg, "ov_config", None)
        return BatchMinerUClient(**kwargs)
    return OVMinerUClient(**kwargs)
