#!/usr/bin/env python3
"""Render deterministic Chinese patent figures for the PiERN patent drafts.

The figures are text-heavy patent diagrams. A deterministic renderer keeps all
Chinese labels exact while still giving the drawings a polished, review-ready
layout.
"""

from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/patents/figures"
DEFAULT_PATENT_OUTPUT_DIR = REPO_ROOT / "docs/patents/figures_patent"

WIDTH = 1800
HEIGHT = 1000
FIGURE_BOTTOM = 905
CAPTION_Y = 950

REGULAR_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
)
BOLD_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
)

INK = "#172033"
MUTED = "#5b6476"
BORDER = "#9aa6b8"
RULE = "#516179"
CARD = "#ffffff"
SURFACE = "#f7f9fc"
HEADER = "#eef4fb"
HEADER_ALT = "#f2f7f2"
HEADER_WARN = "#fff7e6"
BLUE = "#2f6ea9"
GREEN = "#2f7d5a"
AMBER = "#a66a00"
PATENT_STYLE = False
SHADOW = "#d9e1ec"
HEADER_LINE = "#c8d2df"
LABEL_OUTLINE = "#d4dde8"
CAPTION_FILL = "#0f172a"
CARD_RADIUS = 22
CANVAS_RADIUS = 18
CARD_BORDER_WIDTH = 3


def apply_render_style(style: str) -> None:
    """Apply either the polished review style or the black-white filing style."""

    global AMBER
    global BLUE
    global BORDER
    global CANVAS_RADIUS
    global CAPTION_FILL
    global CARD
    global CARD_BORDER_WIDTH
    global CARD_RADIUS
    global GREEN
    global HEADER
    global HEADER_ALT
    global HEADER_LINE
    global HEADER_WARN
    global INK
    global LABEL_OUTLINE
    global MUTED
    global PATENT_STYLE
    global RULE
    global SHADOW
    global SURFACE

    if style == "polished":
        INK = "#172033"
        MUTED = "#5b6476"
        BORDER = "#9aa6b8"
        RULE = "#516179"
        CARD = "#ffffff"
        SURFACE = "#f7f9fc"
        HEADER = "#eef4fb"
        HEADER_ALT = "#f2f7f2"
        HEADER_WARN = "#fff7e6"
        BLUE = "#2f6ea9"
        GREEN = "#2f7d5a"
        AMBER = "#a66a00"
        PATENT_STYLE = False
        SHADOW = "#d9e1ec"
        HEADER_LINE = "#c8d2df"
        LABEL_OUTLINE = "#d4dde8"
        CAPTION_FILL = "#0f172a"
        CARD_RADIUS = 22
        CANVAS_RADIUS = 18
        CARD_BORDER_WIDTH = 3
        return

    if style == "patent":
        INK = "#000000"
        MUTED = "#000000"
        BORDER = "#000000"
        RULE = "#000000"
        CARD = "#ffffff"
        SURFACE = "#ffffff"
        HEADER = "#ffffff"
        HEADER_ALT = "#ffffff"
        HEADER_WARN = "#ffffff"
        BLUE = "#000000"
        GREEN = "#000000"
        AMBER = "#000000"
        PATENT_STYLE = True
        SHADOW = ""
        HEADER_LINE = "#000000"
        LABEL_OUTLINE = "#000000"
        CAPTION_FILL = "#000000"
        CARD_RADIUS = 0
        CANVAS_RADIUS = 0
        CARD_BORDER_WIDTH = 3
        return

    raise ValueError(f"未知附图样式: {style}")


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    title: str
    bullets: tuple[str, ...] = ()
    fill: str = CARD
    header: str = HEADER
    border: str = BORDER

    @property
    def left(self) -> int:
        return self.x

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def top(self) -> int:
        return self.y

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


def first_existing(paths: Sequence[str]) -> str:
    for path in paths:
        if Path(path).exists():
            return path
    raise RuntimeError("No CJK font found. Install Noto Sans CJK or adjust font paths.")


REGULAR_PATH = first_existing(REGULAR_FONT_CANDIDATES)
BOLD_PATH = first_existing(BOLD_FONT_CANDIDATES)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD_PATH if bold else REGULAR_PATH, size)


F = {
    "title": font(34, bold=True),
    "caption": font(34, bold=True),
    "card_title": font(30, bold=True),
    "card_title_small": font(25, bold=True),
    "body": font(25),
    "body_small": font(22),
    "label": font(22, bold=True),
    "tiny": font(20),
}


def text_size(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=text_font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    if hasattr(text_font, "getbbox"):
        bbox = text_font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    return draw.textsize(text, font=text_font)


def centered_text(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str = INK,
) -> None:
    x1, y1, x2, y2 = rect
    lines = text.split("\n")
    line_heights = [text_size(draw, line, text_font)[1] for line in lines]
    total_h = sum(line_heights) + (len(lines) - 1) * 8
    y = y1 + (y2 - y1 - total_h) / 2 - 2
    for idx, line in enumerate(lines):
        w, h = text_size(draw, line, text_font)
        draw.text((x1 + (x2 - x1 - w) / 2, y), line, font=text_font, fill=fill)
        y += h + 8


def wrap_cjk(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    return textwrap.wrap(text, width=max_chars, break_long_words=True, break_on_hyphens=False)


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float, float, float],
    *,
    radius: int = 0,
    fill: str | None = None,
    outline: str | None = None,
    width: int = 1,
) -> None:
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
        return

    x1, y1, x2, y2 = (int(v) for v in xy)
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if fill:
        draw.rectangle((x1 + r, y1, x2 - r, y2), fill=fill)
        draw.rectangle((x1, y1 + r, x2, y2 - r), fill=fill)
        draw.pieslice((x1, y1, x1 + 2 * r, y1 + 2 * r), 180, 270, fill=fill)
        draw.pieslice((x2 - 2 * r, y1, x2, y1 + 2 * r), 270, 360, fill=fill)
        draw.pieslice((x2 - 2 * r, y2 - 2 * r, x2, y2), 0, 90, fill=fill)
        draw.pieslice((x1, y2 - 2 * r, x1 + 2 * r, y2), 90, 180, fill=fill)
    if outline:
        for offset in range(width):
            xx1, yy1, xx2, yy2 = x1 + offset, y1 + offset, x2 - offset, y2 - offset
            rr = max(0, r - offset)
            draw.line((xx1 + rr, yy1, xx2 - rr, yy1), fill=outline)
            draw.line((xx2, yy1 + rr, xx2, yy2 - rr), fill=outline)
            draw.line((xx2 - rr, yy2, xx1 + rr, yy2), fill=outline)
            draw.line((xx1, yy2 - rr, xx1, yy1 + rr), fill=outline)
            if rr > 0:
                draw.arc((xx1, yy1, xx1 + 2 * rr, yy1 + 2 * rr), 180, 270, fill=outline)
                draw.arc((xx2 - 2 * rr, yy1, xx2, yy1 + 2 * rr), 270, 360, fill=outline)
                draw.arc((xx2 - 2 * rr, yy2 - 2 * rr, xx2, yy2), 0, 90, fill=outline)
                draw.arc((xx1, yy2 - 2 * rr, xx1 + 2 * rr, yy2), 90, 180, fill=outline)


def draw_card(draw: ImageDraw.ImageDraw, box: Box, *, compact: bool = False) -> None:
    radius = CARD_RADIUS
    fill = CARD if PATENT_STYLE else box.fill
    header = HEADER if PATENT_STYLE else box.header
    border = BORDER if PATENT_STYLE else box.border
    if SHADOW:
        shadow = (box.x + 5, box.y + 6, box.right + 5, box.bottom + 6)
        rounded_rect(draw, shadow, radius=radius, fill=SHADOW)
    rounded_rect(
        draw,
        (box.x, box.y, box.right, box.bottom),
        radius=radius,
        fill=fill,
        outline=border,
        width=CARD_BORDER_WIDTH,
    )

    if not box.bullets and box.h <= 130:
        rounded_rect(
            draw,
            (box.x, box.y, box.right, box.bottom),
            radius=radius,
            fill=header,
            outline=border,
            width=CARD_BORDER_WIDTH,
        )
        title_font = F["card_title_small"] if len(box.title) > 10 else F["card_title"]
        centered_text(draw, (box.x + 16, box.y + 2, box.right - 16, box.bottom - 2), box.title, title_font)
        return

    header_h = 62 if compact else 72
    rounded_rect(draw, (box.x, box.y, box.right, box.y + header_h + radius), radius=radius, fill=header)
    draw.rectangle((box.x, box.y + header_h, box.right, box.y + header_h + radius), fill=fill)
    draw.line((box.x + 1, box.y + header_h, box.right - 1, box.y + header_h), fill=HEADER_LINE, width=2)

    title_font = F["card_title_small"] if compact or len(box.title) > 10 else F["card_title"]
    centered_text(draw, (box.x + 16, box.y + 7, box.right - 16, box.y + header_h - 2), box.title, title_font)

    body_font = F["body_small"] if compact else F["body"]
    line_step = 34 if compact else 39
    max_chars = max(8, int((box.w - 58) / (body_font.size * 0.56)))
    wrapped_bullets = [wrap_cjk(bullet, max_chars) for bullet in box.bullets]
    needed_lines = sum(len(lines) for lines in wrapped_bullets)
    available_h = box.h - header_h - 34
    if box.bullets and needed_lines * line_step > available_h:
        body_font = F["tiny"]
        line_step = 29
        max_chars = max(8, int((box.w - 58) / (body_font.size * 0.56)))
        wrapped_bullets = [wrap_cjk(bullet, max_chars) for bullet in box.bullets]
        needed_lines = sum(len(lines) for lines in wrapped_bullets)
    y = box.y + header_h + 20
    if box.bullets and needed_lines * line_step < available_h - 18:
        y += min(12, (available_h - needed_lines * line_step) // 4)
    for wrapped in wrapped_bullets:
        for line_idx, line in enumerate(wrapped):
            prefix = "• " if line_idx == 0 else "  "
            draw.text((box.x + 34, y), prefix + line, font=body_font, fill=INK)
            y += line_step


def arrow_head(draw: ImageDraw.ImageDraw, tip: tuple[int, int], angle: float, color: str, size: int = 18) -> None:
    tx, ty = tip
    pts = []
    for delta in (math.pi * 0.84, -math.pi * 0.84):
        pts.append((tx + math.cos(angle + delta) * size, ty + math.sin(angle + delta) * size))
    draw.polygon([tip, pts[0], pts[1]], fill=color)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = RULE,
    width: int = 4,
    dashed: bool = False,
    label: str | None = None,
    label_offset: tuple[int, int] = (0, -28),
) -> None:
    if PATENT_STYLE:
        color = RULE
    x1, y1 = start
    x2, y2 = end
    if dashed:
        length = math.hypot(x2 - x1, y2 - y1)
        steps = max(1, int(length // 18))
        for i in range(steps):
            if i % 2 == 0:
                a = i / steps
                b = (i + 1) / steps
                draw.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), fill=color, width=width)
    else:
        draw.line((start, end), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    arrow_head(draw, end, angle, color)
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        tw, th = text_size(draw, label, F["label"])
        pad_x, pad_y = 14, 8
        rounded_rect(
            draw,
            (mx - tw / 2 - pad_x, my - th / 2 - pad_y, mx + tw / 2 + pad_x, my + th / 2 + pad_y),
            radius=12,
            fill="#ffffff",
            outline=LABEL_OUTLINE,
            width=2,
        )
        draw.text((mx - tw / 2, my - th / 2 - 2), label, font=F["label"], fill=INK)


def draw_poly_arrow(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[int, int]],
    *,
    color: str = RULE,
    width: int = 4,
    dashed: bool = False,
    label: str | None = None,
    label_at: tuple[int, int] | None = None,
) -> None:
    for start, end in zip(points, points[1:]):
        is_last = end == points[-1]
        draw_arrow(draw, start, end, color=color, width=width, dashed=dashed and not is_last)
        if not is_last:
            # Hide intermediate arrow heads so the path reads as one routed arrow.
            x, y = end
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=SURFACE, outline=SURFACE)
    if label and label_at:
        tw, th = text_size(draw, label, F["label"])
        x, y = label_at
        rounded_rect(
            draw,
            (x - 16, y - 9, x + tw + 16, y + th + 10),
            radius=0 if PATENT_STYLE else 12,
            fill="#ffffff",
            outline=LABEL_OUTLINE,
            width=2,
        )
        draw.text((x, y - 2), label, font=F["label"], fill=INK)


def make_canvas(caption: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(image)
    rounded_rect(
        draw,
        (24, 24, WIDTH - 24, FIGURE_BOTTOM),
        radius=CANVAS_RADIUS,
        fill=SURFACE,
        outline=BORDER,
        width=3,
    )
    draw.rectangle((24, FIGURE_BOTTOM - 12, WIDTH - 24, FIGURE_BOTTOM), fill="#ffffff")
    tw, _ = text_size(draw, caption, F["caption"])
    draw.text(((WIDTH - tw) / 2, CAPTION_Y - 17), caption, font=F["caption"], fill=CAPTION_FILL)
    return image, draw


def save(image: Image.Image, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if PATENT_STYLE:
        image = image.convert("L").convert("RGB")
    image.save(output_dir / name, optimize=True)


def draw_row_flow(draw: ImageDraw.ImageDraw, boxes: Sequence[Box], *, color: str = BLUE) -> None:
    for box in boxes:
        draw_card(draw, box, compact=box.w < 240)
    for left, right in zip(boxes, boxes[1:]):
        draw_arrow(draw, (left.right + 2, left.y + left.h // 2), (right.x - 2, right.y + right.h // 2), color=color)


def data_synthesis_fig1(output_dir: Path) -> None:
    image, draw = make_canvas("图 1 数据合成平台总体架构图")
    top = [
        Box(60, 90, 240, 245, "原始科学数据", ("时序数组", "参数数组", "场景属性"), header=HEADER),
        Box(335, 90, 240, 245, "统一校验", ("形状校验", "有限值校验", "一致性校验"), header=HEADER),
        Box(610, 90, 240, 245, "场景知识", ("仿真器注册", "参数说明", "输出说明"), header=HEADER_ALT),
        Box(885, 90, 240, 245, "无数值模板", ("语言结构", "槽位索引", "变换规则"), header="#f4f0fb"),
        Box(1160, 90, 240, 245, "本地数值填充", ("读取真实数值", "确定性填充", "保存映射"), header=HEADER_WARN),
        Box(1435, 90, 300, 245, "路由数据构建", ("正负样本", "场景标签", "源数据签名"), header="#eef7f3"),
    ]
    draw_row_flow(draw, top)
    task = Box(255, 555, 530, 230, "任务进度管理", ("任务编排", "进度监控", "异常处理"), header="#eef4fb")
    store = Box(970, 535, 650, 275, "文件存储管理", ("原始数据", "模板数据", "语言样本", "路由数据", "缓存清单"), header="#eef7f3")
    draw_card(draw, task)
    draw_card(draw, store)
    draw_arrow(draw, (top[1].x + top[1].w // 2, top[1].bottom + 2), (task.x + task.w // 2, task.y - 2), color=GREEN)
    draw_arrow(draw, (top[4].x + top[4].w // 2, top[4].bottom + 2), (store.x + store.w // 2, store.y - 2), color=GREEN)
    draw_arrow(draw, (task.right + 2, task.y + task.h // 2), (store.x - 2, store.y + store.h // 2), color=BLUE)
    save(image, output_dir, "data_synthesis_fig1.png")


def data_synthesis_fig2(output_dir: Path) -> None:
    image, draw = make_canvas("图 2 四阶段数据合成流水线流程图")
    boxes = [
        Box(110, 120, 310, 500, "阶段一 数据接入", ("读取数据", "统一契约校验", "生成数据摘要"), header=HEADER),
        Box(530, 120, 310, 500, "阶段二 模板生成", ("融合场景知识", "生成无数值模板", "保存槽位映射"), header="#f4f0fb"),
        Box(950, 120, 310, 500, "阶段三 样本填充", ("选择模板和样本", "读取真实数值", "本地确定性填充"), header=HEADER_WARN),
        Box(1370, 120, 310, 500, "阶段四 路由构建", ("应用聊天模板", "构造正负样本", "写入训练数据"), header="#eef7f3"),
    ]
    draw_row_flow(draw, boxes)
    status = Box(310, 715, 1180, 120, "统一任务状态与文件保护", ("运行状态、处理速率、日志事件、受保护文件列表",), header="#f1f5f9")
    draw_card(draw, status, compact=True)
    status_targets = [430, 720, 1080, 1370]
    for box, target_x in zip(boxes, status_targets):
        draw_poly_arrow(draw, [(box.x + box.w // 2, box.bottom + 2), (box.x + box.w // 2, 675), (target_x, status.y - 2)], color=GREEN, width=3)
    save(image, output_dir, "data_synthesis_fig2.png")


def data_synthesis_fig3(output_dir: Path) -> None:
    image, draw = make_canvas("图 3 无数值语言模板与本地数值填充映射图")
    template = Box(70, 110, 330, 635, "无数值模板", ("问题文本", "参数占位", "输出占位", "时间选择", "通道选择"), header="#f4f0fb")
    mapping = Box(520, 90, 740, 300, "映射关系", ("参数槽位 → 参数数组列", "输出槽位 → 通道与时间", "变换规则 → 本地计算", "样本编号 → 原始记录"), header=HEADER)
    engine = Box(520, 520, 740, 245, "本地填充引擎", ("解析模板", "读取数组", "应用变换", "生成语言样本"), header=HEADER_WARN)
    sample = Box(1390, 170, 330, 520, "填充后样本", ("问题文本", "真实参数", "观测数值", "样本元数据"), header="#eef7f3")
    param = Box(600, 805, 220, 72, "参数数组", (), header="#ffffff")
    series = Box(960, 805, 220, 72, "时序数组", (), header="#ffffff")
    for box in (template, mapping, engine, sample, param, series):
        draw_card(draw, box, compact=box.h <= 80)
    draw_arrow(draw, (template.right + 2, 395), (mapping.x - 2, 255), color=BLUE)
    draw_arrow(draw, (mapping.x + mapping.w // 2, mapping.bottom + 2), (engine.x + engine.w // 2, engine.y - 2), color=BLUE)
    draw_arrow(draw, (engine.right + 2, 640), (sample.x - 2, 430), color=BLUE)
    draw_arrow(draw, (param.x + param.w // 2, param.y - 2), (engine.x + 190, engine.bottom + 2), color=GREEN)
    draw_arrow(draw, (series.x + series.w // 2, series.y - 2), (engine.x + 540, engine.bottom + 2), color=GREEN)
    save(image, output_dir, "data_synthesis_fig3.png")


def data_synthesis_fig4(output_dir: Path) -> None:
    image, draw = make_canvas("图 4 路由正负样本构建示意图")
    full = Box(80, 125, 400, 600, "完整语言样本", ("系统提示", "用户问题", "助手响应", "场景标签"), header=HEADER)
    pos = Box(690, 125, 420, 440, "正样本构造", ("截取助手响应前上下文", "标签：需要专家接管", "保存场景与源签名"), header="#eef7f3")
    neg = Box(1290, 125, 420, 440, "负样本构造", ("从前缀随机截断", "标签：尚不接管", "保存截断位置"), header=HEADER_WARN)
    pos_out = Box(710, 735, 310, 75, "路由训练正样本", (), header="#eef7f3")
    neg_out = Box(1325, 735, 310, 75, "路由训练负样本", (), header=HEADER_WARN)
    for box in (full, pos, neg, pos_out, neg_out):
        draw_card(draw, box, compact=box.h <= 90)
    draw_arrow(draw, (full.right + 2, 350), (pos.x - 2, 350), color=GREEN, label="助手起始位置", label_offset=(0, -36))
    draw_arrow(draw, (full.right + 2, 520), (neg.x - 2, 520), color=AMBER, label="随机前缀截断", label_offset=(0, 34))
    draw_arrow(draw, (pos.x + pos.w // 2, pos.bottom + 2), (pos_out.x + pos_out.w // 2, pos_out.y - 2), color=GREEN)
    draw_arrow(draw, (neg.x + neg.w // 2, neg.bottom + 2), (neg_out.x + neg_out.w // 2, neg_out.y - 2), color=AMBER)
    save(image, output_dir, "data_synthesis_fig4.png")


def data_synthesis_fig5(output_dir: Path) -> None:
    image, draw = make_canvas("图 5 任务进度、文件保护和派生数据存储关系图")
    task = Box(80, 130, 360, 600, "运行任务", ("模板生成", "样本填充", "路由构建", "训练任务"), header=HEADER)
    state = Box(570, 85, 660, 305, "任务状态管理", ("任务标识", "阶段进度", "处理速率", "日志事件", "错误信息"), header=HEADER)
    guard = Box(570, 525, 660, 270, "文件保护判断", ("读取文件列表", "写入目录列表", "保护状态查询", "拒绝删除或覆盖"), header=HEADER_WARN)
    store = Box(1400, 130, 320, 600, "存储对象", ("原始数据", "语言模板", "语言样本", "路由数据", "清单与签名"), header="#eef7f3")
    for box in (task, state, guard, store):
        draw_card(draw, box)
    draw_arrow(draw, (task.right + 2, 300), (state.x - 2, 230), color=BLUE)
    draw_arrow(draw, (state.x + state.w // 2, state.bottom + 2), (guard.x + guard.w // 2, guard.y - 2), color=BLUE)
    draw_arrow(draw, (guard.right + 2, 650), (store.x - 2, 540), color=AMBER)
    draw_arrow(draw, (state.right + 2, 240), (store.x - 2, 300), color=GREEN, dashed=True, label="状态同步", label_offset=(0, -36))
    save(image, output_dir, "data_synthesis_fig5.png")


def auto_training_fig1(output_dir: Path) -> None:
    image, draw = make_canvas("图 1 自动训练平台总体架构图")
    titles = [
        ("训练工作台", ("选择数据", "提交配置", "查看曲线")),
        ("训练管理器", ("任务编排", "资源校验", "状态监控")),
        ("数据集发现", ("场景聚合", "样本统计", "元数据读取")),
        ("图形处理器锁", ("显存状态", "利用率状态", "活跃任务锁")),
        ("预处理缓存", ("确定性名称", "训练测试切分", "输入缓存")),
        ("训练子进程", ("加载数据", "模型训练", "评估验证")),
        ("日志与权重", ("日志文件", "指标文件", "检查点文件")),
    ]
    boxes = []
    x = 50
    for title, bullets in titles:
        boxes.append(Box(x, 95, 205, 485, title, bullets, header=HEADER if len(boxes) < 3 else "#eef7f3"))
        x += 245
    draw_row_flow(draw, boxes)
    manager = Box(450, 705, 1210, 135, "文件管理器", ("数据集、缓存、检查点、日志、空间与权限管理",), header="#f1f5f9")
    draw_card(draw, manager, compact=True)
    for box in boxes[2:]:
        draw_arrow(draw, (box.x + box.w // 2, box.bottom + 2), (box.x + box.w // 2, manager.y - 2), color=GREEN, width=3)
    save(image, output_dir, "auto_training_fig1.png")


def auto_training_fig2(output_dir: Path) -> None:
    image, draw = make_canvas("图 2 训练任务创建与图形处理器锁定流程图")
    boxes = [
        Box(80, 160, 260, 480, "提交训练配置", ("仿真器", "场景集合", "训练参数"), header=HEADER),
        Box(425, 160, 260, 480, "初步校验", ("数据存在", "参数有效", "模型可解析"), header=HEADER),
        Box(770, 160, 260, 480, "进入互斥锁", ("刷新资源", "读取活跃任务"), header="#f1f5f9"),
        Box(1115, 160, 260, 480, "二次图形处理器校验", ("显存阈值", "利用率阈值", "平台锁定状态"), header=HEADER_WARN),
        Box(1460, 160, 260, 480, "创建训练任务", ("写入任务记录", "设置运行目录", "标记资源占用"), header="#eef7f3"),
    ]
    draw_row_flow(draw, boxes)
    reject = Box(1215, 740, 380, 85, "校验失败：拒绝启动并返回原因", (), header="#fff1f2", border="#e5a6ad")
    draw_card(draw, reject, compact=True)
    draw_arrow(draw, (boxes[3].x + boxes[3].w // 2, boxes[3].bottom + 2), (reject.x + reject.w // 2, reject.y - 2), color="#b84a5a", dashed=True)
    save(image, output_dir, "auto_training_fig2.png")


def auto_training_fig3(output_dir: Path) -> None:
    image, draw = make_canvas("图 3 路由训练数据到预处理缓存构建流程图")
    data = Box(80, 120, 350, 625, "路由训练数据", ("正负样本", "场景标签", "聊天模板", "嵌入元数据"), header=HEADER)
    key = Box(610, 95, 600, 300, "缓存名称计算", ("仿真器", "排序后场景", "测试比例", "输入表示", "嵌入模型与分词器"), header=HEADER)
    cache = Box(610, 500, 600, 310, "缓存内容", ("训练测试切分", "场景标识映射", "标签数组", "嵌入或词元缓存", "预处理摘要"), header=HEADER_WARN)
    read = Box(1390, 180, 330, 520, "训练读取", ("命中缓存", "摘要校验", "批量加载"), header="#eef7f3")
    for box in (data, key, cache, read):
        draw_card(draw, box)
    draw_arrow(draw, (data.right + 2, 405), (key.x - 2, 245), color=BLUE)
    draw_arrow(draw, (key.x + key.w // 2, key.bottom + 2), (cache.x + cache.w // 2, cache.y - 2), color=BLUE)
    draw_arrow(draw, (cache.right + 2, 650), (read.x - 2, 440), color=GREEN)
    save(image, output_dir, "auto_training_fig3.png")


def auto_training_fig4(output_dir: Path) -> None:
    image, draw = make_canvas("图 4 训练状态多源推断流程图")
    sources = [
        Box(80, 120, 250, 230, "进程状态", ("进程是否存活",), header=HEADER),
        Box(410, 120, 250, 230, "训练日志", ("最近阶段信息",), header=HEADER),
        Box(740, 120, 250, 230, "指标文件", ("训练与测试曲线",), header=HEADER),
        Box(1070, 120, 250, 230, "停止文件", ("停止令牌与结果",), header=HEADER_WARN),
        Box(1400, 120, 250, 230, "检查点目录", ("最新与最终权重",), header="#eef7f3"),
    ]
    infer = Box(
        610,
        525,
        580,
        250,
        "状态推断器",
        ("启动中 / 运行中 / 评估中", "停止中 / 已完成", "已安全停止 / 外部终止"),
        header="#f1f5f9",
    )
    view = Box(1300, 575, 360, 110, "前端任务详情与曲线展示", (), header="#eef7f3")
    for box in (*sources, infer, view):
        draw_card(draw, box, compact=box.h <= 240)
    infer_targets = [650, 775, 900, 1025, 1150]
    for box, target_x in zip(sources, infer_targets):
        draw_arrow(draw, (box.x + box.w // 2, box.bottom + 2), (target_x, infer.y - 2), color=BLUE, width=3)
    draw_arrow(draw, (infer.right + 2, infer.y + infer.h // 2), (view.x - 2, view.y + view.h // 2), color=GREEN)
    save(image, output_dir, "auto_training_fig4.png")


def auto_training_fig5(output_dir: Path) -> None:
    image, draw = make_canvas("图 5 检查点保存、清理、删除和恢复训练关系图")
    loop = Box(70, 110, 360, 660, "训练循环", ("轮次检查点", "最新检查点", "最终检查点"), header=HEADER)
    retain = Box(560, 90, 680, 260, "保留策略", ("保留最近若干轮次", "始终保留最新权重", "始终保留最终权重", "自动清理旧轮次"), header=HEADER_WARN)
    delete = Box(560, 515, 680, 280, "安全删除", ("任务已停止", "路径在运行目录内", "非外部路径", "删除后更新清单"), header="#fff1f2", border="#e5a6ad")
    resume = Box(1390, 120, 330, 610, "恢复校验", ("仿真器", "场景集合", "输入表示", "嵌入模型", "分词器", "预处理摘要"), header="#eef7f3")
    for box in (loop, retain, delete, resume):
        draw_card(draw, box)
    draw_arrow(draw, (loop.right + 2, 325), (retain.x - 2, 220), color=BLUE)
    draw_arrow(draw, (retain.x + retain.w // 2, retain.bottom + 2), (delete.x + delete.w // 2, delete.y - 2), color=AMBER)
    draw_arrow(draw, (retain.right + 2, 220), (resume.x - 2, 325), color=GREEN)
    draw_arrow(draw, (delete.right + 2, 650), (resume.x - 2, 525), color="#b84a5a")
    save(image, output_dir, "auto_training_fig5.png")


RENDERERS = (
    data_synthesis_fig1,
    data_synthesis_fig2,
    data_synthesis_fig3,
    data_synthesis_fig4,
    data_synthesis_fig5,
    auto_training_fig1,
    auto_training_fig2,
    auto_training_fig3,
    auto_training_fig4,
    auto_training_fig5,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render PiERN patent figures.")
    parser.add_argument(
        "--style",
        choices=("polished", "patent"),
        default="polished",
        help="附图样式：polished 为审稿美观版，patent 为黑白线框专利化版本。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录。默认 polished 输出到 docs/patents/figures，patent 输出到 docs/patents/figures_patent。",
    )
    args = parser.parse_args()
    apply_render_style(args.style)
    output_dir = args.output_dir or (DEFAULT_PATENT_OUTPUT_DIR if args.style == "patent" else DEFAULT_OUTPUT_DIR)

    for renderer in RENDERERS:
        renderer(output_dir)
    print(f"Rendered {len(RENDERERS)} {args.style}-style figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
