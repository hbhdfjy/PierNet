#!/usr/bin/env python3
"""Convert PierNet patent Markdown drafts to Word documents.

The Markdown files remain the source of truth. This script regenerates Word
documents on demand so reviewers can work with docx files without manually
copying content between formats.

The implementation intentionally uses only Python's standard library. When a
Word template is provided, the script reuses its package-level style/settings
parts and replaces only the document body and image relationships. Without a
template, it writes a minimal valid docx package with the same patent document
section order and explicit paragraph formatting.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKDOWN_FILES = (
    REPO_ROOT / "docs/patents/DATA_SYNTHESIS_PATENT_DRAFT.md",
    REPO_ROOT / "docs/patents/AUTO_TRAINING_PATENT_DRAFT.md",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs"
DEFAULT_TEMPLATE_CANDIDATES = (
    REPO_ROOT / "docs/专利模板.docx",
    REPO_ROOT / "docs/patent_template.docx",
)

WORD_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
PACKAGE_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

EMU_PER_INCH = 914_400
MAX_IMAGE_CX = int(6.2 * EMU_PER_INCH)
MAX_IMAGE_CY = int(8.0 * EMU_PER_INCH)


class PatentFormatError(RuntimeError):
    """Raised when a patent Markdown file misses required sections."""


@dataclass(frozen=True)
class Figure:
    caption: str
    path: Path


@dataclass(frozen=True)
class Claim:
    number: int
    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class DescriptionItem:
    kind: str
    text: str


@dataclass(frozen=True)
class PatentDraft:
    title: str
    abstract: tuple[str, ...]
    claims: tuple[Claim, ...]
    description: tuple[DescriptionItem, ...]
    figures: tuple[Figure, ...]


def extract_markdown_section(lines: list[str], start_header: str, end_header: str | None = None) -> list[str]:
    start = None
    end = len(lines)
    for idx, line in enumerate(lines):
        if line.strip() == start_header:
            start = idx + 1
            break
    if start is None:
        return []

    if end_header is not None:
        for idx in range(start, len(lines)):
            if lines[idx].strip() == end_header:
                end = idx
                break
    else:
        for idx in range(start, len(lines)):
            line = lines[idx]
            if line.startswith("## ") and line.strip() != start_header:
                end = idx
                break
    return lines[start:end]


def paragraphs_from_lines(lines: Iterable[str]) -> tuple[str, ...]:
    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            text = "".join(buffer).strip()
            if text:
                paragraphs.append(text)
            buffer.clear()

    for raw in lines:
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("#") or line.startswith("!["):
            flush()
            continue
        if re.match(r"^\d+[.．]\s*", line):
            flush()
            paragraphs.append(line)
            continue
        buffer.append(line)
    flush()
    return tuple(paragraphs)


def parse_claims(lines: Iterable[str]) -> tuple[Claim, ...]:
    claims: list[Claim] = []
    current_number: int | None = None
    current_lines: list[str] = []

    def finish_claim() -> None:
        nonlocal current_number, current_lines
        if current_number is not None:
            claims.append(Claim(current_number, paragraphs_from_lines(current_lines)))
        current_number = None
        current_lines = []

    for raw in lines:
        line = raw.strip()
        match = re.match(r"^###\s*权利要求\s*(\d+)\s*$", line)
        if match:
            finish_claim()
            current_number = int(match.group(1))
            continue
        current_lines.append(raw)
    finish_claim()
    return tuple(claims)


def parse_description(lines: Iterable[str], md_path: Path) -> tuple[tuple[DescriptionItem, ...], tuple[Figure, ...]]:
    items: list[DescriptionItem] = []
    figures: list[Figure] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            text = "".join(buffer).strip()
            if text:
                items.append(DescriptionItem("paragraph", text))
            buffer.clear()

    for raw in lines:
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("### "):
            flush()
            items.append(DescriptionItem("section", line[4:].strip()))
            continue
        if line.startswith("#### "):
            flush()
            items.append(DescriptionItem("subsection", line[5:].strip()))
            continue
        image_match = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", line)
        if image_match:
            flush()
            figures.append(Figure(image_match.group(1).strip(), (md_path.parent / image_match.group(2).strip()).resolve()))
            continue
        if line.startswith("## "):
            flush()
            continue
        if re.match(r"^\d+[.．]\s*", line):
            flush()
            items.append(DescriptionItem("paragraph", line))
            continue
        buffer.append(line)
    flush()
    return tuple(items), tuple(figures)


def parse_patent_markdown(md_path: Path) -> PatentDraft:
    lines = md_path.read_text(encoding="utf-8").splitlines()

    title = paragraphs_from_lines(extract_markdown_section(lines, "## 发明名称"))
    abstract = paragraphs_from_lines(extract_markdown_section(lines, "## 摘要", "## 权利要求书"))
    claims = parse_claims(extract_markdown_section(lines, "## 权利要求书", "## 说明书"))
    description, figures = parse_description(extract_markdown_section(lines, "## 说明书"), md_path)

    missing = []
    if not title:
        missing.append("## 发明名称")
    if not abstract:
        missing.append("## 摘要")
    if not claims:
        missing.append("## 权利要求书")
    if not description:
        missing.append("## 说明书")
    if not figures:
        missing.append("说明书中的附图")
    if missing:
        raise PatentFormatError(f"{md_path} 缺少必要内容: {', '.join(missing)}")
    for figure in figures:
        if not figure.path.exists():
            raise PatentFormatError(f"{md_path} 引用的附图不存在: {figure.path}")

    return PatentDraft(title[0], abstract, claims, description, figures)


def image_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    raise PatentFormatError(f"仅支持 PNG 附图: {path}")


def scaled_image_emu(path: Path) -> tuple[int, int]:
    width_px, height_px = image_dimensions(path)
    cx = int(width_px / 96 * EMU_PER_INCH)
    cy = int(height_px / 96 * EMU_PER_INCH)
    scale = min(MAX_IMAGE_CX / cx, MAX_IMAGE_CY / cy, 1.0)
    return int(cx * scale), int(cy * scale)


def run_properties(size: int, bold: bool = False, font: str = "仿宋") -> str:
    bold_xml = "<w:b/>" if bold else ""
    return (
        f'<w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{font}"/>'
        f'{bold_xml}<w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>'
    )


def paragraph_properties(
    *,
    align: str | None = None,
    first_line: bool = False,
    border: bool = False,
    before: int = 0,
    after: int = 0,
    line: int = 312,
) -> str:
    parts = [f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>']
    if first_line:
        parts.append('<w:ind w:firstLine="560"/>')
    if align:
        parts.append(f'<w:jc w:val="{align}"/>')
    if border:
        parts.append('<w:pBdr><w:bottom w:val="single" w:sz="6" w:space="1" w:color="000000"/></w:pBdr>')
    return "<w:pPr>" + "".join(parts) + "</w:pPr>"


def paragraph(
    text: str = "",
    *,
    kind: str = "body",
    align: str | None = None,
    first_line: bool | None = None,
    bold: bool | None = None,
    border: bool = False,
    before: int = 0,
    after: int = 0,
) -> str:
    styles = {
        "top_title": ("宋体", 36, True, "center", False),
        "document_title": ("黑体", 32, True, "center", False),
        "section": ("黑体", 28, True, None, False),
        "subsection": ("黑体", 28, True, None, False),
        "caption": ("仿宋", 24, False, "center", False),
        "body": ("仿宋", 28, False, None, True),
    }
    font, size, default_bold, default_align, default_first_line = styles.get(kind, styles["body"])
    align = default_align if align is None else align
    first_line = default_first_line if first_line is None else first_line
    bold = default_bold if bold is None else bold
    return (
        f"<w:p>{paragraph_properties(align=align, first_line=first_line, border=border, before=before, after=after)}"
        f'<w:r>{run_properties(size, bold, font)}<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
    )


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def image_paragraph(rel_id: str, cx: int, cy: int, doc_pr_id: int, name: str) -> str:
    safe_name = escape(name)
    return (
        '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{doc_pr_id}" name="{safe_name}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="{safe_name}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
        "</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"
    )


def section_properties() -> str:
    return (
        '<w:sectPr><w:pgSz w:w="11907" w:h="16840"/>'
        '<w:pgMar w:top="1418" w:right="947" w:bottom="851" w:left="1418" '
        'w:header="851" w:footer="851" w:gutter="0"/>'
        '<w:cols w:space="425"/><w:docGrid w:type="lines" w:linePitch="312"/></w:sectPr>'
    )


def image_relationships(figures: Iterable[Figure]) -> tuple[dict[Path, str], str]:
    rel_ids: dict[Path, str] = {}
    relationships: list[str] = []
    for figure in figures:
        if figure.path in rel_ids:
            continue
        rel_id = f"rId{len(rel_ids) + 1}"
        rel_ids[figure.path] = rel_id
        relationships.append(
            f'<Relationship Id="{rel_id}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/{escape(figure.path.name)}"/>'
        )
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{PACKAGE_RELS_NS}">{"".join(relationships)}</Relationships>'
    )
    return rel_ids, xml


def document_xml(draft: PatentDraft, rel_ids: dict[Path, str]) -> str:
    body: list[str] = []

    body.append(paragraph("说   明   书   摘   要", kind="top_title", border=True, after=240))
    for item in draft.abstract:
        body.append(paragraph(item))

    body.append(page_break())
    body.append(paragraph("摘   要   附   图", kind="top_title", border=True, after=240))
    first_figure = draft.figures[0]
    body.append(image_paragraph(rel_ids[first_figure.path], *scaled_image_emu(first_figure.path), 1, first_figure.caption))

    body.append(page_break())
    body.append(paragraph("权   利   要   求   书", kind="top_title", border=True, after=240))
    for claim in draft.claims:
        if not claim.paragraphs:
            continue
        body.append(paragraph(f"{claim.number}．{claim.paragraphs[0]}"))
        for item in claim.paragraphs[1:]:
            body.append(paragraph(item))

    body.append(page_break())
    body.append(paragraph("说      明      书", kind="top_title", border=True, after=240))
    body.append(paragraph(draft.title, kind="document_title", after=120))
    for item in draft.description:
        if item.kind == "section":
            body.append(paragraph(item.text, kind="section", before=120, after=60))
        elif item.kind == "subsection":
            body.append(paragraph(item.text, kind="subsection", before=80, after=40))
        else:
            body.append(paragraph(item.text))

    body.append(page_break())
    body.append(paragraph("说   明   书   附   图", kind="top_title", border=True, after=240))
    for doc_pr_id, figure in enumerate(draft.figures, start=2):
        body.append(paragraph(figure.caption, kind="caption", after=120))
        body.append(image_paragraph(rel_ids[figure.path], *scaled_image_emu(figure.path), doc_pr_id, figure.caption))

    body.append(section_properties())
    namespaces = " ".join(f'xmlns:{prefix}="{uri}"' for prefix, uri in WORD_NS.items())
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {namespaces}><w:body>{"".join(body)}</w:body></w:document>'


def content_types_xml() -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Types xmlns="{CONTENT_TYPES_NS}">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    ).encode("utf-8")


def package_relationships_xml() -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{PACKAGE_RELS_NS}">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    ).encode("utf-8")


def update_template_content_types(xml_bytes: bytes) -> bytes:
    ET.register_namespace("", CONTENT_TYPES_NS)
    root = ET.fromstring(xml_bytes)
    if not any(
        element.tag == f"{{{CONTENT_TYPES_NS}}}Default" and element.attrib.get("Extension") == "png"
        for element in root
    ):
        root.insert(0, ET.Element(f"{{{CONTENT_TYPES_NS}}}Default", {"Extension": "png", "ContentType": "image/png"}))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_docx(draft: PatentDraft, output_path: Path, template_path: Path | None = None) -> None:
    rel_ids, rels_xml = image_relationships(draft.figures)
    document = document_xml(draft, rel_ids).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")

    with ZipFile(temp_output, "w", ZIP_DEFLATED) as docx:
        if template_path is not None:
            with ZipFile(template_path, "r") as template:
                for info in template.infolist():
                    name = info.filename
                    if name in {"word/document.xml", "word/_rels/document.xml.rels"} or name.startswith("word/media/"):
                        continue
                    if name == "[Content_Types].xml":
                        docx.writestr(name, update_template_content_types(template.read(name)))
                    else:
                        docx.writestr(info, template.read(name))
        else:
            docx.writestr("[Content_Types].xml", content_types_xml())
            docx.writestr("_rels/.rels", package_relationships_xml())

        docx.writestr("word/document.xml", document)
        docx.writestr("word/_rels/document.xml.rels", rels_xml.encode("utf-8"))
        for figure in draft.figures:
            docx.write(figure.path, f"word/media/{figure.path.name}")

    temp_output.replace(output_path)


def resolve_template(path_arg: str | None) -> Path | None:
    candidates: list[Path] = []
    if path_arg:
        candidates.append(Path(path_arg).expanduser())
    env_template = os.environ.get("PierNet_PATENT_TEMPLATE")
    if env_template:
        candidates.append(Path(env_template).expanduser())
    candidates.extend(DEFAULT_TEMPLATE_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if path_arg:
        raise FileNotFoundError(f"模板文件不存在: {path_arg}")
    return None


def with_figure_dir(draft: PatentDraft, figure_dir: Path) -> PatentDraft:
    figure_dir = figure_dir.resolve()
    figures = tuple(Figure(figure.caption, figure_dir / figure.path.name) for figure in draft.figures)
    for figure in figures:
        if not figure.path.exists():
            raise PatentFormatError(f"替换后的附图不存在: {figure.path}")
    return replace(draft, figures=figures)


def default_output_name(draft: PatentDraft, suffix: str = "") -> str:
    return f"{draft.title}{suffix}.docx"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 PierNet 专利 Markdown 草案转换为中文命名的 Word 文件。")
    parser.add_argument(
        "markdown",
        nargs="*",
        type=Path,
        help="要转换的专利 Markdown 文件；省略时转换 docs/patents 下两个正式草案。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Word 输出目录，默认是项目 docs 目录。",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="可选 Word 模板路径；也可通过 PierNet_PATENT_TEMPLATE 环境变量指定。",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="可选附图目录；指定后使用该目录下同名 PNG 替换 Markdown 中引用的附图。",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="追加到中文 Word 文件名末尾的后缀，例如：'（专利化附图版）'。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    markdown_files = tuple(path.resolve() for path in args.markdown) if args.markdown else DEFAULT_MARKDOWN_FILES
    template_path = resolve_template(args.template)
    output_dir = args.output_dir.resolve()
    figure_dir = args.figure_dir.resolve() if args.figure_dir else None

    print(f"[patent-docx] output_dir={output_dir}")
    print(f"[patent-docx] template={template_path if template_path else 'internal-minimal-docx'}")
    if figure_dir:
        print(f"[patent-docx] figure_dir={figure_dir}")

    for md_path in markdown_files:
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown 文件不存在: {md_path}")
        draft = parse_patent_markdown(md_path)
        if figure_dir:
            draft = with_figure_dir(draft, figure_dir)
        output_path = output_dir / default_output_name(draft, args.output_suffix)
        write_docx(draft, output_path, template_path)
        print(f"[patent-docx] wrote {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[patent-docx] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
