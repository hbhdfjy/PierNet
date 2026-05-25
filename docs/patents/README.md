# PierNet 专利文件

本目录集中存放 PierNet 项目的正式专利文本、专利化附图和 Word 生成说明。Markdown 文件是源文件，Word 文件是可重复生成的审稿交付件。

## 文件清单

- `DATA_SYNTHESIS_PATENT_DRAFT.md`：数据合成平台正式专利文本草案，包含摘要、权利要求书、说明书和附图说明。
- `AUTO_TRAINING_PATENT_DRAFT.md`：自动训练平台正式专利文本草案，包含摘要、权利要求书、说明书和附图说明。
- `figures_patent/`：两件专利对应的黑白线框专利化附图，共 10 张；这是当前唯一保留并被 Markdown 引用的正式附图目录。

## 生成 Word 文件

默认命令会读取两个正式草案，并把中文命名的专利化 Word 文件输出到 `docs/` 目录。Word 文件是可重复生成的审稿交付件，不作为 Git 源文件长期跟踪：

```bash
python3 scripts/patents/patent_md_to_docx.py --output-suffix '（专利化附图版）'
```

若需要严格套用外部 Word 模板，可指定模板路径：

```bash
python3 scripts/patents/patent_md_to_docx.py --template /path/to/专利模板.docx --output-suffix '（专利化附图版）'
```

也可以通过环境变量指定模板：

```bash
PierNet_PATENT_TEMPLATE=/path/to/专利模板.docx python3 scripts/patents/patent_md_to_docx.py --output-suffix '（专利化附图版）'
```

## 生成附图

专利附图包含大量中文术语，使用确定性绘图脚本生成，避免大模型图片生成造成文字错误。重新生成当前正式附图时运行：

```bash
python3 scripts/patents/render_patent_figures.py --style patent
```

脚本会输出到 `docs/patents/figures_patent/`，随后可重新运行 Word 生成命令，把最新附图嵌入 `.docx` 文件中。

## 后续建议

- 由专利代理人基于正式草案进一步调整权利要求层级和法律表述。
- 提交前确认是否继续采用 `figures_patent/` 黑白线框图，或进一步改成代理机构偏好的纯编号图。
- 在提交前统一确认发明人、申请人、技术交底日期和对外公开风险。
