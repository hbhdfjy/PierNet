# PiERN 专利文件

本目录集中存放 PiERN 项目的专利规划、大纲和正式文本草案。

## 文件清单

- `DATA_SYNTHESIS_OUTLINE.md`：数据合成平台专利技术大纲。
- `DATA_SYNTHESIS_PATENT_DRAFT.md`：数据合成平台正式专利文本草案，包含摘要、权利要求书、说明书和附图说明。
- `AUTO_TRAINING_OUTLINE.md`：自动训练平台专利技术大纲。
- `AUTO_TRAINING_PATENT_DRAFT.md`：自动训练平台正式专利文本草案，包含摘要、权利要求书、说明书和附图说明。
- `figures/`：两件专利对应的中文附图图片文件，共 10 张，由确定性绘图脚本生成以保证图中文字准确。

## 生成 Word 文件

Markdown 文件是可追溯的源文件，Word 文件只作为审稿和提交前交流用的生成物。需要重新生成时，在项目根目录运行：

```bash
python3 scripts/patents/patent_md_to_docx.py
```

该命令默认读取两个正式草案，并将中文命名的 `.docx` 文件输出到 `docs/` 目录。若需要严格套用外部 Word 模板，可指定模板路径：

```bash
python3 scripts/patents/patent_md_to_docx.py --template /path/to/专利模板.docx
```

也可以通过环境变量指定模板：

```bash
PIERN_PATENT_TEMPLATE=/path/to/专利模板.docx python3 scripts/patents/patent_md_to_docx.py
```

## 生成附图

专利附图包含大量中文术语，默认使用确定性绘图脚本生成，避免大模型图片生成造成文字错误。需要重新生成附图时，在项目根目录运行：

```bash
python3 scripts/patents/render_patent_figures.py
```

生成后的 PNG 文件会覆盖 `docs/patents/figures/` 下的同名附图。随后可重新运行 Word 生成命令，把新附图嵌入到 `.docx` 文件中。

## 后续建议

- 由专利代理人基于正式草案进一步调整权利要求层级和法律表述。
- 提交前由代理人确认附图是否需要改成专利局偏好的纯编号线框图。
- 在提交前统一确认发明人、申请人、技术交底日期和对外公开风险。
