# PhaseFormer-L 汇报材料

本目录保存 0926 与 0930 两版汇报材料、对应预览 PDF、0930 讲稿版及生成脚本。`template.pptx` 是两版生成脚本共同使用的模板。

## 生成汇报

在本目录运行，依赖 Python 3、`python-pptx` 与 `lxml`：

```bash
python3 build_deck.py
python3 build_deck_0930.py
```

脚本第一个参数可指定输出文件，第二个参数可指定中文字体。默认使用黑体；预览可使用 `Heiti SC`。

生成脚本仅生成普通汇报 PPTX。现有预览 PDF 和讲稿 PPTX 作为独立文档保留，运行上述脚本不会更新它们。

## 本地预览检查

`qa.sh` 与 `qa_0930.sh` 需要额外的演示文稿检查工具，其中包含 `scripts/office/validate.py` 和 `scripts/office/soffice.py`，以及 LibreOffice、Poppler。检查工具不随本仓库分发。

将 `PPTX_QA_SKILL_DIR` 设置为该工具目录后，在本目录执行：

```bash
bash qa.sh
bash qa_0930.sh
```

检查脚本沿用 macOS Homebrew 的 Fontconfig 路径，并生成字体替换后的临时 PPTX、PDF 和逐页 JPG。这些临时产物已加入忽略规则；正式命名的预览 PDF 仍保留在版本管理中。
