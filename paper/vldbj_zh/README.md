# PhaseFormer+ 中文版

`PhaseFormer+_中文版.pdf` 是论文《PhaseFormer+: Phase, Level, and Amplitude Tokens for Efficient Spatio-Temporal Forecasting》的中文翻译（正文、附录、摘要与图表标题），仅供阅读，英文投稿版在 `../vldbj/`。

- 公式、数值、表格数据、引用编号与英文版逐项一致；图片沿用英文版文件，图内文字保持英文。
- 参考文献保持原文（英文）。
- 源文件：`main.tex` + `body/*.tex`。用 XeLaTeX 编译（需要 macOS 自带的 STSong / Heiti SC 字体）：

```bash
xelatex main && bibtex main && xelatex main && xelatex main
```
