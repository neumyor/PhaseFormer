"""Render the 3-slide leadership deck for the low-rank read/write finding.

Numbers come from `research_runs/lowrank_checkpoint_information_v1/` (q=1/8,
3 seeds, 6 settings).  Run:  python3 scripts/render_lowrank_semantic_readwrite_slide.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
FAINT = RGBColor(0x99, 0x99, 0x99)
BLUE = RGBColor(0x3A, 0x66, 0xA8)
LIGHT = RGBColor(0xEA, 0xF0, 0xF9)
BORDER = RGBColor(0xC8, 0xD8, 0xEF)
PAPER = RGBColor(0xF4, 0xF5, 0xF7)
GREEN = RGBColor(0x2E, 0x7D, 0x52)
AMBER = RGBColor(0xB5, 0x6A, 0x14)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "PingFang SC"
FOOTER = "残差支路低秩压缩 · 读写成分分析 · 2026-09-17"


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    return frame


def para(frame, runs, *, size=12.5, bold=False, color=INK, space_before=0,
         space_after=4, align=PP_ALIGN.LEFT, line=1.22, first=False):
    p = frame.paragraphs[0] if first else frame.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    p.line_spacing = line
    for item in runs:
        run = p.add_run()
        run.text = item[0]
        run.font.size = Pt(item[1] if len(item) > 1 else size)
        run.font.bold = item[2] if len(item) > 2 else bold
        run.font.color.rgb = item[3] if len(item) > 3 else color
        run.font.name = FONT
    return p


def card(slide, x, y, w, h, *, fill=WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.06, weight=1.25):
    box = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        box.adjustments[0] = radius
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    if line is None:
        box.line.fill.background()
    else:
        box.line.color.rgb = line
        box.line.width = Pt(weight)
    box.shadow.inherit = False
    box.text_frame.word_wrap = True
    return box


def title(slide, text, note):
    frame = textbox(slide, 0.6, 0.40, 12.13, 0.62)
    para(frame, [(text, 26, True, INK)], first=True)
    frame2 = textbox(slide, 0.6, 1.02, 12.13, 0.34)
    para(frame2, [(note, 12, False, MUTED)], first=True)
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.40),
                                  Inches(1.05), Inches(0.045))
    rule.fill.solid()
    rule.fill.fore_color.rgb = BLUE
    rule.line.fill.background()
    rule.shadow.inherit = False


def footer(slide):
    frame = textbox(slide, 0.6, 6.98, 12.13, 0.3)
    para(frame, [(FOOTER, 9.5, False, FAINT)], first=True)


def arrow(slide, x, y, w, h, label):
    box = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = BORDER
    box.line.fill.background()
    box.shadow.inherit = False
    frame = box.text_frame
    frame.word_wrap = False
    para(frame, [(label, 9.5, False, BLUE)], align=PP_ALIGN.CENTER, first=True,
         space_after=0)


# ---------------------------------------------------------------- slide 1

def slide_conclusion(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "低秩支路到底读什么、写什么",
          "一句话结论：能说清的是「主模式」——它读最近一段的水平，写整段未来的整体平移；更细的成分目前还说不清")

    banner = card(slide, 0.6, 1.62, 12.13, 0.78, fill=LIGHT, line=BORDER)
    frame = banner.text_frame
    frame.margin_left = frame.margin_right = Inches(0.22)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(frame, [("结论：", 15, True, BLUE),
                 ("这条支路不重建未来波形，它只做一件事——", 15, False, INK),
                 ("把最近一段时间的水平差异，换算成整段未来该平移多少。", 15, True, INK)],
         first=True, space_after=0)

    stages = [
        (0.60, 2.10, "历史窗口", "720 维", "最近 720 步的完整波形"),
        (3.42, 1.00, "读取", "", "取一个方向"),
        (4.62, 2.10, "低维状态", "1–2 维", "一个标量，代表\n「近期水平」"),
        (7.00, 1.00, "写出", "", "取一个方向"),
        (8.20, 2.10, "未来修正", "96–720 维", "整段加同一个\n常数"),
        (10.60, 2.13, "与主干融合", "gate ≈ 0.21–0.51", "相位主干管波形\n支路管水平校正"),
    ]
    for x, w, head, big, sub in stages:
        if w < 1.5:
            arrow(slide, x, 3.20, w, 0.42, "")
            continue
        box = card(slide, x, 2.10, w, 2.20, fill=WHITE if "融合" not in head else LIGHT)
        frame = box.text_frame
        frame.margin_left = frame.margin_right = Inches(0.14)
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        para(frame, [(head, 12.5, True, BLUE)], align=PP_ALIGN.CENTER, first=True,
             space_after=3)
        if big:
            para(frame, [(big, 19, True, INK)], align=PP_ALIGN.CENTER, space_after=3)
        for line in sub.split("\n"):
            para(frame, [(line, 10.5, False, MUTED)], align=PP_ALIGN.CENTER,
                 space_after=1)

    takeaways = [
        ("读什么", "最近一小段窗口的加权水平，近似一个短窗指数加权平均。"
                   "三个随机种子给出同一个方向。", GREEN),
        ("写什么", "整段未来加同一个常数，也就是整体平移。"
                   "方向与「常数」模板相似度 0.90–0.99。", GREEN),
        ("占多大", "这一件事 1–2 个维度就能完成。主模式承载的修正能量，"
                   "在多数场景占 60%–80%，长周期场景降到约 20%。", BLUE),
    ]
    for i, (head, body, accent) in enumerate(takeaways):
        x = 0.60 + i * 4.14
        box = card(slide, x, 4.62, 3.85, 2.02, fill=WHITE)
        frame = box.text_frame
        frame.margin_left = frame.margin_right = Inches(0.18)
        frame.margin_top = frame.margin_bottom = Inches(0.14)
        para(frame, [(head, 14, True, accent)], first=True, space_after=5)
        para(frame, [(body, 11.5, False, MUTED)], line=1.32)

    footer(slide)
    return slide


# ---------------------------------------------------------------- slide 2

def slide_evidence(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "三条独立证据：方向对得上，而且删掉就变差",
          "先看方向是否对得上（相关性证据），再用干预确认它是否真的被依赖（因果证据）")

    rows = [
        ("证据一 · 读",
         "输入方向确实指向「近期水平」",
         "与近期加权水平模板的相似度 0.67–0.81；与远期、周期类模板明显更低。"
         "输入侧字典解释率 0.57–0.74。",
         "方向证据"),
        ("证据二 · 写",
         "输出方向确实写成「整体平移」",
         "与「常数」模板相似度 0.90–0.99，输出侧字典解释率 0.92–0.97，"
         "跨种子方向一致性 0.98–0.998。",
         "方向证据"),
        ("证据三 · 干预",
         "删掉它会变差，只留它几乎不损失",
         "只保留该子空间：融合误差变化不超过 0.0006（即修正几乎完全复现）。"
         "删掉该子空间：融合误差相对上升 4.3%–16.0%，且 72/72 个实验单元"
         "都超出同维随机删除的 95% 区间。",
         "因果证据"),
    ]
    for i, (tag, head, body, kind) in enumerate(rows):
        y = 1.66 + i * 1.62
        box = card(slide, 0.6, y, 12.13, 1.44, fill=WHITE)
        accent = GREEN if kind == "因果证据" else BLUE
        bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.60),
                                     Inches(y), Inches(0.075), Inches(1.44))
        bar.adjustments[0] = 0.5
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent
        bar.line.fill.background()
        bar.shadow.inherit = False

        frame = textbox(slide, 0.86, y + 0.15, 2.55, 1.14, anchor=MSO_ANCHOR.MIDDLE)
        para(frame, [(tag, 15, True, accent)], first=True, space_after=3)
        para(frame, [(kind, 10, False, FAINT)])

        frame = textbox(slide, 3.62, y + 0.15, 8.85, 1.14, anchor=MSO_ANCHOR.MIDDLE)
        para(frame, [(head, 14, True, INK)], first=True, space_after=5)
        para(frame, [(body, 11.5, False, MUTED)], line=1.30)

    note = card(slide, 0.6, 6.48, 12.13, 0.46, fill=PAPER, line=None)
    frame = note.text_frame
    frame.margin_left = Inches(0.2)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(frame, [("为什么这三条要一起看：", 11, True, INK),
                 ("方向对得上只能说明「长得像」，不能说明模型真的在用；"
                  "只有删掉/只留的干预才能证明它被依赖。前两条与前人观察一致，"
                  "第三条是本轮新增的因果证据。", 11, False, MUTED)],
         first=True, space_after=0)

    footer(slide)
    return slide


# ---------------------------------------------------------------- slide 3

def slide_limits(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "哪些还不能说，以及一个更重要的发现",
          "三条边界决定了对内可以怎么讲；最后一条改变了我们对这条支路定位的理解")

    limits = [
        ("边界一 · 是条件性机制，不是普适规律",
         "6 个数据集里 4 个匹配「近期水平 → 整体位移」（ETTh2-720/96、ETTm2-192/96），"
         "两个 Weather 是反例——它们的首位模式指向曲率/慢趋势，且 H1 在 Weather-192 上"
         "完全不成立。因此只能写成数据集条件性的机制。", AMBER),
        ("边界二 · 只有「主模式」说得清",
         "第 2 个及以后的模式输入解释率从 0.8 以上掉到 0.2–0.7，且跨种子不稳定，"
         "无法给出唯一命名。这就是计划里预留的「多组等价信息通路」情形。", AMBER),
        ("边界三 · 还分不清「语义有效」还是「方向数量有效」",
         "在维度对齐的 57 个单元上，「删除语义方向」与「删除同数量的主成分方向」"
         "的融合误差变化逐单元完全相同（差值恰为 0）；另 15 个单元两者维度不同。"
         "本轮没有随机子空间对照，所以不能排除「任意同数量的主要方向同样有效」。", AMBER),
    ]
    for i, (head, body, accent) in enumerate(limits):
        y = 1.62 + i * 1.32
        box = card(slide, 0.6, y, 12.13, 1.16, fill=WHITE)
        bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.60),
                                     Inches(y), Inches(0.075), Inches(1.16))
        bar.adjustments[0] = 0.5
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent
        bar.line.fill.background()
        bar.shadow.inherit = False
        frame = textbox(slide, 0.88, y + 0.13, 11.6, 0.9, anchor=MSO_ANCHOR.MIDDLE)
        para(frame, [(head, 13.5, True, accent)], first=True, space_after=4)
        para(frame, [(body, 11, False, MUTED)], line=1.30)

    y = 5.62
    box = card(slide, 0.6, y, 12.13, 1.30, fill=LIGHT, line=BORDER)
    frame = box.text_frame
    frame.margin_left = frame.margin_right = Inches(0.24)
    frame.margin_top = frame.margin_bottom = Inches(0.14)
    para(frame, [("更重要的发现：", 13.5, True, BLUE),
                 ("这条支路不是独立预测器，它只在融合里产生价值。", 13.5, True, INK)],
         first=True, space_after=5)
    para(frame, [("删掉它承载的信息后，支路自身的误差反而在 52/72 个单元上变小，"
                  "但融合后的最终误差 72/72 全部变大（中位 +30.4%）。"
                  "也就是说：它单独看甚至不如「不修正」，但与相位主干组合后是净增益。"
                  "这解释了为什么大幅压缩参数仍能保留价值，也说明评估这条支路必须看融合结果，"
                  "不能看支路自己的指标。", 11.5, False, MUTED)], line=1.32)

    footer(slide)
    return slide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else root / "docs" / "PhaseFormer_lowrank_readwrite_slide.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)

    prs = Presentation()
    prs.slide_width = Emu(12192000)
    prs.slide_height = Emu(6858000)
    for build in (slide_conclusion, slide_evidence, slide_limits):
        build(prs)
    prs.save(str(out))
    print(f"wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()