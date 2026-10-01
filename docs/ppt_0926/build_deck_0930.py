"""Build the PhaseFormer-L leadership deck (minipaper 0930) on the group template."""
import copy
import math
import sys

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

EA_FONT = sys.argv[2] if len(sys.argv) > 2 else "黑体"   # QA render swaps in "Heiti SC"
OUT = sys.argv[1] if len(sys.argv) > 1 else "PhaseFormer-L_汇报_0930.pptx"
LATIN = "Times New Roman"

DARK = "003DA6"      # template deep blue
LIGHT = "009FE3"     # template bright blue
BLUE = "0000FF"      # pure-blue accent
RED = "FF0000"       # pure-red accent
TEXT = "1A1A1A"
GRAY = "6B6B6B"
TINT = "EDF3FA"
BORDER = "A9CBEA"
WHITE = "FFFFFF"

prs = Presentation("template.pptx")
tmpl = list(prs.slides)
S_COVER, S_SECTION, S_OVERVIEW, S_FIGURE, S_END = tmpl


def rgb(h):
    return RGBColor.from_string(h)


def set_font(run, size, color=TEXT, bold=False, italic=False):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = rgb(color)
    f.name = LATIN
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        for old in rPr.findall(qn(tag)):
            rPr.remove(old)
    latin = rPr.find(qn("a:latin"))
    ea = etree.SubElement(rPr, qn("a:ea"))
    ea.set("typeface", EA_FONT)
    cs = etree.SubElement(rPr, qn("a:cs"))
    cs.set("typeface", EA_FONT)
    latin.addnext(ea)
    ea.addnext(cs)


def fix_ea(el):
    """Force every East-Asian font slot under el to EA_FONT."""
    for tag in ("a:ea", "a:cs"):
        for node in el.iter(qn(tag)):
            node.set("typeface", EA_FONT)


def tb(slide, x, y, w, h, paras, size=16, color=TEXT, bold=False, align=PP_ALIGN.LEFT,
       anchor=MSO_ANCHOR.TOP, space_after=0, line=1.1):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line
        p.space_after = Pt(space_after)
        runs = [(para, {})] if isinstance(para, str) else para
        for item in runs:
            text, opt = (item, {}) if isinstance(item, str) else item
            r = p.add_run()
            r.text = text
            set_font(r, opt.get("size", size), opt.get("color", color),
                     opt.get("bold", bold), opt.get("italic", False))
            if "baseline" in opt:
                r._r.get_or_add_rPr().set("baseline", str(opt["baseline"]))
    return box


def card(slide, x, y, w, h, fill=TINT, border=BORDER, radius=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if border:
        s.line.color.rgb = rgb(border)
        s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def badge(slide, x, y, d, text, fill=DARK, size=16):
    c = card(slide, x, y, d, d, fill=fill, border=None, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    set_font(r, size, WHITE, bold=True)
    return c


def arrow(slide, x1, y1, x2, y2, color=DARK, width=1.75):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def set_shape_text(shape, text):
    """Replace text of a template shape, keeping its run formatting."""
    p = shape.text_frame.paragraphs[0]
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    fix_ea(shape._element)


def shape_by_name(slide, name):
    return next(s for s in slide.shapes if s.name == name)


HEADER = ["页标题", "页码", "深蓝横线", "亮蓝横线"]
page_no = [3]


def content_slide(title):
    s = prs.slides.add_slide(S_OVERVIEW.slide_layout)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    bg = S_OVERVIEW._element.find(qn("p:cSld")).find(qn("p:bg"))
    if bg is not None:
        s._element.find(qn("p:cSld")).insert(0, copy.deepcopy(bg))
    for name in HEADER:
        s.shapes._spTree.append(copy.deepcopy(shape_by_name(S_OVERVIEW, name)._element))
    set_shape_text(shape_by_name(s, "页标题"), title)
    page_no[0] += 1
    set_shape_text(shape_by_name(s, "页码"), str(page_no[0]))
    return s


def style_chart(chart, size=12):
    chart.has_title = False
    chart.font.name = LATIN
    chart.font.size = Pt(size)
    chart.font.color.rgb = rgb(TEXT)
    for d in chart._chartSpace.iter(qn("a:defRPr")):
        for tag in ("a:ea", "a:cs"):
            for old in d.findall(qn(tag)):
                d.remove(old)
        latin = d.find(qn("a:latin"))
        if latin is None:
            latin = etree.SubElement(d, qn("a:latin"))
            latin.set("typeface", LATIN)
        ea = etree.Element(qn("a:ea"))
        ea.set("typeface", EA_FONT)
        cs = etree.Element(qn("a:cs"))
        cs.set("typeface", EA_FONT)
        latin.addnext(ea)
        ea.addnext(cs)


# ---------------------------------------------------------------- 1. cover
set_shape_text(shape_by_name(S_COVER, "封面标题"), "PhaseFormer-L：条件水平自适应")
title_box = shape_by_name(S_COVER, "封面标题")
title_box.top = Inches(0.62)
tb(S_COVER, 1.25, 1.52, 10.83, 0.45, "相位建模之外的低维互补修正 —— 从理论到落地的完整证据链",
   size=20, color=DARK, align=PP_ALIGN.CENTER)
set_shape_text(shape_by_name(S_COVER, "汇报人"), "汇报人：Yiming Niu")
set_shape_text(shape_by_name(S_COVER, "汇报日期"), "2026 / 09 / 30")

# ---------------------------------------------------------------- 2. section / agenda
set_shape_text(shape_by_name(S_SECTION, "页标题"), "汇报内容")
t = shape_by_name(S_SECTION, "论文题目")
set_shape_text(t, "PhaseFormer-L：相位分词之外的条件水平自适应")
t.top = Inches(1.75)
for r in t.text_frame.paragraphs[0].runs:
    r.font.size = Pt(30)
info = shape_by_name(S_SECTION, "文献信息")
set_shape_text(info, "PhaseFormer（ICLR 2026）期刊扩展 · Minipaper 0930")
info.top = Inches(2.75)
agenda = [("01", "研究问题"), ("02", "理论依据"), ("03", "方法设计"), ("04", "实验证据"), ("05", "价值总结")]
aw, gap = 2.05, 0.35
ax0 = (13.333 - (5 * aw + 4 * gap)) / 2
for i, (num, name) in enumerate(agenda):
    x = ax0 + i * (aw + gap)
    card(S_SECTION, x, 3.85, aw, 1.9)
    badge(S_SECTION, x + (aw - 0.75) / 2, 4.1, 0.75, num, size=18)
    tb(S_SECTION, x, 5.0, aw, 0.45, name, size=18, color=DARK, bold=True, align=PP_ALIGN.CENTER)

# ---------------------------------------------------------------- 3. overview (template slide 3)
set_shape_text(shape_by_name(S_OVERVIEW, "页标题"), "一页看懂")
set_shape_text(shape_by_name(S_OVERVIEW, "核心思想"),
               "相位主干之外仍有一块可预测误差：结构简单、可解释，补上即稳定增益")
set_shape_text(shape_by_name(S_OVERVIEW, "问题内容"),
               "PhaseFormer 擅长刻画周期内的形状，但跨周期的“水平漂移”仍有一部分没被利用")
set_shape_text(shape_by_name(S_OVERVIEW, "方法内容"),
               "三条定理：原模型能建模部分水平但有天花板，缺口是低秩的；据此增加门控线性修正分支")
res = shape_by_name(S_OVERVIEW, "结果内容")
set_shape_text(res, "三个数据集 12 个设定误差全部下降（最高 ")
p = res.text_frame.paragraphs[0]
base = p.runs[0]
for text, color in [("8.73%", RED), ("）；修正分支仅需 ", TEXT), ("3.5%–6.1%", RED), (" 的参数", TEXT)]:
    r = copy.deepcopy(base._r)
    p._p.append(r)
    p.runs[-1].text = text
    p.runs[-1].font.color.rgb = rgb(color)
    if color == RED:
        p.runs[-1].font.bold = True
set_shape_text(shape_by_name(S_OVERVIEW, "论文引用"),
               "参考：Niu et al. PhaseFormer. ICLR 2026. arXiv:2510.04134；本工作：PhaseFormer-L 期刊扩展稿（Minipaper 0930）")

# ---------------------------------------------------------------- 4. research question
s = content_slide("研究问题：相位建模之后还剩什么？")
tb(s, 0.65, 1.3, 12.0, 0.5, "PhaseFormer 抓住了“周期内的形状”，我们关注“跨周期的水平漂移”还有没有可挖的价值",
   size=20, color=TEXT)
cd = CategoryChartData()
n = 120
levels = [0.0, 0.7, 0.35, 1.35, 2.0]
cd.categories = list(range(n))
sig, lev = [], []
for t_ in range(n):
    L = levels[t_ // 24]
    lev.append(L)
    sig.append(L + math.sin(2 * math.pi * t_ / 24) + 0.35 * math.sin(4 * math.pi * t_ / 24 + 0.6))
cd.add_series("观测序列", sig)
cd.add_series("每个周期的水平", lev)
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(0.55), Inches(2.0), Inches(6.6), Inches(4.0), cd)
ch = gf.chart
style_chart(ch)
ch.has_legend = False
ch.category_axis.visible = False
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
for ser, color, w in zip(ch.plots[0].series, [LIGHT, RED], [2.25, 2.5]):
    ser.smooth = False
    ser.format.line.color.rgb = rgb(color)
    ser.format.line.width = Pt(w)
    ser.marker.style = None
    ser.marker.style = XL_MARKER_STYLE.NONE
ch.plots[0].series[1].format.line.dash_style = MSO_LINE_DASH_STYLE.DASH
tb(s, 0.65, 6.15, 6.4, 0.4, [[("蓝线：", {"color": LIGHT, "bold": True}), "周期内形状（相位路径强项）   ",
                             ("红线：", {"color": RED, "bold": True}), "跨周期水平漂移"]], size=14)
cards = [
    ("PhaseFormer 的做法", "把不同周期中同一相位的观测放在一起建模，参数少、预测长，擅长刻画周期内形状。", DARK),
    ("已有能力", "理论与实测均确认：原模型能解释 19%–66% 的水平变化；本工作补的是它够不到的部分。", DARK),
    ("关键问题", "相位路径预测之后，剩下的误差里还有没有可预测的部分？结构是什么？能否低成本补上？", BLUE),
]
for i, (head, body, hc) in enumerate(cards):
    y = 2.0 + i * 1.35
    card(s, 7.55, y, 5.15, 1.15, fill=TINT if i < 2 else "E6ECFF", border=BORDER if i < 2 else BLUE)
    tb(s, 7.8, y + 0.13, 4.7, 0.35, head, size=16, color=hc, bold=True)
    tb(s, 7.8, y + 0.5, 4.75, 0.6, body, size=14)
tb(s, 7.55, 6.1, 5.15, 0.4, [[("目标：", {"bold": True, "color": RED}), "找到并补上这块“剩余价值”"]], size=16)

# ---------------------------------------------------------------- 5. roadmap
s = content_slide("研究路线：五步证据链，环环相扣")
tb(s, 0.65, 1.3, 12.0, 0.5, "每一步回答一个问题，前一步的结论是后一步的出发点", size=20)
steps = [
    ("理论推导", "缺口在哪？", "原模型有水平天花板\n天花板外缺口是低秩的"),
    ("量化测量", "缺口有多大？", "少数方向拿到\n绝大部分收益"),
    ("机制识别", "补的是什么？", "模型学到理论预言的\n“读水平→写平移”"),
    ("干预验证", "真的有用吗？", "删掉它，整体\n预测一致变差"),
    ("工程落地", "能否更省？", "按低维结构压缩\n参数减少 94% 以上"),
]
w_, g_ = 2.2, 0.35
x0 = (13.333 - (5 * w_ + 4 * g_)) / 2
for i, (name, q, ans) in enumerate(steps):
    x = x0 + i * (w_ + g_)
    badge(s, x + (w_ - 0.9) / 2, 2.1, 0.9, str(i + 1), fill=DARK if i != 3 else BLUE, size=22)
    tb(s, x, 3.15, w_, 0.4, name, size=18, color=DARK, bold=True, align=PP_ALIGN.CENTER)
    tb(s, x, 3.6, w_, 0.35, q, size=14, color=GRAY, align=PP_ALIGN.CENTER)
    card(s, x, 4.05, w_, 1.15)
    tb(s, x + 0.1, 4.05, w_ - 0.2, 1.15, ans.split("\n"), size=14, align=PP_ALIGN.CENTER,
       anchor=MSO_ANCHOR.MIDDLE)
    if i < 4:
        arrow(s, x + (w_ + 0.9) / 2 + 0.08, 2.55, x + w_ + g_ + (w_ - 0.9) / 2 - 0.08, 2.55, color=LIGHT)
card(s, 0.65, 5.65, 12.05, 0.8, fill="E6ECFF", border=BLUE)
tb(s, 0.9, 5.65, 11.6, 0.8, [[("效果锚点：", {"bold": True, "color": BLUE}),
                             "PhaseFormer-L 在三个数据集全部 12 个设定上误差双指标下降，最高降低 ",
                             ("8.73%", {"bold": True, "color": RED}), ""]],
   size=16, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 6. theory
s = content_slide("理论：能建模、有天花板、缺口低秩")
tb(s, 0.65, 1.3, 12.0, 0.5, "三条定理回答三个问题：原模型能做什么、做不到什么、做不到的部分怎么补", size=20)
SUB = {"baseline": -25000}
SUP = {"baseline": 30000}
cols = [
    ("定理 1　能建模部分水平",
     [("F", {}), ("(x + c", {}), ("1", {"bold": True}), (")  =  F", {}), ("(x) + c", {}), ("1", {"bold": True})],
     ["整体的水平平移能无损地穿过注意力，传到每个相位",
      "原模型能实现“近期加权水平”这类最优水平预测"]),
    ("定理 2　有结构性天花板",
     [("|m̂ − μ|  <  C", {}), ("φ", SUB), (" · σ", {})],
     ["最后一层归一化限制了预测水平：离历史均值最多 Cφ 个标准差",
      "需要越过天花板的样本，这部分误差原模型无法消除"]),
    ("定理 3　缺口是低秩的",
     [("rank(W", {}), ("*", SUP), (")  ≤  m", {})],
     ["天花板外的缺口集中在少数固定形状",
      "最优线性修正的秩不超过形状数，读取的正是“近期水平”"]),
]
cw, cg = 3.85, 0.25
for i, (head, formula, body) in enumerate(cols):
    x = 0.65 + i * (cw + cg)
    card(s, x, 2.0, cw, 3.4, fill=TINT if i != 1 else "E6ECFF", border=BORDER if i != 1 else BLUE)
    tb(s, x + 0.25, 2.2, cw - 0.5, 0.4, head, size=18, color=DARK if i != 1 else BLUE, bold=True)
    card(s, x + 0.25, 2.75, cw - 0.5, 0.7, fill=WHITE, border=BORDER)
    runs = [(t_, dict(o, italic=True, color=BLUE)) for t_, o in formula]
    tb(s, x + 0.25, 2.75, cw - 0.5, 0.7, [runs], size=20,
       align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, x + 0.25, 3.65, cw - 0.5, 1.7, body, size=14, space_after=8)
card(s, 0.65, 5.65, 12.05, 0.8, fill="E6ECFF", border=BLUE)
tb(s, 0.9, 5.65, 11.6, 0.8, [[("推论：", {"bold": True, "color": BLUE}),
                             "修正器不需要很大，只需补天花板之外的部分，主导方向是 ",
                             ("“近期水平 → 整体平移”", {"bold": True, "color": RED})]],
   size=16, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 6b. measured envelope
s = content_slide("实测：天花板真实存在，预测越长越紧")
tb(s, 0.65, 1.3, 12.0, 0.5, "直接从训练好的模型权重中读出天花板，再逐个样本检验理论", size=20)
stats = [
    ("168 / 168", "次评估中，模型预测从未越过理论天花板", "定理在真实模型上严格成立", DARK),
    ("19%–66%", "的水平变化由原模型自己解释", "原模型确实能建模部分水平", DARK),
    ("22%", "ETTh2 预测 720 步：原模型误差中结构上无法消除的部分", "天花板带来实打实的损失", RED),
]
for i, (big, what, meaning, c_) in enumerate(stats):
    y = 2.0 + i * 1.3
    card(s, 0.65, y, 5.3, 1.15)
    tb(s, 0.85, y, 1.95, 1.15, big, size=26, color=c_, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, 2.85, y + 0.1, 2.95, 0.4, meaning, size=15, color=DARK, bold=True)
    tb(s, 2.85, y + 0.5, 2.95, 0.6, what, size=12, color=GRAY)
tb(s, 6.35, 1.95, 6.4, 0.4, "需要越过天花板的样本占比（验证集），%", size=15, color=DARK, bold=True)
cd = CategoryChartData()
cd.categories = ["预测 96 步", "192 步", "336 步", "720 步"]
cd.add_series("ETTh2", [3.0, 6.2, 11.1, 48.2])
cd.add_series("ETTm2", [0.7, 1.0, 4.3, 14.4])
cd.add_series("Weather", [2.7, 4.8, 6.4, 11.0])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.25), Inches(2.3), Inches(6.55), Inches(3.55), cd)
ch = gf.chart
style_chart(ch)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.TOP
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(12)
pl = ch.plots[0]
pl.gap_width = 60
pl.overlap = -5
pl.has_data_labels = True
pl.data_labels.number_format = '0.0'
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(12)
for ser, c_ in zip(pl.series, [DARK, LIGHT, "8CC8EC"]):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = rgb(c_)
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.value_axis.maximum_scale = 55
ch.value_axis.minimum_scale = 0
ch.category_axis.tick_labels.font.size = Pt(12)
ch.category_axis.format.line.color.rgb = rgb(BORDER)
style_chart(ch)
card(s, 0.65, 6.0, 12.05, 0.55, fill="E6ECFF", border=BLUE)
tb(s, 0.9, 6.0, 11.6, 0.55, [[("结论：", {"bold": True, "color": BLUE}),
                             "预测越长，原模型“够不到”的样本越多——这正是新增修正分支要补的地方"]],
   size=16, anchor=MSO_ANCHOR.MIDDLE)
tb(s, 0.65, 6.62, 12.1, 0.35, "测量对象：主对比实验中训练好的 84 对模型（7 个数据集 × 4 个预测长度 × 3 次训练），验证集与测试集各评估一次",
   size=12, color=GRAY)

# ---------------------------------------------------------------- 7. method
s = content_slide("方法：相位主干 + 门控线性修正")
tb(s, 0.65, 1.3, 12.0, 0.5, "结构改动很小：在原模型旁并联一条线性分支，由门控自动决定用多少", size=20)


def node(x, y, w, h, lines, fill=TINT, border=BORDER, color=TEXT, size=15, bold=False):
    card(s, x, y, w, h, fill=fill, border=border)
    tb(s, x + 0.08, y, w - 0.16, h, lines, size=size, color=color, bold=bold,
       align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


node(0.65, 3.35, 1.4, 1.0, ["历史序列", "L = 720"], size=15)
node(2.5, 2.1, 2.4, 1.0, ["相位主干", "PhaseFormer"], fill=DARK, border=None, color=WHITE, bold=True)
node(2.5, 4.6, 2.4, 1.0, ["锚定线性修正", "分支（新增）"], fill=BLUE, border=None, color=WHITE, bold=True)
node(5.4, 2.1, 1.25, 1.0, "相位预测", size=15)
node(5.4, 4.6, 1.25, 1.0, "修正预测", size=15)
node(7.1, 3.35, 1.45, 1.0, ["门控融合", "g ∈ (0, 1)"], fill="E6ECFF", border=BLUE, size=15)
arrow(s, 2.05, 3.7, 2.5, 2.6)
arrow(s, 2.05, 4.0, 2.5, 5.1)
arrow(s, 4.9, 2.6, 5.4, 2.6)
arrow(s, 4.9, 5.1, 5.4, 5.1)
arrow(s, 6.65, 2.6, 7.1, 3.65)
arrow(s, 6.65, 5.1, 7.1, 4.05)
tb(s, 0.65, 5.95, 8.0, 0.45, [[("融合： ", {"bold": True, "color": DARK}),
                             ("ŷ = (1 − g) · ŷ", {"italic": True}), ("φ", {"italic": True, "baseline": -25000}),
                             ("  +  g · ŷ", {"italic": True}), ("r", {"italic": True, "baseline": -25000}),
                             ("　三部分联合训练", {})]], size=16)
points = [
    ("锚定", "以最后一个观测为基准，只学习“相对变化”"),
    ("门控", "自动学习修正信号的使用比例，不需要时影响很小"),
    ("联合训练", "只补相位主干没做到的部分，与主干形成互补"),
]
for i, (h_, b_) in enumerate(points):
    y = 2.0 + i * 1.4
    card(s, 9.0, y, 3.7, 1.25)
    badge(s, 9.2, y + 0.18, 0.55, str(i + 1), size=16)
    tb(s, 9.95, y + 0.17, 2.6, 0.4, h_, size=17, color=DARK, bold=True)
    tb(s, 9.95, y + 0.57, 2.6, 0.65, b_, size=13)

# ---------------------------------------------------------------- 8. main result
s = content_slide("效果：12 个设定全部提升，最高 8.73%")
tb(s, 0.65, 1.3, 12.0, 0.5, "在 ETTh2、ETTm2、Weather 三个数据集的全部预测长度上，两项误差指标同时改善", size=20)
stats = [("12 / 12", "设定上 MSE、MAE 双指标全部下降"), ("8.73%", "最大降幅（ETTm2，预测 96 步）"),
         ("3 次", "独立训练取平均，结果稳定可复现")]
for i, (big, small) in enumerate(stats):
    y = 2.0 + i * 1.45
    card(s, 0.65, y, 3.6, 1.25)
    tb(s, 0.85, y + 0.1, 3.2, 0.65, big, size=36, color=RED if i == 1 else DARK, bold=True)
    tb(s, 0.85, y + 0.78, 3.3, 0.4, small, size=14)
cd = CategoryChartData()
labels = ["ETTh2-96", "ETTh2-192", "ETTh2-336", "ETTh2-720", "ETTm2-96", "ETTm2-192", "ETTm2-336",
          "ETTm2-720", "Weather-96", "Weather-192", "Weather-336", "Weather-720"]
vals = [3.06, 1.37, 1.50, 5.68, 8.73, 5.88, 3.12, 1.26, 2.42, 1.55, 1.83, 0.52]
cd.categories = labels
cd.add_series("MSE 降幅 (%)", vals)
tb(s, 4.6, 1.95, 8.1, 0.4, "相对原始 PhaseFormer（仅相位路径）的预测误差（MSE）降低幅度，%", size=15,
   color=DARK, bold=True)
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(4.5), Inches(2.35), Inches(8.3), Inches(3.95), cd)
ch = gf.chart
style_chart(ch)
ch.has_legend = False
pl = ch.plots[0]
pl.gap_width = 55
pl.has_data_labels = True
dl = pl.data_labels
dl.number_format = '0.00'
dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(12)
dl.font.name = LATIN
ser = pl.series[0]
ser.format.fill.solid()
ser.format.fill.fore_color.rgb = rgb(DARK)
for idx in range(12):
    pt = ser.points[idx]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = rgb(RED if idx == 4 else (DARK if idx < 4 else (LIGHT if idx < 8 else "5B7FC7")))
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.value_axis.maximum_scale = 10
ch.value_axis.minimum_scale = 0
ch.category_axis.tick_labels.font.size = Pt(12)
ch.category_axis.format.line.color.rgb = rgb(BORDER)
style_chart(ch)
tb(s, 0.65, 6.45, 12.1, 0.4, "对比设置：同配置、同随机种子下的原始 PhaseFormer；输入长度 720，预测长度 96 / 192 / 336 / 720",
   size=12, color=GRAY)

# ---------------------------------------------------------------- 8b. where the gain comes from
s = content_slide("机理：增益来自主干够不到的水平")
tb(s, 0.65, 1.3, 12.0, 0.5, "新增分支补在了理论指出的位置：天花板之外的样本收益最大，水平也交给了分支", size=20)
tb(s, 0.65, 1.95, 6.6, 0.4, "ETTh2 每个样本的平均误差下降（验证集，×10⁻²）", size=15, color=DARK, bold=True)
cd = CategoryChartData()
cd.categories = ["预测 96 步", "192 步", "336 步", "720 步"]
cd.add_series("天花板之内的样本", [0.59, -0.05, -0.38, 0.70])
cd.add_series("天花板之外的样本", [23.18, 14.03, 9.37, 6.58])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.55), Inches(2.3), Inches(6.7), Inches(3.2), cd)
ch = gf.chart
style_chart(ch)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.TOP
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(12)
pl = ch.plots[0]
pl.gap_width = 60
pl.overlap = -5
pl.has_data_labels = True
pl.data_labels.number_format = '0.0;-0.0;0.0'
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(12)
for ser, c_ in zip(pl.series, ["8CC8EC", RED]):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = rgb(c_)
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.value_axis.maximum_scale = 27
ch.value_axis.minimum_scale = -1
ch.category_axis.tick_labels.font.size = Pt(12)
ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
ch.category_axis.format.line.color.rgb = rgb(BORDER)
style_chart(ch)
tb(s, 0.65, 5.58, 6.6, 0.4, [[("天花板外的样本，单个收益是其他样本的 ", {}), ("4 倍以上", {"bold": True, "color": RED}),
                             ("；24 / 24 次评估一致", {})]], size=13)
rows = [("55%–99%", "融合预测中的水平由新增分支承担", "分支接管了水平", BLUE),
        ("1.3–4.8 倍", "联合训练后，主干自身的水平天花板收窄", "主干让出水平，腾出能力刻画形状", DARK)]
for i, (big, what, meaning, c_) in enumerate(rows):
    y = 2.0 + i * 1.75
    card(s, 7.6, y, 5.1, 1.55)
    tb(s, 7.85, y + 0.12, 4.6, 0.6, big, size=30, color=c_, bold=True)
    tb(s, 7.85, y + 0.72, 4.6, 0.38, meaning, size=16, color=DARK, bold=True)
    tb(s, 7.85, y + 1.1, 4.6, 0.38, what, size=12, color=GRAY)
tb(s, 7.6, 5.55, 5.1, 0.5, "统计范围：ETTh2、ETTm2、Weather 全部 12 个设定", size=12, color=GRAY)
card(s, 0.65, 6.05, 12.05, 0.6, fill="E6ECFF", border=BLUE)
tb(s, 0.9, 6.05, 11.6, 0.6, [[("一句话：", {"bold": True, "color": BLUE}),
                             "理论说原模型在哪里受限 → 实测新增分支恰好补在那里，并与主干形成分工"]],
   size=16, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 9. spectrum
s = content_slide("量化：可预测收益集中在一个方向")
tb(s, 0.65, 1.3, 12.0, 0.5, "在 28 个数据集-预测长度组合上测量“一个线性修正器最多能拿到多少收益、分布在几个方向上”",
   size=18)
stats = [
    ("64%–86%", "第一个方向独占的可达收益", "一个方向就拿到大头"),
    ("2–7 个", "拿到 90% 收益所需的方向数", "缺口确实是低维的"),
    ("0.89–0.99", "与“整体平移”的相似度（满分 1）", "主要作用就是整体平移"),
]
for i, (big, what, meaning) in enumerate(stats):
    x = 0.65 + i * 4.1
    card(s, x, 2.0, 3.8, 2.75)
    tb(s, x + 0.25, 2.2, 3.3, 0.8, big, size=40, color=RED if i == 0 else DARK, bold=True,
       align=PP_ALIGN.CENTER)
    tb(s, x + 0.25, 3.1, 3.3, 0.75, what, size=14, color=GRAY, align=PP_ALIGN.CENTER)
    tb(s, x + 0.25, 4.0, 3.3, 0.5, meaning, size=18, color=DARK, bold=True, align=PP_ALIGN.CENTER)
card(s, 0.65, 5.1, 12.05, 1.25, fill="E6ECFF", border=BLUE)
badge(s, 0.95, 5.35, 0.75, "✓", fill=BLUE, size=20)
tb(s, 1.95, 5.1, 10.5, 1.25, [[("与理论一致，且在相位主干之后依然成立：", {"bold": True, "color": BLUE})],
                             [("扣除相位主干的预测后重新测量，主方向与原结果的一致度 ≥ ", {}),
                              ("0.999", {"bold": True, "color": RED}), ("，指向同一个“近期水平”方向", {})]],
   size=16, anchor=MSO_ANCHOR.MIDDLE, space_after=4)

# ---------------------------------------------------------------- 10. mechanism
s = content_slide("机制：模型学到“读近期水平→写平移”")
tb(s, 0.65, 1.3, 12.0, 0.5, "拆开训练好的修正分支，看它“读什么、写什么”——与理论预言逐项吻合", size=20)
rows = [("18 / 18", "组模型", "读取的都是“近期加权水平”", DARK),
        ("12 / 18", "组（ETT 数据）", "把它写成整段预测的“整体平移”", DARK),
        ("6 / 18", "组（Weather）", "写成平滑的“倾斜 / 弯曲”修正", LIGHT)]
for i, (big, unit, desc, c_) in enumerate(rows):
    y = 2.0 + i * 1.2
    card(s, 0.65, y, 6.0, 1.0)
    tb(s, 0.85, y, 1.85, 1.0, big, size=28, color=c_, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, 2.75, y + 0.14, 3.8, 0.35, unit, size=13, color=GRAY)
    tb(s, 2.75, y + 0.48, 3.8, 0.45, desc, size=16)
tb(s, 0.65, 5.7, 6.0, 0.8, [[("结论：", {"bold": True, "color": BLUE}),
                            "两类数据读的是同一个量，只是“写法”随数据特点不同——正是理论中 a、b 两个因子的体现"]],
   size=14)
tb(s, 7.1, 1.95, 5.6, 0.4, "典型模型（ETTh2-720）各机制的收益占比，%", size=15, color=DARK, bold=True)
cd = CategoryChartData()
cd.categories = ["读水平→写平移", "周期形状修正", "高阶细节修正"]
cd.add_series("贡献 (%)", [75.9, 15.0, 4.3])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(7.0), Inches(2.35), Inches(5.8), Inches(3.3), cd)
ch = gf.chart
style_chart(ch)
ch.has_legend = False
pl = ch.plots[0]
pl.gap_width = 70
pl.has_data_labels = True
pl.data_labels.number_format = '0.0'
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(14)
ser = pl.series[0]
for idx, c_ in enumerate([BLUE, LIGHT, "8CC8EC"]):
    pt = ser.points[idx]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = rgb(c_)
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.value_axis.maximum_scale = 90
ch.value_axis.minimum_scale = 0
ch.category_axis.tick_labels.font.size = Pt(13)
ch.category_axis.format.line.color.rgb = rgb(BORDER)
style_chart(ch)
card(s, 7.1, 5.75, 5.6, 0.75, fill="E6ECFF", border=BLUE)
tb(s, 7.3, 5.75, 5.3, 0.75, [["72 个模型中，", ("3–5 个", {"bold": True, "color": RED}),
                             "机制即解释 ", ("≥95%", {"bold": True, "color": RED}), " 的收益"]],
   size=15, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 11. intervention
s = content_slide("验证：删掉这个方向，整体一致变差")
tb(s, 0.65, 1.3, 12.0, 0.5, "固定模型其余部分，只删除理论指出的“近期水平”方向，并用随机方向做对照", size=20)
card(s, 0.65, 2.0, 3.7, 3.0)
tb(s, 0.85, 2.2, 3.3, 0.4, "只看修正分支自身", size=16, color=GRAY, bold=True, align=PP_ALIGN.CENTER)
tb(s, 0.85, 2.7, 3.3, 0.9, "52 / 72", size=40, color=GRAY, bold=True, align=PP_ALIGN.CENTER)
tb(s, 0.85, 3.75, 3.3, 0.9, ["个模型反而“变好”", "单独看，这个方向似乎没用"], size=14, color=GRAY,
   align=PP_ALIGN.CENTER)
arrow(s, 4.45, 3.5, 5.05, 3.5, color=BLUE, width=2.5)
card(s, 5.15, 2.0, 3.7, 3.0, fill="FFF0F0", border=RED)
tb(s, 5.35, 2.2, 3.3, 0.4, "看融合后的整体预测", size=16, color=RED, bold=True, align=PP_ALIGN.CENTER)
tb(s, 5.35, 2.7, 3.3, 0.9, "72 / 72", size=40, color=RED, bold=True, align=PP_ALIGN.CENTER)
tb(s, 5.35, 3.75, 3.3, 0.9, ["个模型全部“变差”", "误差中位数上升 30.43%"], size=14, align=PP_ALIGN.CENTER)
pts = [("对照严格", "以同维度随机方向为对照：18 / 18 组中，删除理论方向的影响均明显超出对照范围"),
       ("反转说明互补", "单看分支是“多余”的，放进整体却“必不可少”——它的价值来自与相位主干的配合"),
       ("结论", "这个方向是整体预测必需的互补部件，而不是巧合出现的相关特征")]
for i, (h_, b_) in enumerate(pts):
    y = 2.0 + i * 1.2
    tb(s, 9.25, y, 3.5, 0.35, h_, size=16, color=BLUE if i == 2 else DARK, bold=True)
    tb(s, 9.25, y + 0.36, 3.5, 0.68, b_, size=13)
card(s, 0.65, 5.4, 12.05, 0.95, fill="E6ECFF", border=BLUE)
tb(s, 0.9, 5.4, 11.6, 0.95, [[("一句话：", {"bold": True, "color": BLUE}),
                             "理论预言的方向 → 模型确实学到 → 删掉就变差，三者闭环，说明机制解释是",
                             ("可验证的", {"bold": True, "color": RED}), "而非事后附会"]],
   size=16, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 12. compression
s = content_slide("落地：参数大减，效果基本不变")
tb(s, 0.65, 1.3, 12.0, 0.5, "既然有用的方向只有少数几个，修正分支就可以大幅“瘦身”", size=20)


def bar_row(y, label, frac_txt, frac, note, note_hl):
    tb(s, 0.65, y, 3.2, 0.5, label, size=16, color=DARK, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    card(s, 3.95, y + 0.05, 5.0, 0.4, fill="E3E8EF", border=None, radius=0.5)
    card(s, 3.95, y + 0.05, max(5.0 * frac, 0.45), 0.4, fill=BLUE, border=None, radius=0.5)
    tb(s, 3.95 + max(5.0 * frac, 0.45) + 0.15, y, 1.9, 0.5, frac_txt, size=16, color=BLUE, bold=True,
       anchor=MSO_ANCHOR.MIDDLE)
    tb(s, 10.75, y, 2.0, 0.5, [[(note_hl, {"bold": True, "color": RED}), note]], size=15,
       anchor=MSO_ANCHOR.MIDDLE)


tb(s, 3.95, 2.05, 5.0, 0.35, "参数占比（相对未压缩版本）", size=14, color=GRAY)
tb(s, 10.75, 2.05, 2.0, 0.35, "效果", size=14, color=GRAY)
bar_row(2.5, "修正分支（未压缩）", "100%", 1.0, "基准", "")
bar_row(3.25, "修正分支（压缩后）", "3.5%–6.1%", 0.061, " 收益保留", "92%–102%")
bar_row(4.25, "整模型（未压缩）", "100%", 1.0, "基准", "")
bar_row(5.0, "整模型（压缩后）", "16.4%–33.0%", 0.33, " 误差差异", "仅 0.72%")
card(s, 0.65, 5.85, 12.05, 0.8, fill="E6ECFF", border=BLUE)
tb(s, 0.9, 5.85, 11.6, 0.8, [[("意义：", {"bold": True, "color": BLUE}),
                             "理论告诉我们“为什么能压”，实测告诉我们“压到多少合适”——精度与成本可按需取舍"]],
   size=16, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 13. summary
s = content_slide("总结：理论 — 机制 — 效果 — 效率，形成闭环")
items = [
    ("理论清晰", "三条定理：原模型能建模部分水平但有天花板，缺口低秩；实测确认天花板存在，分支正好补在天花板之外"),
    ("机制可解释", "训练出的模型确实学到这一机制；3–5 个可命名的机制解释 ≥95% 的收益，且经删除实验验证"),
    ("效果稳定", "ETTh2、ETTm2、Weather 全部 12 个设定误差双指标下降，最高降低 8.73%"),
    ("部署高效", "修正分支压缩到 3.5%–6.1% 的参数，收益基本保留；整模型可降至 16.4%–33.0%"),
]
for i, (h_, b_) in enumerate(items):
    x = 0.65 + (i % 2) * 6.15
    y = 1.4 + (i // 2) * 2.05
    card(s, x, y, 5.9, 1.8)
    badge(s, x + 0.3, y + 0.3, 0.7, str(i + 1), fill=BLUE if i == 2 else DARK, size=18)
    tb(s, x + 1.25, y + 0.3, 4.4, 0.45, h_, size=20, color=DARK, bold=True)
    tb(s, x + 1.25, y + 0.82, 4.45, 0.95, b_, size=14)
card(s, 0.65, 5.65, 12.05, 0.8, fill="E6ECFF", border=BLUE)
tb(s, 0.9, 5.65, 11.6, 0.8, [[("价值：", {"bold": True, "color": BLUE}),
                             "在 PhaseFormer（ICLR 2026）基础上形成一条",
                             ("从理论到落地的完整主线", {"bold": True, "color": RED}),
                             "，支撑期刊扩展版本"]], size=16, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- reorder: drop figure template, end slide last
sldIdLst = prs.slides._sldIdLst
ids = list(sldIdLst)
fig_id = ids[3]
prs.part.drop_rel(fig_id.rId)
sldIdLst.remove(fig_id)
end_id = ids[4]
sldIdLst.remove(end_id)
sldIdLst.append(end_id)

fix_ea(S_END._element)
for sl in prs.slides:
    fix_ea(sl._element)
prs.save(OUT)
print("saved", OUT)
