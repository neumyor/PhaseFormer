"""Build spectrum_traffic.pptx (native XY chart) from spectrum_traffic_data.csv.

    python make_spectrum_traffic.py [data.csv] [out.pptx]
Then: soffice --headless --convert-to pdf spectrum_traffic.pptx ; pdfcrop
"""
import csv, sys, os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.chart.data import XyChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_MARKER_STYLE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION
from pptx.dml.color import RGBColor

here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "spectrum_traffic_data.csv")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "spectrum_traffic.pptx")

rows = list(csv.DictReader(open(src)))
xs = [int(r["phase_length"]) for r in rows]
ys = [float(r["amplitude"]) for r in rows]
peaks = sorted([r for r in rows if int(r["rank"]) > 0], key=lambda r: int(r["rank"]))

FONT = "Arial"
BLUE = RGBColor(0x1F, 0x5F, 0xA8)
RED = RGBColor(0xD6, 0x27, 0x28)
GREY = RGBColor(0x40, 0x40, 0x40)
GRID = RGBColor(0xDD, 0xDD, 0xDD)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(3.62), Inches(1.75)
slide = prs.slides.add_slide(prs.slide_layouts[6])

cd = XyChartData()
s1 = cd.add_series("Spectrum")
for x, y in zip(xs, ys):
    s1.add_data_point(x, y)
s2 = cd.add_series("Top-5 peaks")
for r in peaks:
    s2.add_data_point(int(r["phase_length"]), float(r["amplitude"]))

gf = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, 0, 0,
                            prs.slide_width, prs.slide_height, cd)
chart = gf.chart
chart.has_legend = False
chart.has_title = False
chart.font.name = FONT
chart.font.size = Pt(7)
chart.font.color.rgb = GREY

# series 1: thin blue line
p1 = chart.plots[0].series[0]
p1.smooth = False
p1.format.line.color.rgb = BLUE
p1.format.line.width = Pt(0.9)
p1.marker.style = XL_MARKER_STYLE.NONE
# series 2: markers only, labelled with the phase length
p2 = chart.plots[0].series[1]
p2.smooth = False
p2.format.line.fill.background()
p2.marker.style = XL_MARKER_STYLE.CIRCLE
p2.marker.size = 4
p2.marker.format.fill.solid()
p2.marker.format.fill.fore_color.rgb = RED
p2.marker.format.line.color.rgb = RED
for i, r in enumerate(peaks):
    dl = p2.points[i].data_label
    tf = dl.text_frame
    tf.text = r["phase_length"]
    run = tf.paragraphs[0].runs[0]
    run.font.size = Pt(7.5)
    run.font.bold = True
    run.font.name = FONT
    run.font.color.rgb = RED
    dl.position = XL_LABEL_POSITION.ABOVE

def style_axis(ax, title, lo, hi, major, grid):
    ax.minimum_scale, ax.maximum_scale, ax.major_unit = lo, hi, major
    ax.has_title = True
    ax.axis_title.text_frame.text = title
    r = ax.axis_title.text_frame.paragraphs[0].runs[0]
    r.font.size, r.font.name, r.font.bold = Pt(8), FONT, False
    r.font.color.rgb = GREY
    ax.tick_labels.font.size = Pt(7)
    ax.tick_labels.font.name = FONT
    ax.tick_labels.font.color.rgb = GREY
    ax.format.line.color.rgb = GREY
    ax.format.line.width = Pt(0.5)
    ax.has_major_gridlines = grid
    if grid:
        ax.major_gridlines.format.line.color.rgb = GRID
        ax.major_gridlines.format.line.width = Pt(0.4)
    ax.has_minor_gridlines = False

style_axis(chart.category_axis, "Phase length", 0, 50, 10, False)
style_axis(chart.value_axis, "Amplitude", 0, 450, 150, True)
chart.value_axis.tick_labels.number_format = "0"
chart.value_axis.tick_labels.number_format_is_linked = False
chart.category_axis.tick_labels.number_format = "0"
chart.category_axis.tick_labels.number_format_is_linked = False

prs.save(dst)
print("saved", dst)
