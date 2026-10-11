#!/bin/sh
# Export PowerPoint figures to cropped vector PDFs in ../ and report the
# printed font sizes at the width they are included with in the paper.
#   ./export.sh overview 495        # figure name, included width in pt
#   ./export.sh la_overview 495
set -e
cd "$(dirname "$0")"
name=$1
width=${2:-495}
export FONTCONFIG_FILE=/opt/homebrew/etc/fonts/fonts.conf FONTCONFIG_PATH=/opt/homebrew/etc/fonts
tmp=$(mktemp -d)
soffice --headless --convert-to pdf --outdir "$tmp" "$name.pptx" >/dev/null 2>&1
pdfcrop --margins 1 "$tmp/$name.pdf" "../$name.pdf" >/dev/null
rm -rf "$tmp"
python3 - "../$name.pdf" "$width" <<'EOF'
import sys, collections, pymupdf
doc = pymupdf.open(sys.argv[1]); page = doc[0]
scale = float(sys.argv[2]) / page.rect.width
sizes = collections.Counter()
for b in page.get_text('dict')['blocks']:
    for l in b.get('lines', []):
        for s in l['spans']:
            if s['text'].strip():
                sizes[round(s['size'] * scale, 1)] += len(s['text'].strip())
print(f"{sys.argv[1]}: {page.rect.width:.0f}x{page.rect.height:.0f}pt, printed at {sys.argv[2]}pt "
      f"(scale {scale:.3f}); printed font sizes (pt: chars):",
      ', '.join(f'{k}: {v}' for k, v in sorted(sizes.items())))
print('raster images:', len(page.get_images()))
fonts = sorted({f[3].split('+')[-1] for f in page.get_fonts()})
print('fonts:', ', '.join(fonts))
EOF
