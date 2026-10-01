set -e
S="${PPTX_QA_SKILL_DIR:?Set PPTX_QA_SKILL_DIR to the presentation QA tool directory}"
export FONTCONFIG_FILE=/opt/homebrew/etc/fonts/fonts.conf FONTCONFIG_PATH=/opt/homebrew/etc/fonts
python3 build_deck_0930.py "PhaseFormer-L_汇报_0930.pptx"
python3 - <<'EOF'
import zipfile
zi=zipfile.ZipFile("PhaseFormer-L_汇报_0930.pptx"); zo=zipfile.ZipFile("qa0930.pptx","w",zipfile.ZIP_DEFLATED)
for it in zi.infolist():
    d=zi.read(it.filename)
    if it.filename.endswith(".xml"): d=d.decode("utf8").replace("黑体","Heiti SC").encode("utf8")
    zo.writestr(it,d)
zo.close()
EOF
python3 "$S/scripts/office/validate.py" "PhaseFormer-L_汇报_0930.pptx" --original template.pptx 2>&1 | tail -4
python3 "$S/scripts/office/soffice.py" --headless --convert-to pdf qa0930.pptx 2>&1 | grep -v -i fontconfig | tail -1
rm -f s0930-*.jpg; pdftoppm -jpeg -scale-to 1000 qa0930.pdf s0930; ls s0930-*
