set -e
S="${PPTX_QA_SKILL_DIR:?Set PPTX_QA_SKILL_DIR to the presentation QA tool directory}"
export FONTCONFIG_FILE=/opt/homebrew/etc/fonts/fonts.conf FONTCONFIG_PATH=/opt/homebrew/etc/fonts
python3 build_deck.py "PhaseFormer-L_汇报_0926.pptx"
python3 - <<'EOF'
import zipfile
zi=zipfile.ZipFile("PhaseFormer-L_汇报_0926.pptx"); zo=zipfile.ZipFile("qa.pptx","w",zipfile.ZIP_DEFLATED)
for it in zi.infolist():
    d=zi.read(it.filename)
    if it.filename.endswith(".xml"): d=d.decode("utf8").replace("黑体","Heiti SC").encode("utf8")
    zo.writestr(it,d)
zo.close()
EOF
python3 "$S/scripts/office/validate.py" "PhaseFormer-L_汇报_0926.pptx" --original template.pptx 2>&1 | tail -4
python3 "$S/scripts/office/soffice.py" --headless --convert-to pdf qa.pptx 2>&1 | grep -v -i fontconfig | tail -1
rm -f slide-*.jpg; pdftoppm -jpeg -scale-to 1000 qa.pdf slide; ls slide-*
