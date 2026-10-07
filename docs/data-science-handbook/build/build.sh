#!/usr/bin/env bash
# Ghép các chương Markdown thành 1 file và xuất PDF.
# Yêu cầu: pandoc >= 3, Node.js + playwright (Chromium).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT_MD="$ROOT/Data_Science_End_to_End_Handbook.md"
OUT_HTML="$HERE/handbook.html"
OUT_PDF="$ROOT/Data_Science_End_to_End_Handbook.pdf"

# 1) Ghép Markdown (mỗi chương cách nhau 1 dòng trống)
: > "$OUT_MD"
for f in "$ROOT"/chapters/[0-9][0-9]_*.md; do
  cat "$f" >> "$OUT_MD"
  printf '\n\n' >> "$OUT_MD"
done
echo "Markdown written: $OUT_MD"

# 2) Markdown -> HTML (mục lục, công thức MathML, tô màu code)
pandoc "$OUT_MD" \
  --from gfm+tex_math_dollars+pipe_tables \
  --to html5 --standalone --toc --toc-depth=2 --mathml \
  --highlight-style=pygments \
  --metadata title="Data Science End-to-End Handbook" --metadata lang=vi \
  --include-before-body="$HERE/cover.html" \
  --css="$HERE/handbook.css" --embed-resources \
  -o "$OUT_HTML"

# Pandoc hiển thị title mặc định ở đầu trang -> ẩn đi vì đã có trang bìa
sed -i 's|<header id="title-block-header">|<header id="title-block-header" style="display:none">|' "$OUT_HTML"

# 3) HTML -> PDF
NODE_PATH="$(npm root -g)" node "$HERE/render_pdf.cjs" "$OUT_HTML" "$OUT_PDF"
