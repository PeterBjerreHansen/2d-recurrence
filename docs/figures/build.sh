#!/usr/bin/env bash
# Rebuild every figure PDF and its 2000-px PNG. Usage: ./build.sh [name.tex ...]
set -euo pipefail
cd "$(dirname "$0")"
sources=("$@")
[ ${#sources[@]} -eq 0 ] && sources=(*.tex)
for source in "${sources[@]}"; do
  [ "$source" = figure_style.tex ] && continue
  tectonic --chatter minimal "$source"
done
uv run --no-project --with pymupdf python - "${sources[@]}" <<'PY'
import sys
import pymupdf
for source in sys.argv[1:]:
    if source == 'figure_style.tex':
        continue
    pdf = source[:-4] + '.pdf'
    page = pymupdf.open(pdf)[0]
    zoom = 2000 / page.rect.width
    page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).save(pdf[:-4] + '.png')
    print('rendered', pdf[:-4] + '.png')
PY
