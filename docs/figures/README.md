# Figures

Each figure has a TikZ source (`.tex`), a compiled PDF, and a 2000-px PNG used by the README and `docs/concepts.md`.

| Figure | Source |
| --- | --- |
| Training-time information flows, layer-wise | `training_time_layer_wise.tex` |
| Training-time information flows, functional blocks | `training_time_functional.tex` |
| Inference-time information flows, layer-wise | `inference_time_layer_wise.tex` |
| Inference-time information flows, functional blocks | `inference_time_functional.tex` |

Rebuild from this directory with [Tectonic](https://tectonic-typesetting.github.io/) (any LaTeX with TikZ works), then render the PNGs:

```sh
for f in *.tex; do tectonic "$f"; done
uv run --no-project --with pymupdf python -c "
import pymupdf, glob
for pdf in glob.glob('*.pdf'):
    page = pymupdf.open(pdf)[0]; zoom = 2000 / page.rect.width
    page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).save(pdf[:-4] + '.png')
"
```
