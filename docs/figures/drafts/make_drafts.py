"""Generate draft layer-wise figures (feedback items 1-5) as TikZ sources.

Run from this directory: ``python3 make_drafts.py`` writes
``training_time_layer_wise_draft.tex`` and ``inference_time_layer_wise_draft.tex``,
which use the shared ``../figure_style.tex``.

Layout A throughout: L1 prelude, L2 T-buffer, L3-L6 core, L7 T-source, L8 coda.
"""
from pathlib import Path

# Shared vertical scale: extra space above L1 and L2 leaves room for the T and D
# injection points.
ROWS = {1: 0.80, 2: 1.55, 3: 2.30, 4: 2.85, 5: 3.40, 6: 3.95, 7: 4.50, 8: 5.05}
T_Y = (ROWS[1] + ROWS[2]) / 2
D_Y = (ROWS[2] + ROWS[3]) / 2
POSITIONS = ['$t{-}1$', '$t$', '$t{+}1$']
PANELS = {'tl': (2.35, 7.35), 'tr': (13.00, 7.35), 'bl': (2.35, -1.05), 'br': (13.00, -1.05)}


def preamble(cell_w):
    return (
        '\\documentclass[tikz,border=7pt]{standalone}\n'
        '\\usepackage{amsmath}\n'
        '\\usetikzlibrary{arrows.meta,calc,positioning,decorations.pathreplacing}\n'
        f'\\def\\figcellw{{{cell_w}}}\\def\\figcellh{{0.32cm}}\n'
        '\\input{../figure_style}\n\n'
        '\\begin{document}\n\\begin{tikzpicture}[x=1cm,y=1cm]\n'
    )


def scaffold(title):
    return f'''
\\node[font=\\bfseries\\LARGE] at (11.9,17.30) {{{title}}};
\\node[header, minimum width=10.35cm, minimum height=0.85cm, font=\\bfseries\\large] at (7.525,15.945) {{Depth recurrence OFF}};
\\node[header, minimum width=10.35cm, minimum height=0.85cm, font=\\bfseries\\large] at (18.175,15.945) {{Depth recurrence ON}};
\\node[sideheader, minimum width=8.05cm, minimum height=0.85cm, rotate=90, font=\\bfseries\\large] at (1.805,11.38) {{Temporal recurrence OFF}};
\\node[sideheader, minimum width=8.05cm, minimum height=0.85cm, rotate=90, font=\\bfseries\\large] at (1.805,2.98) {{Temporal recurrence ON}};
'''


def panel_open(key, title, caption):
    x, y = PANELS[key]
    return (f'\n% ---------------- {title}\n'
            f'\\begin{{scope}}[shift={{({x},{y})}}]\n'
            '  \\draw[panel] (0,0) rectangle (10.35,8.05);\n'
            f'  \\node[font=\\bfseries\\large] at (5.175,7.62) {{{title}}};\n'
            f'  \\node[descbox, text width=9.6cm] at (5.175,6.92) {{{caption}}};\n')


def row_labels(recurrent, x=2.30):
    out = []
    if not recurrent:
        out.append(f'  \\draw[decorate,decoration={{brace,amplitude=4pt}}] ({x},{ROWS[1] - 0.16}) -- ({x},{ROWS[8] + 0.16})'
                   ' node[brace label] {L1--L8\\\\blocks};\n')
        return ''.join(out)
    for k, text in ((1, 'L1 prelude'), (2, 'L2 T-buffer'), (7, 'L7 T-source'), (8, 'L8 coda')):
        out.append(f'  \\node[anchor=east, font=\\footnotesize] at ({x + 0.12},{ROWS[k]}) {{{text}}};\n')
    out.append(f'  \\draw[decorate,decoration={{brace,amplitude=4pt}}] ({x},{ROWS[3] - 0.16}) -- ({x},{ROWS[6] + 0.16})'
               ' node[brace label] {L3--L6\\\\core};\n')
    return ''.join(out)


def node(style, name, x, y, extra=''):
    return f'  \\node[{style}{extra}] ({name}) at ({x:.3f},{y:.3f}) {{}};\n'


def arrow(style, a, b):
    return f'  \\draw[{style}] ({a}.north) -- ({b}.south);\n'


def legend(rows, height=1.45):
    """rows: list of rows, each a list of (kind, style, text) items laid out left to right."""
    out = ['\n% ---------------- Legend\n', '\\begin{scope}[shift={(2.35,-2.85)}]\n',
           f'  \\draw[panel] (0,0) rectangle (21.00,{height});\n']
    for r, items in enumerate(rows):
        y = height - 0.42 - r * 0.55
        x = 0.6
        for kind, style, text in items:
            if kind == 'box':
                out.append(f'  \\node[{style}] at ({x + 0.35:.2f},{y:.2f}) {{}};\n')
                out.append(f'  \\node[legendtext] at ({x + 0.85:.2f},{y:.2f}) {{{text}}};\n')
            elif kind == 'arrow':
                out.append(f'  \\draw[{style}] ({x:.2f},{y:.2f}) -- ({x + 0.75:.2f},{y:.2f});\n')
                out.append(f'  \\node[legendtext] at ({x + 0.85:.2f},{y:.2f}) {{{text}}};\n')
            elif kind == 'inject':
                arrow_style, circle = style
                out.append(f'  \\draw[{arrow_style}] ({x:.2f},{y:.2f}) -- ({x + 0.55:.2f},{y:.2f});\n')
                out.append(f'  \\node[{circle}] at ({x + 0.75:.2f},{y:.2f}) {{{circle[-1]}}};\n')
                out.append(f'  \\node[legendtext] at ({x + 1.0:.2f},{y:.2f}) {{{text}}};\n')
            x += 4.95
    out.append('\\end{scope}\n')
    return ''.join(out)


# ---------------------------------------------------------------------------
# Training time: pass i-1 and pass i side by side for each position

PREV_X = [2.95, 5.20, 7.45]
PAIR = 0.95  # distance between a position's pass i-1 and pass i columns


def training_panel(key, title, caption, depth, temporal):
    s = panel_open(key, title, caption)
    recurrent = depth or temporal
    s += row_labels(recurrent)
    if not recurrent:
        xs = [3.70, 5.60, 7.50]
        s += '  \\node[font=\\bfseries\\footnotesize] at (5.60,5.62) {single pass};\n'
        for c, x in enumerate(xs):
            for k in range(1, 9):
                s += node('active', f'{key}c{c}r{k}', x, ROWS[k])
            for k in range(1, 8):
                s += arrow('flow', f'{key}c{c}r{k}', f'{key}c{c}r{k + 1}')
            s += f'  \\node[font=\\footnotesize] at ({x},0.28) {{{POSITIONS[c]}}};\n'
        return s + '\\end{scope}\n'
    s += '  \\node[anchor=east, font=\\footnotesize, text=black!60] at (2.42,5.62) {pass};\n'
    later = []  # state arrows and injection points drawn last, on top
    for c, px in enumerate(PREV_X):
        cx = px + PAIR
        p, q = f'{key}p{c}', f'{key}q{c}'  # pass i-1 and pass i columns
        s += f'  \\node[font=\\footnotesize, text=black!60] at ({px},5.62) {{$i{{-}}1$}};\n'
        s += f'  \\node[font=\\bfseries\\footnotesize] at ({cx},5.62) {{$i$}};\n'
        # The prelude runs once per trajectory: one box shared by both passes.
        s += node('active', f'{key}pre{c}', (px + cx) / 2, ROWS[1], f', minimum width={PAIR + 0.62}cm')
        # Pass i-1 (non-final): buffer and core run; the source runs only for a
        # temporal write; the coda never runs.
        for k in range(2, 9):
            style = 'ghost'
            if k == 6 and depth:
                style = 'depthmark'
            if k == 7:
                style = 'tempmark' if temporal else 'notrun'
            if k == 8:
                style = 'notrun'
            s += node(style, f'{p}r{k}', px, ROWS[k])
        s += f'  \\draw[ghostflow] ({key}pre{c}.north -| {p}r2) -- ({p}r2.south);\n'
        for k in range(2, 7 if not temporal else 7):
            s += arrow('ghostflow', f'{p}r{k}', f'{p}r{k + 1}')
        # Pass i (final): L2-L8 all run.
        for k in range(2, 9):
            s += node('active', f'{q}r{k}', cx, ROWS[k])
        s += f'  \\draw[flow] ({key}pre{c}.north -| {q}r2) -- ({q}r2.south);\n'
        for k in range(2, 8):
            s += arrow('flow', f'{q}r{k}', f'{q}r{k + 1}')
        s += f'  \\node[font=\\footnotesize] at ({(px + cx) / 2},0.28) {{{POSITIONS[c]}}};\n'
        if depth:
            # Same position: pass i-1's core output feeds pass i before L3.
            later.append(f'  \\draw[dread, rounded corners=3pt] ({p}r6.east) -- ++(0.165,0) |- ({cx - 0.19:.3f},{D_Y:.3f});\n')
            later.append(f'  \\node[injectD] at ({cx:.3f},{D_Y:.3f}) {{D}};\n')
        if temporal and c > 0:
            # Next position: pass i-1's T-source output at t-1 feeds pass i at t, after L1.
            src = f'{key}p{c - 1}r7'
            later.append(f'  \\draw[tread] ({src}.east) .. controls +(0.9,-0.1) and +(-0.55,0.9) .. ({cx - 0.13:.3f},{T_Y + 0.13:.3f});\n')
            later.append(f'  \\node[injectT] at ({cx:.3f},{T_Y:.3f}) {{T}};\n')
    return s + ''.join(later) + '\\end{scope}\n'


def training_figure():
    s = preamble('0.62cm') + scaffold('Training-time information flows')
    s += training_panel('tl', 'Vanilla transformer',
                        'One pass through L1--L8 at every position (embedding and head not shown).',
                        depth=False, temporal=False)
    s += training_panel('tr', 'Depth-recurrent / looped LM',
                        'Pass $i{-}1$ writes the core output (L6) as depth state; pass $i$ reads it at the '
                        'same position, just before the core.', depth=True, temporal=False)
    s += training_panel('bl', 'Temporally recurrent',
                        'Pass $i{-}1$ writes the T-source output (L7) as temporal state; pass $i$ reads it one '
                        'position later, just after the prelude.', depth=False, temporal=True)
    s += training_panel('br', 'Depth + temporal hybrid',
                        'Both at once: depth state stays at its position (short orange arrows), temporal state '
                        'moves one position forward (long purple arrows).', depth=True, temporal=True)
    s += legend([
        [('box', 'ghost', 'pass $i{-}1$ layer'), ('box', 'active', 'pass $i$ layer'),
         ('box', 'notrun', 'not run on this pass'), ('box', 'active, minimum width=1.0cm', 'prelude: runs once')],
        [('box', 'depthmark', 'writes depth state'), ('box', 'tempmark', 'writes temporal state'),
         ('inject', ('dread', 'injectD'), 'depth read (same position)'),
         ('inject', ('tread', 'injectT'), 'temporal read (next position)')],
    ])
    return s + '\\end{tikzpicture}\n\\end{document}\n'


# ---------------------------------------------------------------------------
# Inference time: token t-1 (already computed) and token t

TOKEN_X = [3.35, 6.55]


def inference_panel(key, title, caption, depth, temporal):
    s = panel_open(key, title, caption)
    s += row_labels(depth or temporal)
    s += f'  \\node[font=\\bfseries\\footnotesize, text=black!60] at ({TOKEN_X[0]},5.62) {{token $t{{-}}1$}};\n'
    s += f'  \\node[font=\\bfseries\\footnotesize] at ({TOKEN_X[1]},5.62) {{token $t$}};\n'
    later = []
    for c, x in enumerate(TOKEN_X):
        name = f'{key}k{c}'
        current = c == 1
        for k in range(1, 9):
            style = 'active' if current else 'ghost'
            if k == 6 and depth:
                style = 'depthmark' if current else 'ghostdepth'
            if k == 7 and temporal:
                style = 'tempmark' if current else 'ghosttemp'
            s += node(style, f'{name}r{k}', x, ROWS[k])
        flow = 'flow' if current else 'ghostflow'
        for k in range(1, 8):
            s += arrow(flow, f'{name}r{k}', f'{name}r{k + 1}')
    q = f'{key}k1'
    if depth:
        # Loop the core J times: L6's output re-enters before L3 with the same anchor.
        later.append(f'  \\draw[dread, rounded corners=4pt] ({q}r6.east) -- ++(0.45,0) |- ({TOKEN_X[1] + 0.19:.3f},{D_Y:.3f});\n')
        later.append(f'  \\node[font=\\footnotesize, text=depthdraw, anchor=west] at ({TOKEN_X[1] + 0.95:.3f},{(ROWS[6] + D_Y) / 2:.3f}) {{repeat $J$ times}};\n')
        later.append(f'  \\node[injectD] at ({TOKEN_X[1]:.3f},{D_Y:.3f}) {{D}};\n')
    if temporal:
        p = f'{key}k0'
        later.append(f'  \\draw[tread] ({p}r7.east) .. controls +(1.4,-0.2) and +(-1.2,1.2) .. ({TOKEN_X[1] - 0.13:.3f},{T_Y + 0.13:.3f});\n')
        later.append(f'  \\node[injectT] at ({TOKEN_X[1]:.3f},{T_Y:.3f}) {{T}};\n')
        later.append(f'  \\draw[tread] ({q}r7.east) .. controls +(0.7,0) and +(-0.5,0) .. ({TOKEN_X[1] + 1.55:.3f},{ROWS[8] + 0.35:.3f})'
                     ' node[anchor=west, font=\\footnotesize, text=tempdraw] {to token $t{+}1$};\n')
    return s + ''.join(later) + '\\end{scope}\n'


def inference_figure():
    s = preamble('0.90cm') + scaffold('Inference-time information flows')
    s += inference_panel('tl', 'Vanilla transformer',
                         'Each token runs L1--L8 once; no state is carried between tokens.', False, False)
    s += inference_panel('tr', 'Depth-recurrent / looped LM',
                         'The core (L3--L6) runs $J$ times per token; its output re-enters before L3. '
                         'Depth state is discarded at the next token.', True, False)
    s += inference_panel('bl', 'Temporally recurrent',
                         'Token $t{-}1$\'s T-source output (L7) is read by token $t$ after L1, so the chain '
                         'runs through every earlier token.', False, True)
    s += inference_panel('br', 'Depth + temporal hybrid',
                         'Temporal state crosses tokens; depth state loops only within the current token.', True, True)
    s += legend([
        [('box', 'ghost', 'token $t{-}1$ layer'), ('box', 'active', 'token $t$ layer'),
         ('box', 'depthmark', 'writes depth state'), ('box', 'tempmark', 'writes temporal state')],
        [('arrow', 'flow', 'within-token flow'), ('inject', ('dread', 'injectD'), 'depth read (loop)'),
         ('inject', ('tread', 'injectT'), 'temporal read'), ('arrow', 'tread', 'state passed to next token')],
    ])
    return s + '\\end{tikzpicture}\n\\end{document}\n'


if __name__ == '__main__':
    here = Path(__file__).parent
    (here / 'training_time_layer_wise_draft.tex').write_text(training_figure())
    (here / 'inference_time_layer_wise_draft.tex').write_text(inference_figure())
    print('wrote the two draft figures')
