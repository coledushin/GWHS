"""
Shared library for GWHS exam report generation.

Each exam's generate_report.py imports this and calls lib.run(config).

Config keys:
  file          – path to the Excel data file (relative to CWD or absolute)
  title         – report title shown in the header and <title>
  output        – path to write the HTML output
  home_link     – href for the ← Home nav link (default: '../../index.html')
  class_colors  – dict mapping class name → hex color (optional, merged over CLASS_COLORS)
  show_classes  – include per-class charts (default True)
  subject       – 'US' or 'Global' (default: inferred from content-standard codes)
"""

import json
import re as _re
import uuid
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# ── Palette ───────────────────────────────────────────────────────────────────
C = dict(
    red='#DC4A5A', orange='#F2A541', teal='#35AE97', green='#1E8A6E',
    blue='#3D6FE8', purple='#8B5CF6', navy='#1F2A44', gray='#8A8F98', lgray='#D9DCE1',
)
GRID = '#EEF0F3'
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"

# Default class colors — overridden by config['class_colors'] in run()
CLASS_COLORS = {'Bermejo': C['blue'], 'Sourial': C['blue'], 'Dushin': C['orange'], 'Kovelsky': C['teal'], 'Antaky': C['red'],
                'Other': C['gray']}

LAYOUT = dict(
    plot_bgcolor='white', paper_bgcolor='white',
    font=dict(family=FONT, size=12, color=C['navy']),
    hoverlabel=dict(bgcolor='white', bordercolor=C['lgray'], font_size=12, font_family=FONT,
                    font_color=C['navy']),
    margin=dict(l=20, r=20, t=50, b=20),
)

LEGEND_ITEMS = [
    ('Below 50% — reteach', C['red']),
    ('50–70% — review',     C['orange']),
    ('Above 70% — strong',  C['teal']),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def pct_color(p):
    if p >= 0.70: return C['teal']
    if p >= 0.50: return C['orange']
    return C['red']


def score_color(s, nq):
    p = s / nq
    if p >= 0.78: return C['green']
    if p >= 0.56: return C['teal']
    if p >= 0.34: return C['orange']
    return C['red']


def hex_to_rgba(hex_color, alpha=0.09):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def add_color_legend(fig):
    for label, color in LEGEND_ITEMS:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color=color, size=10, symbol='square'),
            name=label, showlegend=True))


def _style(fig):
    """Shared chart polish: left-aligned titles, soft gridlines, muted axis text."""
    fig.update_layout(title=dict(x=0, xref='container', xanchor='left', pad=dict(l=4),
                                 font=dict(size=15, color=C['navy'], family=FONT)))
    fig.update_xaxes(gridcolor=GRID, linecolor=C['lgray'], tickfont_color='#5B6475',
                     title_font=dict(size=12, color='#5B6475'))
    fig.update_yaxes(gridcolor=GRID, linecolor=C['lgray'], tickfont_color='#5B6475',
                     title_font=dict(size=12, color='#5B6475'))
    return fig


def to_div(fig):
    return _style(fig).to_html(full_html=False, include_plotlyjs=False,
                               config={'displayModeBar': False, 'responsive': True})


def section(title, anchor, content_html, subtitle='', num=None):
    badge = f'<span class="sec-num">{num}</span>' if num is not None else ''
    sub   = f'<p class="subtitle">{subtitle}</p>' if subtitle else ''
    return (f'<section id="{anchor}"><div class="sec-head">{badge}'
            f'<div><h2>{title}</h2>{sub}</div></div>{content_html}</section>')


# ── Written-response structure (NYS Regents) ──────────────────────────────────
# Global II:  Part II   CRQs 29–34b, 1 credit each (Set 1 = 29–31, Set 2 = 32–34b)
#             Part III  Enduring Issues Essay, 0–5 rubric (two raters averaged → half points)
# US History: Part II   two Short-Essay Questions (29, 30), 0–5 rubric each
#             Part IIIA Civic Literacy short answers (31–36), 1 credit each
#             Part IIIB Civic Literacy Essay (37), 0–5 rubric (two raters averaged)
WRITTEN_PARTS = {'Part II', 'Part III', 'Part IIIA', 'Part IIIB'}
_PART_ALIASES = {'Part II': ['SEQ', 'CRQ'], 'Part III': ['EIE'],
                 'Part IIIA': ['IIIA'], 'Part IIIB': ['IIIB', 'CLE']}

ESSAY_TARGET = 3   # rubric level treated as "on track" when summarizing 0–5 essays


def _infer_subject(q):
    """'US' or 'Global' from the content-standard codes (11.x = US, 10.x = Global)."""
    codes = q['Content_Std'].dropna().astype(str) if 'Content_Std' in q.columns else pd.Series(dtype=str)
    us, gl = codes.str.startswith('11.').sum(), codes.str.startswith('10.').sum()
    return 'US' if us > gl else ('Global' if gl > us else None)


def _task_tag(text):
    """Short task label for a 1-credit question, from its wording."""
    t = str(text).lower()
    for keys, tag in [
        (('cause-and-effect', 'cause and effect'),                   'Cause & effect'),
        (('turning point',),                                         'Turning point'),
        (('similarit', 'difference'),                                'Compare'),
        (('historical circumstance', 'historical context',
          'geographic context'),                                     'Context'),
        (('audience', 'purpose', 'point of view', 'bias', 'reliab'), 'Sourcing'),
        (('impact', 'effect'),                                       'Impact'),
        (('effort',),                                                'Efforts'),
    ]:
        if any(k in t for k in keys):
            return tag
    return ''


def _written_items(df, written_raw, subject):
    """Rubric type, max points, display group and label for each written question."""
    items = []
    n_seq = sum(1 for _, part, *_ in written_raw if part == 'Part II')
    for wid, part, row, _ in written_raw:
        col      = f'W_{wid}'
        vals     = df[col].dropna()
        question = '' if pd.isna(row.get('Question', np.nan)) else str(row['Question']).strip()
        content  = row.get('Content_standard', row.get('Content', ''))
        is_us    = subject == 'US' or (subject is None and vals.max() > 1 if len(vals) else False)

        if part == 'Part II' and is_us:
            kind, mx, group = 'essay', 5, 'Short-Essay Questions'
            k = sum(1 for it in items if it['group'] == group) + 1
            label = f'SEQ {k}' if n_seq > 1 else 'SEQ'
        elif part == 'Part II':
            kind, mx, group, label = 'short', 1, 'Constructed Response Questions', f'Q{wid}'
        elif part == 'Part IIIA':
            kind, mx, group, label = 'short', 1, 'Civic Literacy Short Answers', f'Q{wid}'
        elif part == 'Part IIIB':
            kind, mx, group, label = 'essay', 5, 'Civic Literacy Essay', 'Civic Literacy Essay'
        else:   # Part III
            kind, mx, group, label = 'essay', 5, 'Enduring Issues Essay', 'Enduring Issues Essay'

        set_name = ''
        if kind == 'short' and part == 'Part II':
            m = _re.match(r'^(\d+)', wid)
            if m and 29 <= int(m.group(1)) <= 34:
                set_name = 'Set 1' if int(m.group(1)) <= 31 else 'Set 2'

        items.append(dict(id=wid, col=col, part=part, kind=kind, max=mx, group=group,
                          label=label, set=set_name, question=question,
                          tag=_task_tag(question) if kind == 'short' else '',
                          std=_std_code(content) if pd.notna(content) else '',
                          entered=bool(len(vals))))
    return items


def load_conversion(file):
    """Parse a Regents 'Conversion Chart' sheet → (lookup, levels, raw_max) or None.

    lookup[(raw_credits, essay_score)] = scale score; levels = [(n, lo, hi), …] for Levels 1–5.
    """
    try:
        c = pd.read_excel(file, sheet_name='Conversion Chart', header=None)
    except Exception:
        return None
    hdr_row = next((i for i in range(len(c))
                    if pd.to_numeric(c.iloc[i, 2:], errors='coerce').notna().sum() >= 6), None)
    if hdr_row is None:
        return None
    essay_vals = pd.to_numeric(c.iloc[hdr_row, 2:], errors='coerce')
    lookup = {}
    for i in range(hdr_row + 1, len(c)):
        raw = pd.to_numeric(c.iloc[i, 1], errors='coerce')
        if pd.isna(raw):
            continue
        for j, ev in essay_vals.items():
            v = pd.to_numeric(c.iloc[i, j], errors='coerce')
            if pd.notna(ev) and pd.notna(v):
                lookup[(int(raw), float(ev))] = float(v)
    levels = []
    for i in range(len(c)):
        for j in range(c.shape[1]):
            cell = c.iloc[i, j]
            m = _re.search(r'Level\s*(\d)', str(cell)) if isinstance(cell, str) else None
            if not m:
                continue
            rng = _re.search(r'(\d+)\s*-\s*(\d+)', cell)
            if not rng and i + 1 < len(c):
                rng = _re.search(r'(\d+)\s*-\s*(\d+)', str(c.iloc[i + 1, j]))
            if rng:
                levels.append((int(m.group(1)), int(rng.group(1)), int(rng.group(2))))
    if not lookup:
        return None
    raw_max = max(r for r, _ in lookup)
    return lookup, sorted(set(levels)), raw_max


# ── Data loading ──────────────────────────────────────────────────────────────
def _std_code(v):
    """Strip a concatenated '<code>: <name>' content standard down to its code."""
    if pd.isna(v):
        return v
    return str(v).split(':', 1)[0].strip()


def load_data(file, subject=None):
    scores_raw = pd.read_excel(file, sheet_name='Scores')
    questions  = pd.read_excel(file, sheet_name='Questions')
    std_sheet  = pd.read_excel(file, sheet_name='Content Standards')

    # Written-response questions (Part II / III), one per id
    non_mc_ids = set()
    q_id_col   = None
    written_rows = []
    if 'Part' in questions.columns:
        q_id_col = '#' if '#' in questions.columns else ('CMA #' if 'CMA #' in questions.columns else None)
        if q_id_col:
            parts      = questions['Part'].astype(str).str.strip()
            non_mc_ids = set(questions[parts != 'Part I'][q_id_col].astype(str).str.strip().tolist())
            seen = set()
            for _, row in questions[parts.isin(WRITTEN_PARTS)].iterrows():
                wid = str(row[q_id_col]).strip()
                if wid not in seen:      # e.g. the Global essay has one row per document
                    seen.add(wid)
                    written_rows.append((wid, str(row['Part']).strip(), row))

    # Scores column by Q# ('29', '34a'); a lone question in a part may use the part total (e.g. 'SEQ')
    score_cols = {str(c).strip(): c for c in scores_raw.columns}
    part_sizes = defaultdict(int)
    for _, part, _ in written_rows:
        part_sizes[part] += 1
    written_raw = []   # (id, part, question row, raw Scores column)
    for wid, part, row in written_rows:
        raw = score_cols.get(wid)
        if raw is None and part_sizes[part] == 1:
            raw = next((score_cols[a] for a in _PART_ALIASES.get(part, []) if a in score_cols), None)
        if raw is not None:
            written_raw.append((wid, part, row, raw))

    # MC columns: integer-named Scores columns outside Parts II/III
    mc_cols_raw = [c for c in scores_raw.columns
                   if str(c).strip().isdigit() and str(c).strip() not in non_mc_ids]
    NQ = len(mc_cols_raw)

    cols = std_sheet.columns.tolist()
    cols[0], cols[1] = 'Code', 'Framework'
    std_sheet.columns = cols
    std_sheet['Code'] = std_sheet['Code'].astype(str).str.strip()
    std_lookup = dict(zip(std_sheet['Code'], std_sheet['Framework'].astype(str)))

    def std_name(code, max_len=40):
        c = str(code).strip()
        if not c or c in ('nan', 'None', 'NaN', ''):
            return ''
        name = std_lookup.get(c, '')
        if not name or name.strip() in ('nan', 'None', 'NaN', ''):
            return ''
        return (name[:max_len] + '…') if len(name) > max_len else name

    answer_cols  = [f'Q{i}_ans' for i in range(1, NQ + 1)]
    correct_cols = [f'Q{i}'     for i in range(1, NQ + 1)]

    q = questions.copy()
    q = q.rename(columns={
        'CMA #':            'Q_Num',
        '#':                'Q_Num',
        'Source #':         'Source_Num',
        'Stimulus Type':    'Stimulus_Type',
        'Stimulus_type':    'Stimulus_Type',
        'Source Type':      'Source_Type',
        'Source_type':      'Source_Type',
        'Task Model':       'Task_Model',
        'Task_model':       'Task_Model',
        'Content':          'Content_Std',
        'Content_standard': 'Content_Std',
        'Need OI?':         'Need_OI',
        'Need_OI?':         'Need_OI',
        'Skill_standard':   'Skill',
        'Key':              'Answer',
        'Option 1': 'Opt1', 'Option 2': 'Opt2',
        'Option 3': 'Opt3', 'Option 4': 'Opt4',
        'Option_1': 'Opt1', 'Option_2': 'Opt2',
        'Option_3': 'Opt3', 'Option_4': 'Opt4',
    })
    q['Q_Num'] = pd.to_numeric(q['Q_Num'], errors='coerce')
    # Accept bare ('11.01c.2') or concatenated ('11.01c.2: Colonial …') codes
    if 'Content_Std' in q.columns:
        q['Content_Std'] = q['Content_Std'].map(_std_code)
    q = q[q['Q_Num'].isin(range(1, NQ + 1))].sort_values('Q_Num').reset_index(drop=True)
    q['Q_Num'] = q['Q_Num'].astype(int)

    # Full_Answer = text of the keyed option
    if 'Full Answer' in q.columns:
        q = q.drop(columns=['Full Answer'])
    def _full_ans(r):
        try:
            return str(r[f'Opt{int(r["Answer"])}'])
        except Exception:
            return ''
    q['Full_Answer'] = q.apply(_full_ans, axis=1)

    answer_key = {}
    for _, row in q.iterrows():
        ans = row['Answer']
        if pd.notna(ans):
            answer_key[int(row['Q_Num'])] = int(ans)

    rename = {
        'Student Name': 'Student',
        'ELL?': 'ELL',
        'IEP?': 'IEP',
    }
    for raw_col, i in zip(mc_cols_raw, range(1, NQ + 1)):
        rename[raw_col] = f'Q{i}_ans'
    for wid, _, _, raw_col in written_raw:
        rename[raw_col] = f'W_{wid}'

    df_renamed = scores_raw.rename(columns=rename)
    if 'Class' in df_renamed.columns and df_renamed['Class'].notna().any():
        df = df_renamed.dropna(subset=['Class']).copy()
    else:
        df = df_renamed.copy()
        df['Class'] = 'All Students'
    # Drop rows with no MC answers (empty or summary rows)
    df = df.dropna(subset=answer_cols, how='all')
    if 'Period' not in df.columns or df['Period'].isna().all():
        df['Period'] = 1
    for i in range(1, NQ + 1):
        student_ans = pd.to_numeric(df[f'Q{i}_ans'], errors='coerce')
        df[f'Q{i}'] = (student_ans == answer_key.get(i, -1)).astype(int)
    df['MC_Score'] = df[correct_cols].sum(axis=1)
    df['MC_Pct']   = df['MC_Score'] / NQ
    df['MC_Blank'] = df[answer_cols].isna().sum(axis=1)

    def _is_flagged(series):
        """Handles both NaN-sentinel style ('YES'/NaN) and numeric style (1/0)."""
        def check(v):
            if pd.isna(v): return False
            try:
                return float(v) != 0
            except (ValueError, TypeError):
                return str(v).strip().lower() not in ('false', 'no', 'n', '')
        return series.map(check)

    df['Is_ELL'] = _is_flagged(df['ELL'])
    df['Is_IEP'] = _is_flagged(df['IEP'])

    for wid, *_ in written_raw:
        df[f'W_{wid}'] = pd.to_numeric(df[f'W_{wid}'], errors='coerce')

    subject = subject or _infer_subject(q)
    written = _written_items(df, written_raw, subject)

    return df, q, written, subject, std_lookup, std_name, answer_cols, correct_cols, NQ


# ── Figure builders ───────────────────────────────────────────────────────────
def fig_histogram(df, nq):
    mean_s = df['MC_Score'].mean()
    counts = [(df['MC_Score'] == s).sum() for s in range(nq + 1)]
    b1, b2, b3 = round(nq * 0.34), round(nq * 0.56), round(nq * 0.78)
    bands = [
        ('Needs Support', C['red'],    -0.5,      b1 + 0.5),
        ('Approaching',   C['orange'],  b1 + 0.5, b2 + 0.5),
        ('Meeting Standards',   C['teal'],    b2 + 0.5, b3 + 0.5),
        ('Exceeding',     C['green'],   b3 + 0.5, nq + 0.5),
    ]

    def _bc(s):
        if s > b3: return C['green']
        if s > b2: return C['teal']
        if s > b1: return C['orange']
        return C['red']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(nq + 1)), y=counts,
        marker_color=[_bc(s) for s in range(nq + 1)],
        marker_line_color='white', marker_line_width=1.5,
        hovertemplate='Score %{x}/' + str(nq) + ': %{y} students<extra></extra>',
        showlegend=False,
    ))
    fig.add_vline(x=mean_s, line_dash='dot', line_color='rgba(100,100,100,0.4)', line_width=1.5)
    for label, color, lo, hi in bands:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=color, opacity=0.06, line_width=0,
                      annotation_text=label, annotation_position='top left',
                      annotation_font=dict(size=11, color=color))
    fig.update_layout(**LAYOUT,
        title='How Did Students Score?', height=350,
        xaxis=dict(title=f'Questions Correct (out of {nq})',
                   tickvals=list(range(nq + 1)), showgrid=False, zeroline=False),
        yaxis=dict(title='Number of Students', showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    return fig


def fig_class_bar(df, nq):
    cp = df.groupby(['Class', 'Period'])['MC_Score'].agg(['mean', 'count']).reset_index()
    classes = sorted(df['Class'].unique())
    bar_width, group_gap = 0.22, 0.15
    fig = go.Figure()
    current_x = 0.0
    class_centers = {}
    for cls in classes:
        cls_data = cp[cp['Class'] == cls].sort_values('Period')
        n_bars = len(cls_data)
        for pi, (_, row) in enumerate(cls_data.iterrows()):
            x_pos = current_x + pi * bar_width
            fig.add_trace(go.Bar(
                x=[x_pos], y=[row['mean']],
                marker_color=CLASS_COLORS.get(cls, C['blue']),
                marker_line_color='white', marker_line_width=1,
                text=[f"P{int(row['Period'])}"],
                textposition='inside', insidetextanchor='end',
                textfont=dict(color='white', size=11),
                hovertemplate=f'Avg {row["mean"]:.1f}/{nq}  (n={int(row["count"])})<extra></extra>',
                showlegend=False, width=bar_width * 0.85,
            ))
        class_centers[cls] = current_x + (n_bars - 1) * bar_width / 2
        current_x += n_bars * bar_width + group_gap
    fig.update_xaxes(tickvals=list(class_centers.values()), ticktext=list(class_centers.keys()),
                     showgrid=False, zeroline=False,
                     range=[-0.25, current_x - group_gap + 0.25])
    fig.update_yaxes(title_text=f'Avg Score (out of {nq})', range=[0, nq * 1.28],
                     showgrid=True, gridcolor='#EEEEEE', zeroline=False)
    fig.update_layout(**LAYOUT, title='Average Score by Class & Period', height=320)
    return fig


def fig_grade_bar(df, nq):
    """Average score by grade (6–13); None if fewer than two grades."""
    if 'Grade' not in df.columns:
        return None
    valid = df['Grade'].dropna()
    valid = pd.to_numeric(valid, errors='coerce').dropna()
    valid = valid[valid.between(6, 13)]
    grades = sorted(valid.unique())
    if len(grades) < 2:
        return None
    grade_palette = [C['blue'], C['orange'], C['teal'], C['green'], C['red']]
    df_g = df[pd.to_numeric(df['Grade'], errors='coerce').between(6, 13)]
    fig = go.Figure()
    for i, grade in enumerate(grades):
        g_df = df_g[df_g['Grade'] == grade]
        mean_val = g_df['MC_Score'].mean()
        n = len(g_df)
        fig.add_trace(go.Bar(
            x=[str(int(grade)) if float(grade).is_integer() else str(grade)],
            y=[mean_val],
            name=f'Grade {int(grade) if float(grade).is_integer() else grade}  (n={n})',
            marker_color=grade_palette[i % len(grade_palette)],
            marker_line_color='white', marker_line_width=1.5,
            text=[f'{mean_val:.1f}'], textposition='outside',
            width=0.45,
            hovertemplate=f'Grade {grade}: {mean_val:.1f}/{nq} avg  (n={n})<extra></extra>',
        ))
    fig.update_xaxes(title_text='', showgrid=False, zeroline=False)
    fig.update_yaxes(title_text=f'Avg Score (out of {nq})', range=[0, nq * 1.28],
                     showgrid=True, gridcolor='#EEEEEE', zeroline=False)
    fig.update_layout(**LAYOUT, title='Average Score by Grade', height=320,
                      showlegend=False)
    return fig


def fig_class_dist(df, nq):
    classes = sorted(df['Class'].unique())
    fig = go.Figure()
    for cls in classes:
        cls_df = df[df['Class'] == cls]
        x_pts  = list(range(nq + 1))
        pcts   = np.array([(cls_df['MC_Score'] == s).sum() / len(cls_df) for s in x_pts])
        color  = CLASS_COLORS.get(cls, C['blue'])
        sigma  = max(0.4, nq / 28)
        y_smooth = np.clip(gaussian_filter1d(pcts.astype(float), sigma=sigma), 0, None).tolist()
        fig.add_trace(go.Scatter(
            x=x_pts, y=y_smooth, mode='lines',
            line=dict(color=color, width=2.5),
            name=f'{cls} (n={len(cls_df)})',
            hovertemplate='%{y:.1%} of class at score %{x}<extra></extra>',
        ))
    tick_step = max(1, nq // 10)
    fig.update_xaxes(title_text=f'Score (out of {nq})',
                     tickvals=list(range(0, nq + 1, tick_step)),
                     showgrid=False, zeroline=False)
    fig.update_yaxes(title_text='Share of Class', tickformat='.0%',
                     showgrid=True, gridcolor='#EEEEEE', zeroline=False)
    fig.update_layout(**LAYOUT, title='Score Distribution by Class', height=320,
                      legend=dict(orientation='h', y=-0.22, x=0.5, xanchor='center'))
    return fig


def difficulty_html(df, q, std_name, nq, show_classes=True):
    """Heatmap with Hardest→Easiest / Question Order toggle using .q-btn buttons."""
    overall_p     = [df[f'Q{i}'].mean() for i in range(1, nq + 1)]
    order         = list(np.argsort(overall_p))
    q_nums_sorted = [order[j] + 1 for j in range(nq)]
    col_labels    = [f'{q_nums_sorted[j]}' for j in range(nq)]
    full_q        = [str(q['Question'].tolist()[order[j]]).strip() for j in range(nq)]
    answers       = [str(q['Full_Answer'].tolist()[order[j]]).strip() for j in range(nq)]
    std_labels    = [std_name(q['Content_Std'].tolist()[order[j]], 35) for j in range(nq)]

    row_names = ['Overall'] + (sorted(df['Class'].unique().tolist()) if show_classes else [])
    ns        = []
    z_data, text_data, hover_data = [], [], []
    for row_name in row_names:
        fdf   = df if row_name == 'Overall' else df[df['Class'] == row_name]
        n     = len(fdf)
        ns.append(n)
        row_p = [float(fdf[f'Q{q_nums_sorted[j]}'].mean()) for j in range(nq)]
        z_data.append(row_p)
        text_data.append([f'{p:.0%}' for p in row_p])
        hover_data.append([
            f'<b>{q_nums_sorted[j]}: {std_labels[j]}</b><br>'
            f'{full_q[j]}<br><br>'
            f'✓ {answers[j]}<br>'
            f'{row_p[j]:.0%} correct'
            for j in range(nq)
        ])

    all_vals = [v for row in z_data for v in row]
    zmin_val, zmax_val = min(all_vals), max(all_vals)

    nat_answers = [str(q['Full_Answer'].tolist()[j]).strip() for j in range(nq)]
    nat_stds    = [std_name(q['Content_Std'].tolist()[j], 35) for j in range(nq)]
    nat_qs      = [str(q['Question'].tolist()[j]).strip() for j in range(nq)]

    col_nat = [f'{j+1}' for j in range(nq)]
    z_nat, txt_nat, hov_nat = [], [], []
    for ri, row_name in enumerate(row_names):
        z_nat.append([float(z_data[ri][order.index(j)]) for j in range(nq)])
        txt_nat.append([f'{z_nat[-1][j]:.0%}' for j in range(nq)])
        hov_nat.append([
            f'<b>{j+1}: {nat_stds[j]}</b><br>'
            f'{nat_qs[j]}<br><br>'
            f'✓ {nat_answers[j]}<br>'
            f'{z_nat[-1][j]:.0%} correct'
            for j in range(nq)
        ])

    # Single-row heatmap: hide the "Overall" label
    y_labels = row_names if show_classes else [''] * len(row_names)

    show_text = nq <= 20
    fig = go.Figure(go.Heatmap(
        z=z_data, x=col_labels, y=y_labels,
        colorscale=[[0, '#E5707C'], [0.5, '#FBEBD3'], [1.0, '#4CBBA5']],
        zmin=zmin_val, zmax=zmax_val, xgap=4, ygap=4,
        text=text_data, texttemplate='%{text}' if show_text else '',
        textfont=dict(size=12, color=C['navy']),
        hoverinfo='text', hovertext=hover_data, showscale=True,
        colorbar=dict(title=dict(text='% Correct', font=dict(size=11)), tickformat='.0%',
                      thickness=12, outlinewidth=0,
                      tickvals=[zmin_val, (zmin_val + zmax_val) / 2, zmax_val],
                      ticktext=[f'{zmin_val:.0%}', f'{(zmin_val+zmax_val)/2:.0%}', f'{zmax_val:.0%}']),
    ))
    fig.update_layout(**LAYOUT,
        title='', height=max(240, 80 + 70 * len(row_names)),
        xaxis=dict(side='top', title='Question', tickfont=dict(size=12), showgrid=False, ticklen=0),
        yaxis=dict(autorange='reversed', tickfont=dict(size=12), showgrid=False, ticklen=0,
                   ticksuffix='  '),
    )
    if show_classes:
        fig.add_shape(type='line', xref='paper', yref='y', x0=0, x1=1, y0=0.5, y1=0.5,
                      line=dict(color='white', width=12), layer='above')
    fig.update_layout(margin=dict(l=95 if show_classes else 20, r=20, t=70, b=20))

    chart_div = to_div(fig)
    m = _re.search(r'<div id="([^"]+)"', chart_div)
    fig_id = m.group(1) if m else 'plotly-diff'
    bid    = fig_id[-8:]

    hard_js = json.dumps({'x': [col_labels], 'z': [z_data],  'text': [text_data],  'hovertext': [hover_data]})
    nat_js  = json.dumps({'x': [col_nat],    'z': [z_nat],   'text': [txt_nat],    'hovertext': [hov_nat]})

    btn_html = (
        f'<div class="q-btns">'
        f'<span class="btn-label">Sort</span>'
        f'<button class="q-btn q-btn-active" id="diff-hard-{bid}">Hardest → Easiest</button>'
        f'<button class="q-btn" id="diff-nat-{bid}">Question Order</button>'
        f'</div>'
    )
    script = f'''
<script>
(function(){{
  var hardCfg={hard_js};
  var natCfg={nat_js};
  function attach(){{
    var el=document.getElementById('{fig_id}');
    if(!el){{setTimeout(attach,80);return;}}
    var bH=document.getElementById('diff-hard-{bid}');
    var bN=document.getElementById('diff-nat-{bid}');
    bH.addEventListener('click',function(){{
      Plotly.restyle(el,hardCfg);
      bH.classList.add('q-btn-active'); bN.classList.remove('q-btn-active');
    }});
    bN.addEventListener('click',function(){{
      Plotly.restyle(el,natCfg);
      bN.classList.add('q-btn-active'); bH.classList.remove('q-btn-active');
    }});
  }}
  attach();
}})();
</script>'''
    return chart_div + btn_html + script


def distractor_html(df, q, nq):
    """Question text + bar chart + Q-number buttons."""
    has_opts  = 'Opt1' in q.columns
    q_text_map = {int(r['Q_Num']): str(r['Question']).strip() for _, r in q.iterrows()}

    fig = go.Figure()
    for i in range(nq):
        q_num = i + 1
        qrow  = q.loc[q['Q_Num'] == q_num].iloc[0]
        ca    = int(qrow['Answer'])
        opts  = ([str(qrow[f'Opt{j}']).strip() for j in range(1, 5)]
                 if has_opts else [str(j) for j in range(1, 5)])
        answered = pd.to_numeric(df[f'Q{q_num}_ans'], errors='coerce').dropna().astype(int)
        total    = len(answered)
        counts   = answered.value_counts().reindex([1, 2, 3, 4], fill_value=0)
        pcts     = [counts[j] / total if total > 0 else 0 for j in range(1, 5)]
        fig.add_trace(go.Bar(
            x=[1, 2, 3, 4], y=pcts,
            marker_color=[C['teal'] if j == ca else '#C9CED6' for j in range(1, 5)],
            marker_line_width=0,
            text=[f'{p:.0%}' for p in pcts], textposition='outside',
            hovertext=[
                f'<b>{opts[j-1]}</b><br>'
                + ('✓ Correct' if j == ca else '✗ Incorrect')
                for j in range(1, 5)
            ],
            hoverinfo='text', visible=(q_num == 1), showlegend=False, cliponaxis=False,
            textfont=dict(size=13, color=C['navy']), width=0.6,
        ))

    fig.update_layout(**LAYOUT, height=320,
        xaxis=dict(tickvals=[1, 2, 3, 4], ticktext=['Choice 1', 'Choice 2', 'Choice 3', 'Choice 4'],
                   title='', showgrid=False, zeroline=False, tickfont=dict(size=13)),
        yaxis=dict(tickformat='.0%', title='% of Students', tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], range=[0, 1.12],
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    fig.update_layout(margin=dict(l=0, r=20, t=5, b=20))

    chart_div = to_div(fig)
    m = _re.search(r'<div id="([^"]+)"', chart_div)
    fig_id = m.group(1) if m else 'plotly-dis'
    txt_id = f'dis-q-text-{fig_id[-8:]}'
    lbl_id = f'dis-q-lbl-{fig_id[-8:]}'

    texts_js = '[' + ','.join(json.dumps(q_text_map.get(i, '')) for i in range(1, nq + 1)) + ']'
    btn_items = ''.join(
        f'<button class="q-btn{"  q-btn-active" if q_num == 1 else ""}" '
        f'data-qi="{q_num - 1}" data-fig="{fig_id}">{q_num}</button>'
        for q_num in range(1, nq + 1)
    )

    return f'''
<div class="q-btns q-btns-center"><span class="btn-label">Question</span>{btn_items}</div>
<div class="q-card"><span class="q-card-num" id="{lbl_id}">Q1</span>
<p id="{txt_id}">{q_text_map[1]}</p></div>
<div class="dis-legend"><span><i style="background:{C['teal']}"></i>Correct answer</span>
<span><i style="background:#C9CED6"></i>Incorrect choice — a tall gray bar signals a common misconception</span></div>
{chart_div}
<script>
(function(){{
  var texts={texts_js};
  var td=document.getElementById('{txt_id}');
  var nq={nq};
  function attach(){{
    var el=document.getElementById('{fig_id}');
    if(!el){{setTimeout(attach,80);return;}}
    document.querySelectorAll('.q-btn[data-fig="{fig_id}"]').forEach(function(btn){{
      btn.addEventListener('click',function(){{
        var qi=parseInt(this.dataset.qi);
        var vis=Array(nq).fill(false); vis[qi]=true;
        Plotly.restyle(el,{{visible:vis}});
        td.textContent=texts[qi]||'';
        document.getElementById('{lbl_id}').textContent='Q'+(qi+1);
        document.querySelectorAll('.q-btn[data-fig="{fig_id}"]').forEach(function(b){{
          b.classList.toggle('q-btn-active', b===btn);
        }});
      }});
    }});
  }}
  attach();
}})();
</script>'''


def fig_standards(df, q, std_name, nq):
    rows = []
    for _, row in q.iterrows():
        qn = int(row['Q_Num'])
        rows.append(dict(Q_Num=qn,
            Std_Name=std_name(row['Content_Std'], 45),
            Question=str(row['Question']).strip(),
            Full_Answer=str(row['Full_Answer']).strip(),
            P_Value=df[f'Q{qn}'].mean()))
    sdf = pd.DataFrame(rows).sort_values('P_Value')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[f'Q{int(r.Q_Num)}' for r in sdf.itertuples()],
        x=sdf['P_Value'].tolist(), orientation='h',
        marker_color=[pct_color(p) for p in sdf['P_Value']],
        text=[f'{p:.0%}' for p in sdf['P_Value']], textposition='outside',
        hovertext=[
            f'<b>Q{int(r.Q_Num)}: {r.Std_Name}</b><br>{r.Question}<br><br>'
            f'✓ <b>Correct:</b> {r.Full_Answer}<br><b>% correct:</b> {r.P_Value:.0%}'
            for r in sdf.itertuples()],
        hoverinfo='text', cliponaxis=False, showlegend=False,
    ))
    add_color_legend(fig)
    fig.add_vline(x=0.70, line_dash='dot', line_color=C['navy'], opacity=0.4,
                  annotation_text='70% target', annotation_position='top right',
                  annotation_font=dict(size=10, color=C['navy']))
    fig.update_layout(**LAYOUT,
        title='Content Standards — Which Need More Practice?',
        height=max(380, 40 * nq + 100),
        legend=dict(orientation='h', y=-0.14, x=0.5, xanchor='center', font=dict(size=11)),
        xaxis=dict(tickformat='.0%', title='% Who Answered Correctly',
                   tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], range=[0, 1.12], showgrid=True, gridcolor='#EEEEEE', zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=13), ticksuffix='  '),
    )
    fig.update_layout(margin=dict(l=80, r=20, t=60, b=80))
    return fig


def fig_groups(df, nq):
    masks = {
        'General Ed':    (~df['Is_ELL']) & (~df['Is_IEP']),
        'IEP only':      (~df['Is_ELL']) & df['Is_IEP'],
        'ELL only':       df['Is_ELL']  & (~df['Is_IEP']),
        'Both ELL & IEP': df['Is_ELL']  & df['Is_IEP'],
    }
    groups  = {n: m for n, m in masks.items() if m.sum() > 0}
    palette = [C['blue'], C['teal'], C['orange'], C['red']]
    fig = go.Figure()
    for (name, mask), color in zip(groups.items(), palette):
        mean_val = df[mask]['MC_Score'].mean()
        n = int(mask.sum())
        fig.add_trace(go.Bar(
            x=[name], y=[mean_val], name=f'{name}  (n={n})',
            marker_color=color, marker_line_color='white', marker_line_width=1.5,
            text=[f'{mean_val:.1f}'], textposition='outside', width=0.45, hoverinfo='skip',
        ))
    fig.add_hline(y=df['MC_Score'].mean(), line_dash='dot', line_color=C['navy'], opacity=0.55,
                  annotation_text=f'Overall: {df["MC_Score"].mean():.1f}',
                  annotation_position='top right', annotation_font=dict(size=10, color=C['navy']))
    fig.update_layout(**LAYOUT,
        title='Average Score by Student Group', height=380, showlegend=True,
        legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
        barmode='group',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title=f'Average Score (out of {nq})', range=[0, nq * 1.28],
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    return fig


def fig_boxplot(df, nq):
    class_list = sorted(df['Class'].unique())
    fig = go.Figure()
    for cls in class_list:
        color = CLASS_COLORS.get(cls, C['blue'])
        vals  = df[df['Class'] == cls]['MC_Score'].tolist()
        fig.add_trace(go.Box(
            y=vals, name=cls,
            marker_color=color, line_color=color,
            fillcolor=hex_to_rgba(color, 0.35),
            boxpoints='all', jitter=0.35, pointpos=0,
            marker=dict(size=5, opacity=0.4, color=color),
            hovertemplate=f'{cls}: %{{y}}/{nq}<extra></extra>',
        ))
    fig.update_layout(**LAYOUT,
        title='Score Distribution by Class', height=380, showlegend=False,
        yaxis=dict(title=f'MC Score (out of {nq})', showgrid=True,
                   gridcolor='#EEEEEE', zeroline=False, range=[-0.5, nq + 0.5]),
        xaxis=dict(showgrid=False, zeroline=False),
    )
    return fig


PB_BANDS = [   # (minimum PB, label, color, what it means)
    (0.30, 'Strong',     C['green'],  'Cleanly separates students who know the material from those who don\'t.'),
    (0.20, 'Acceptable', C['teal'],   'Does its job; no action needed.'),
    (0.10, 'Weak',       C['orange'], 'Only loosely tied to overall performance — check the wording and distractors.'),
    (-1.0, 'Poor',       C['red'],    'Almost no relationship, or negative. Check the answer key first.'),
]


def _pb_band(v):
    return next((lbl, col) for lo, lbl, col, _ in PB_BANDS if v >= lo)


def _item_pb(fdf, i):
    """Point-biserial of question i against the rest-of-test MC score; None when undefined."""
    x, rest = fdf[f'Q{i}'], fdf['MC_Score'] - fdf[f'Q{i}']
    if len(fdf) < 3 or x.nunique() < 2 or rest.nunique() < 2:
        return None
    return float(stats.pointbiserialr(x, rest)[0])


def pb_guide_html(n):
    bands = ''.join(
        f'<li><span class="pb-chip" style="background:{col}">{lbl}</span>'
        f'<span class="pb-range">{"below 0.10" if lo < 0 else f"{lo:.2f}+" if lo >= 0.30 else f"{lo:.2f}–{up - 0.01:.2f}"}</span>'
        f'<span class="pb-mean">{mean}</span></li>'
        for (lo, lbl, col, mean), up in zip(PB_BANDS, [None, 0.30, 0.20, 0.10]))
    small = (f'<p class="note"><b>Small sample:</b> with {n} students, one PB can shift by 0.1 or more '
             f'from a handful of answers. Treat it as a prompt to look closer, not a verdict.</p>'
             if n < 60 else '')
    return f'''<div class="pb-guide">
  <div class="pb-text">
    <div class="callout-title">How to read point-biserial (PB)</div>
    <p>A question is only useful if it tells students apart. <b>PB measures that</b>: it is the correlation
    between getting this question right and scoring well on <i>the rest</i> of the multiple-choice questions
    (the question itself is left out, so it can\'t inflate its own score). It runs from −1 to +1.</p>
    <ul>
      <li><b>High PB</b> — students who did well elsewhere got it right; students who struggled missed it.</li>
      <li><b>Near zero</b> — right and wrong answers are scattered. The question isn\'t revealing who understands.</li>
      <li><b>Negative</b> — stronger students missed it more than weaker ones. Usually a wrong answer key or a
      distractor that sounds more right than the answer.</li>
    </ul>
  </div>
  <div class="pb-side">
    <ul class="pb-bands">{bands}</ul>
  </div>
</div>
<div class="pb-quad">
  <div><b>Top right · easy &amp; discriminating</b>Healthy. Most students get it, and the few who miss it are the ones struggling.</div>
  <div><b>Top left · hard &amp; discriminating</b>Where your strongest students pull ahead. Good questions, so a low % correct here isn\'t a flaw.</div>
  <div><b>Bottom right · easy &amp; flat</b>Expected. When nearly everyone gets it right there is little room to separate students.</div>
  <div><b>Bottom left · hard &amp; flat</b>Review first. Students may be guessing, or the wording or a distractor is misleading.</div>
</div>
{small}'''


def pb_flags_html(df, q, std_name, nq):
    """Questions whose overall PB falls below 0.20, with a suggested reading for each."""
    rows, n_strong = [], 0
    for i in range(1, nq + 1):
        pv, pb = float(df[f'Q{i}'].mean()), _item_pb(df, i)
        topic = std_name(q.loc[q['Q_Num'] == i, 'Content_Std'].values[0], 50)
        if pb is None:
            reading = 'Everyone answered alike, so PB can\'t be calculated.'
            tag = f'{pv:.0%} correct'
        elif pb >= 0.30:
            n_strong += 1
            continue
        elif pb >= 0.20:
            continue
        else:
            if pb < 0:
                reading = 'Stronger students missed this more often. Check the answer key and the most-chosen wrong answer.'
            elif pv >= 0.85:
                reading = 'Nearly everyone got it right, so a low PB is expected. Probably fine.'
            elif pv < 0.50:
                reading = 'Hard and not discriminating. Students may be guessing; review the wording and distractors.'
            else:
                reading = 'Only loosely tied to overall performance. Review the wording and distractors.'
            tag = f'PB {pb:.2f} · {pv:.0%} correct'
        rows.append(f'<li><b>Q{i}</b>{" · " + topic if topic else ""} <span class="pill">{tag}</span>'
                    f'<span class="pb-reading">{reading}</span></li>')
    head = f'{n_strong} of {nq} questions have a PB of 0.30 or higher.'
    if not rows:
        return (f'<div class="callout"><div class="callout-title">Questions to review</div>'
                f'<p style="margin:0">{head} None fall below 0.20 on the overall view.</p></div>')
    return (f'<div class="callout"><div class="callout-title">Questions to review</div>'
            f'<p style="margin:0 0 6px">{head} These fall below 0.20 on the overall view:</p>'
            f'<ul class="pb-flags">{"".join(rows)}</ul></div>')


def scatter_html(df, q, std_name, nq, show_classes=True):
    """Difficulty vs. discrimination scatter with Overall / per-class toggle."""
    topics  = [std_name(q.loc[q['Q_Num'] == i, 'Content_Std'].values[0], 60) for i in range(1, nq + 1)]
    q_texts = [str(q.loc[q['Q_Num'] == i, 'Question'].values[0]).strip()[:80] for i in range(1, nq + 1)]

    def compute(fdf):
        p  = [float(fdf[f'Q{i}'].mean()) for i in range(1, nq + 1)]
        raw    = [_item_pb(fdf, i) for i in range(1, nq + 1)]
        pb     = [0.0 if v is None else v for v in raw]
        colors = [pct_color(v) for v in p]
        hover  = [
            f'<b>{i}: {topics[i-1]}</b><br>{q_texts[i-1]}…<br><br>'
            f'Difficulty: {p[i-1]:.0%} correct<br>'
            + (f'Point-biserial: {pb[i-1]:.2f} ({_pb_band(pb[i-1])[0].lower()})'
               if raw[i-1] is not None else 'Point-biserial: n/a (everyone answered alike)')
            for i in range(1, nq + 1)
        ]
        return p, pb, colors, hover

    views = ['Overall'] + (sorted(df['Class'].unique().tolist()) if show_classes else [])
    all_data = {}
    for view in views:
        fdf = df if view == 'Overall' else df[df['Class'] == view]
        all_data[view] = compute(fdf)

    p0, pb0, c0, h0 = all_data['Overall']
    all_pb = [v for d in all_data.values() for v in d[1]]
    y_lo   = min(-0.1, min(all_pb) - 0.05)
    y_hi   = max(0.6, max(all_pb) + 0.12)

    fig = go.Figure()
    fig.add_hrect(y0=y_lo, y1=0.20, fillcolor=C['red'], opacity=0.05, line_width=0, layer='below')
    fig.add_trace(go.Scatter(
        x=p0, y=pb0, mode='markers+text',
        text=[f'{i}' for i in range(1, nq + 1)], textposition='top right',
        textfont=dict(size=11, color=C['navy']),
        marker=dict(color=c0, size=14, line=dict(color='white', width=1.5)),
        hovertext=h0, hoverinfo='text', showlegend=False,
    ))
    for label, color in [('Hard (<50%)', C['red']), ('Medium (50–70%)', C['orange']), ('Easy (>70%)', C['teal'])]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color=color, size=10), name=label, showlegend=True))
    fig.add_hline(y=0.20, line_dash='dot', line_color=C['red'], opacity=0.6,
                  annotation_text='0.20 · below this, review', annotation_position='bottom right',
                  annotation_font=dict(size=10, color=C['red']))
    fig.add_hline(y=0.30, line_dash='dot', line_color=C['green'], opacity=0.6,
                  annotation_text='0.30 · strong', annotation_position='top right',
                  annotation_font=dict(size=10, color=C['green']))
    fig.add_vline(x=0.50, line_dash='dot', line_color=C['gray'], opacity=0.35)
    for x, y, xa, ya, txt in [(0.01, 0.99, 'left', 'top', 'Hard · discriminating'),
                              (0.99, 0.99, 'right', 'top', 'Easy · discriminating'),
                              (0.01, 0.01, 'left', 'bottom', 'Hard · flat — review first'),
                              (0.99, 0.01, 'right', 'bottom', 'Easy · flat — expected')]:
        fig.add_annotation(xref='paper', yref='paper', x=x, y=y, xanchor=xa, yanchor=ya, text=txt,
                           showarrow=False, font=dict(size=10, color=C['gray']))
    fig.update_layout(**LAYOUT,
        title='Difficulty vs. Discrimination', height=460,
        legend=dict(orientation='h', y=-0.16, x=0.5, xanchor='center', font=dict(size=11)),
        xaxis=dict(tickformat='.0%', title='Difficulty (% correct)',
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False, range=[0, 1.05]),
        yaxis=dict(title='Point-biserial (higher = separates students better)',
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False, range=[y_lo, y_hi]),
    )

    chart_div = to_div(fig)
    m = _re.search(r'<div id="([^"]+)"', chart_div)
    fig_id = m.group(1) if m else 'plotly-scatter'
    ctr_id = 'sc-' + fig_id[-8:]

    configs_js = json.dumps({
        view: {
            'x':            [all_data[view][0]],
            'y':            [all_data[view][1]],
            'marker.color': [all_data[view][2]],
            'hovertext':    [all_data[view][3]],
        }
        for view in views
    })

    btns = ''.join(
        f'<button class="q-btn{"  q-btn-active" if i == 0 else ""}" '
        f'data-view="{view}" data-ctr="{ctr_id}">{view}</button>'
        for i, view in enumerate(views)
    )

    script = f'''
<script>
(function(){{
  var cfgs={configs_js};
  function attach(){{
    var el=document.getElementById('{fig_id}');
    if(!el){{setTimeout(attach,80);return;}}
    document.querySelectorAll('.q-btn[data-ctr="{ctr_id}"]').forEach(function(btn){{
      btn.addEventListener('click',function(){{
        Plotly.restyle(el, cfgs[this.dataset.view], [0]);
        document.querySelectorAll('.q-btn[data-ctr="{ctr_id}"]').forEach(function(b){{
          b.classList.toggle('q-btn-active', b===btn);
        }});
      }});
    }});
  }}
  attach();
}})();
</script>'''

    if len(views) > 1:
        return f'<div class="q-btns"><span class="btn-label">View</span>{btns}</div>\n{chart_div}{script}'
    return chart_div


def fig_oi(df, q, nq):
    if 'Need_OI' not in q.columns:
        return None

    def is_oi(val):
        if pd.isna(val): return False
        return str(val).strip().lower() not in ('', 'no', 'n', 'false', '0', 'nan', 'none')

    q_copy = q.copy()
    q_copy['OI_Required'] = q_copy['Need_OI'].apply(is_oi)
    oi_yes = q_copy[q_copy['OI_Required']]['Q_Num'].tolist()
    oi_no  = q_copy[~q_copy['OI_Required']]['Q_Num'].tolist()
    categories = [(lbl, qns) for lbl, qns in [('Requires OI', oi_yes), ('No OI Needed', oi_no)] if qns]
    if len(categories) < 2:
        return None

    labels   = [lbl for lbl, _ in categories]
    vals     = [float(np.mean([df[f'Q{qn}'].mean() for qn in qns])) for _, qns in categories]
    q_counts = [len(qns) for _, qns in categories]

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=[pct_color(v) for v in vals],
        marker_line_color='white', marker_line_width=1.5,
        text=[f'{v:.0%}' for v in vals], textposition='outside', cliponaxis=False,
        hovertemplate='%{customdata} questions<extra></extra>',
        customdata=q_counts, showlegend=False, width=0.4,
    ))
    fig.update_layout(**LAYOUT,
        title='Outside Information Required vs. Not', height=380,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(tickformat='.0%', title='Average % Correct', tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], range=[0, 1.12],
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    return fig


def fig_task_model(df, q, nq):
    tms = q.dropna(subset=['Task_Model'])
    if tms.empty:
        return None

    def tm_sort_key(tm):
        m = _re.match(r'^(\d+)', tm.strip())
        return int(m.group(1)) if m else float('inf')

    def tm_num(tm):
        m = _re.match(r'^(\d+)', tm.strip())
        return m.group(1) if m else tm[:3]

    task_models = sorted(tms['Task_Model'].astype(str).str.strip().unique().tolist(), key=tm_sort_key)
    labels, vals, hover_texts = [], [], []
    for tm in task_models:
        tm_qs  = q[q['Task_Model'].astype(str).str.strip() == tm]['Q_Num'].tolist()
        avg    = float(np.mean([df[f'Q{qn}'].mean() for qn in tm_qs]))
        q_list = ', '.join(str(qn) for qn in sorted(tm_qs))
        labels.append(tm_num(tm))
        vals.append(avg)
        hover_texts.append(f'<b>{tm}</b><br>Questions: {q_list}')

    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=[pct_color(v) for v in vals],
        marker_line_color='white', marker_line_width=1.5,
        text=[f'{v:.0%}' for v in vals], textposition='outside', cliponaxis=False,
        hovertext=hover_texts, hoverinfo='text', showlegend=False, width=0.5,
    ))
    fig.update_layout(**LAYOUT,
        title='Performance by Task Model', height=380,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=13)),
        yaxis=dict(tickformat='.0%', title='Average % Correct', tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], range=[0, 1.12],
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    return fig


def fig_skill(df, q, nq):
    sks = q.dropna(subset=['Skill'])
    if sks.empty:
        return None

    def skill_label(s):
        m = _re.match(r'^([A-Z](?:\.\d+)?)', s.strip())
        return m.group(1) if m else s[:2]

    skills = sorted(sks['Skill'].astype(str).str.strip().unique().tolist())
    labels, vals, hover_texts = [], [], []
    for sk in skills:
        sk_qs  = q[q['Skill'].astype(str).str.strip() == sk]['Q_Num'].tolist()
        avg    = float(np.mean([df[f'Q{qn}'].mean() for qn in sk_qs]))
        q_list = ', '.join(str(qn) for qn in sorted(sk_qs))
        labels.append(skill_label(sk))
        vals.append(avg)
        hover_texts.append(f'<b>{sk}</b><br>Questions: {q_list}')

    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=[pct_color(v) for v in vals],
        marker_line_color='white', marker_line_width=1.5,
        text=[f'{v:.0%}' for v in vals], textposition='outside', cliponaxis=False,
        hovertext=hover_texts, hoverinfo='text', showlegend=False, width=0.4,
    ))
    fig.update_layout(**LAYOUT,
        title='Performance by Skill', height=380,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=13)),
        yaxis=dict(tickformat='.0%', title='Average % Correct', tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], range=[0, 1.12],
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    return fig


def fig_stimulus_type(df, q, nq):
    if 'Stimulus_Type' not in q.columns:
        return None
    sts = q.dropna(subset=['Stimulus_Type'])
    if sts.empty:
        return None

    stim_colors = {'Text-based': C['blue'], 'Visual-based': C['teal'], 'Map-based': C['orange']}
    types = sorted(sts['Stimulus_Type'].astype(str).str.strip().unique().tolist())
    labels, vals, colors, hover_texts = [], [], [], []
    for st in types:
        st_qs  = q[q['Stimulus_Type'].astype(str).str.strip() == st]['Q_Num'].tolist()
        avg    = float(np.mean([df[f'Q{qn}'].mean() for qn in st_qs]))
        q_list = ', '.join(str(qn) for qn in sorted(st_qs))
        labels.append(st)
        vals.append(avg)
        colors.append(stim_colors.get(st, C['gray']))
        hover_texts.append(f'Questions: {q_list}')
    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=colors,
        marker_line_color='white', marker_line_width=1.5,
        text=[f'{v:.0%}' for v in vals], textposition='outside', cliponaxis=False,
        hovertext=hover_texts, hoverinfo='text', showlegend=False, width=0.4,
    ))
    fig.update_layout(**LAYOUT,
        title='Performance by Stimulus Type', height=380,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=13)),
        yaxis=dict(tickformat='.0%', title='Average % Correct', tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], range=[0, 1.12],
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    return fig


# ── Written response & scale score builders ───────────────────────────────────
ESSAY_COLORS = {0: '#B83246', 0.5: '#CF4556', 1: '#E0625F', 1.5: '#EB8058', 2: '#F2A541',
                2.5: '#F6C56F', 3: '#8ED1C0', 3.5: '#5BBFA9', 4: '#35AE97', 4.5: '#22917B', 5: '#16735F'}
LEVEL_COLORS = {1: C['red'], 2: C['orange'], 3: C['teal'], 4: C['green'], 5: '#145C4A'}


def _kpi(value, label, color, hint=''):
    hint_html = f'<div class="kpi-hint">{hint}</div>' if hint else ''
    return (f'<div class="kpi" style="--accent:{color}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{hint_html}</div>')


def _wrap(text, width=70):
    words, lines, cur = str(text).split(), [], ''
    for w in words:
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f'{cur} {w}'.strip()
    return '<br>'.join(lines + ([cur] if cur else []))


def _essay_color(v):
    return ESSAY_COLORS.get(float(v), C['gray'])


def _essay_status_color(avg):
    return C['teal'] if avg >= ESSAY_TARGET else (C['orange'] if avg >= ESSAY_TARGET - 1 else C['red'])


def _written_pool(df, items):
    """Students with at least one written score entered."""
    cols = [it['col'] for it in items if it['entered']]
    return df[df[cols].notna().any(axis=1)] if cols else df.iloc[0:0]


def _views(df, show_classes):
    return ['Overall'] + (sorted(df['Class'].unique().tolist())
                          if show_classes and df['Class'].nunique() > 1 else [])


def _with_views(fig, cfgs, traces=None):
    """Chart plus Overall / per-class buttons. cfgs = {view: Plotly.restyle update}."""
    chart_div = to_div(fig)
    if len(cfgs) <= 1:
        return chart_div
    m = _re.search(r'<div id="([^"]+)"', chart_div)
    fig_id = m.group(1) if m else 'plotly-view'
    ctr_id = 'vw-' + fig_id[-8:]
    btns = ''.join(
        f'<button class="q-btn{" q-btn-active" if i == 0 else ""}" '
        f'data-view="{v}" data-ctr="{ctr_id}">{v}</button>'
        for i, v in enumerate(cfgs))
    tr_js = json.dumps(traces) if traces is not None else 'undefined'
    return f'''<div class="q-btns"><span class="btn-label">View</span>{btns}</div>
{chart_div}
<script>
(function(){{
  var cfgs={json.dumps(cfgs)};
  function attach(){{
    var el=document.getElementById('{fig_id}');
    if(!el){{setTimeout(attach,80);return;}}
    document.querySelectorAll('.q-btn[data-ctr="{ctr_id}"]').forEach(function(btn){{
      btn.addEventListener('click',function(){{
        Plotly.restyle(el, cfgs[this.dataset.view], {tr_js});
        document.querySelectorAll('.q-btn[data-ctr="{ctr_id}"]').forEach(function(b){{
          b.classList.toggle('q-btn-active', b===btn);
        }});
      }});
    }});
  }}
  attach();
}})();
</script>'''


def written_kpi_html(df, items):
    pool = _written_pool(df, items)
    if pool.empty:
        return ''
    tiles, groups = [], []
    for it in items:
        if it['entered'] and it['group'] not in groups:
            groups.append(it['group'])
    for g in groups:
        g_items = [it for it in items if it['group'] == g and it['entered']]
        if g_items[0]['kind'] == 'short':
            credits = pool[[it['col'] for it in g_items]].fillna(0).sum(axis=1)
            n   = len(g_items)
            avg = credits.mean()
            tiles.append(_kpi(f'{avg:.1f}<small> / {n}</small>', g, pct_color(avg / n),
                              f'{avg / n:.0%} of credits earned · 1 credit each'))
        else:
            scores = pd.concat([pool[it['col']].fillna(0) for it in g_items])
            avg    = scores.mean()
            lbl    = f'{g} (avg per essay)' if len(g_items) > 1 else g
            tiles.append(_kpi(f'{avg:.1f}<small> / 5</small>', lbl, _essay_status_color(avg),
                              f'{(scores >= ESSAY_TARGET).mean():.0%} scored {ESSAY_TARGET} or higher'))
    return '<div class="kpi-row">' + ''.join(tiles) + '</div>'


def written_short_html(df, items, show_classes=True):
    """% of students earning the credit on each 1-credit question, grouped by set."""
    items = [it for it in items if it['kind'] == 'short' and it['entered']]
    if not items:
        return ''
    pool  = _written_pool(df, items)
    names = [f"{it['label']} · {it['tag']}" if it['tag'] else it['label'] for it in items]
    y     = [[it['set'] or it['group'] for it in items], names] if any(it['set'] for it in items) else names

    def stats(fdf):
        p = [float(fdf[it['col']].fillna(0).mean()) for it in items]
        hover = [f"<b>{it['label']}</b>{' · ' + it['tag'] if it['tag'] else ''}<br>"
                 f"{_wrap(it['question'])}<br><br>{pv:.0%} earned the credit "
                 f"(n={len(fdf)}; {int(fdf[it['col']].isna().sum())} blank)"
                 for it, pv in zip(items, p)]
        return p, hover

    views = _views(pool, show_classes)
    data  = {v: stats(pool if v == 'Overall' else pool[pool['Class'] == v]) for v in views}
    p0, h0 = data['Overall']
    fig = go.Figure(go.Bar(
        y=y, x=p0, orientation='h', marker_color=[pct_color(v) for v in p0], width=0.62,
        text=[f'{v:.0%}' for v in p0], textposition='outside', cliponaxis=False,
        textfont=dict(size=12, color=C['navy']),
        hovertext=h0, hoverinfo='text', showlegend=False,
    ))
    add_color_legend(fig)
    fig.update_layout(**LAYOUT,
        title='Short Answers — % of Students Earning the Credit',
        height=max(260, 110 + 44 * len(items)),
        legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center', yanchor='top', font=dict(size=11)),
        xaxis=dict(tickformat='.0%', tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], range=[0, 1.12],
                   showgrid=True, zeroline=False),
        yaxis=dict(autorange='reversed', showgrid=False, tickfont=dict(size=12), ticksuffix='  '),
    )
    fig.update_layout(margin=dict(l=20, r=30, t=50, b=60))
    cfgs = {v: {'x': [data[v][0]], 'marker.color': [[pct_color(x) for x in data[v][0]]],
                'text': [[f'{x:.0%}' for x in data[v][0]]], 'hovertext': [data[v][1]]} for v in views}
    return _with_views(fig, cfgs, traces=[0])


def written_essay_html(df, items, show_classes=True):
    """Rubric-score distribution (0–5, with half points) for each essay."""
    items = [it for it in items if it['kind'] == 'essay' and it['entered']]
    if not items:
        return ''
    pool   = _written_pool(df, items)
    scores = sorted(set(float(v) for it in items for v in pool[it['col']].dropna().unique())
                    | {0.0, 1.0, 2.0, 3.0, 4.0, 5.0})

    def stats(fdf):
        n = max(len(fdf), 1)
        shares = {s: [float((fdf[it['col']] == s).sum() / n) for it in items] for s in scores}
        blank  = [float(fdf[it['col']].isna().sum() / n) for it in items]
        labels = [f"{it['label']}  ·  avg {fdf[it['col']].fillna(0).mean():.1f}" for it in items]
        return shares, blank, labels

    views = _views(pool, show_classes)
    data  = {v: stats(pool if v == 'Overall' else pool[pool['Class'] == v]) for v in views}
    shares, blank, labels = data['Overall']

    def _txt(xs):
        return [f'{x:.0%}' if x >= 0.06 else '' for x in xs]

    fig = go.Figure()
    for s in scores:
        dark = s <= 2 or s >= 4
        fig.add_trace(go.Bar(
            y=labels, x=shares[s], orientation='h', name=f'{s:g}',
            marker_color=_essay_color(s), marker_line_color='white', marker_line_width=1,
            text=_txt(shares[s]), textposition='inside', insidetextanchor='middle',
            textfont=dict(color='white' if dark else C['navy'], size=11),
            hovertemplate=f'Score {s:g}: %{{x:.0%}}<extra></extra>',
        ))
    fig.add_trace(go.Bar(
        y=labels, x=blank, orientation='h', name='Blank', marker_color=C['lgray'],
        marker_line_color='white', marker_line_width=1, text=_txt(blank),
        textposition='inside', insidetextanchor='middle', textfont=dict(color=C['navy'], size=11),
        hovertemplate='No score entered: %{x:.0%}<extra></extra>',
    ))
    fig.update_layout(**LAYOUT,
        title='Essays — Rubric Score Distribution (0–5)', barmode='stack',
        height=max(240, 150 + 58 * len(items)),
        legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center', yanchor='top',
                    title=dict(text='Score  ', font=dict(size=11)), traceorder='normal', font=dict(size=11)),
        xaxis=dict(tickformat='.0%', range=[0, 1], showgrid=True, zeroline=False),
        yaxis=dict(autorange='reversed', showgrid=False, tickfont=dict(size=12), ticksuffix='  '),
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=60))
    cfgs = {v: {'x': [data[v][0][s] for s in scores] + [data[v][1]],
                'y': [data[v][2]] * (len(scores) + 1),
                'text': [_txt(data[v][0][s]) for s in scores] + [_txt(data[v][1])]} for v in views}
    html = _with_views(fig, cfgs)

    classes = _views(pool, show_classes)[1:]
    if classes:
        cfig = go.Figure()
        for cls in classes:
            cdf = pool[pool['Class'] == cls]
            avgs = [float(cdf[it['col']].fillna(0).mean()) for it in items]
            cfig.add_trace(go.Bar(
                x=[it['label'] for it in items], y=avgs, name=f'{cls} (n={len(cdf)})',
                marker_color=CLASS_COLORS.get(cls, C['blue']),
                text=[f'{a:.1f}' for a in avgs], textposition='outside', cliponaxis=False,
                textfont=dict(size=11, color=C['navy']),
                hovertemplate=f'{cls}: %{{y:.2f}} / 5<extra></extra>',
            ))
        cfig.add_hline(y=ESSAY_TARGET, line_dash='dot', line_color=C['navy'], opacity=0.45,
                       annotation_text=f'Level {ESSAY_TARGET}', annotation_position='top right',
                       annotation_font=dict(size=10, color=C['navy']))
        cfig.update_layout(**LAYOUT, title='Average Essay Score by Class', barmode='group', height=340,
            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center', yanchor='top'),
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=12)),
            yaxis=dict(title='Average rubric score', tickvals=[0, 1, 2, 3, 4, 5], range=[0, 5.6],
                       showgrid=True, zeroline=False),
        )
        html += to_div(cfig)
    return html


def written_section_html(df, items, show_classes=True):
    entered = [it for it in items if it['entered']]
    if not entered:
        return ('<div class="callout"><div class="callout-title">No written scores yet</div>'
                '<p class="muted" style="margin:0">The written-response questions are listed in the data '
                'file, but no scores have been entered.</p></div>')
    pool = _written_pool(df, items)
    html = ''
    if len(pool) < len(df):
        scored = sorted(pool['Class'].unique().tolist())
        html += (f'<p class="note" style="margin-top:14px">Written scores are entered for <b>{len(pool)} of '
                 f'{len(df)}</b> students' + (f' ({", ".join(scored)})' if len(scored) <= 3 else '')
                 + ' — the results below cover only those students.</p>')
    html += written_kpi_html(df, items)
    short = written_short_html(df, items, show_classes)
    essay = written_essay_html(df, items, show_classes)
    if short:
        html += '<h3 class="sub-head">Short answers <span>1 credit each</span></h3>' + short
    if essay:
        html += ('<h3 class="sub-head">Essays <span>scored 0–5 on the NYSED rubric · '
                 'half points are the average of two raters</span></h3>' + essay)
    missing = [it['label'] for it in items if not it['entered']]
    if missing:
        html += f'<p class="note">No scores entered yet for: {", ".join(missing)}.</p>'
    return html


def written_subtitle(items):
    parts = []
    for g in dict.fromkeys(it['group'] for it in items):
        g_items = [it for it in items if it['group'] == g]
        it = g_items[0]
        if it['kind'] == 'short':
            parts.append(f"{g} ({it['part']}) are worth 1 credit each")
        elif len(g_items) > 1:
            parts.append(f"the {g} ({it['part']}) are each scored 0–5 on a rubric")
        else:
            parts.append(f"the {g} ({it['part']}) is scored 0–5 on a rubric")
    text = '; '.join(parts)
    return (text[0].upper() + text[1:] + '. Blank answers count as 0; students with no written '
            'scores at all are left out of this section.')


# ── Regents scale score ───────────────────────────────────────────────────────
def _level_of(score, levels):
    for n, lo, hi in levels:
        if lo <= score <= hi:
            return n
    return None


def _leverage_html(sdf, conv, items, subject):
    """Project pass rates under simple what-if scenarios using the exam's conversion chart."""
    lookup, levels, raw_max = conv
    essay = next((it for it in items if it['part'] in ('Part III', 'Part IIIB') and it['entered']), None)
    if essay is None:
        return ''
    others = [it for it in items if it is not essay]
    raw    = sdf['MC_Score'] + sum((sdf[it['col']].fillna(0) for it in others), pd.Series(0, index=sdf.index))
    ess    = sdf[essay['col']].fillna(0)
    final  = pd.to_numeric(sdf['Final'], errors='coerce')

    def scale(r_series, e_series):
        return pd.Series([lookup.get((int(min(round(r), raw_max)), float(min(e, 5))), np.nan)
                          for r, e in zip(r_series, e_series)], index=sdf.index)

    base = scale(raw, ess)
    if (base == final).mean() < 0.9:   # data doesn't reproduce the chart — skip projections
        return ''
    cut = next((lo for n, lo, _ in levels if n == 3), 65)
    base_pass = (base >= cut)
    extra = ('two more multiple-choice or short-answer credits, or one rubric level higher on both SEQs'
             if subject == 'US' else 'two more multiple-choice or CRQ credits')
    rows = []
    for name, detail, r2, e2 in [
        (f'Every {essay["label"]} one rubric level higher', 'e.g. a 2 becomes a 3', raw, (ess + 1).clip(upper=5)),
        ('Two more raw credits per student', extra, (raw + 2).clip(upper=raw_max), ess),
    ]:
        proj = scale(r2, e2)
        newly = int(((proj >= cut) & ~base_pass).sum())
        rows.append(f'<tr><td><b>{name}</b><span>{detail}</span></td>'
                    f'<td>{base_pass.mean():.0%} → <b>{(proj >= cut).mean():.0%}</b></td>'
                    f'<td>+{newly}</td><td>+{(proj - base).mean():.1f}</td></tr>')
    return (f'<h3 class="sub-head">What moves the scale score <span>projected with this exam\'s '
            f'conversion chart</span></h3>'
            f'<div class="scroll-x"><table class="lev-table"><thead><tr><th>If…</th><th>Passing ({cut}+)</th>'
            f'<th>Newly passing</th><th>Avg scale gain</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            f'<p class="note">The essay is the column of the conversion chart, so each rubric level is worth '
            f'several scale points; each raw credit is worth about two or three.</p>')


def scale_section_html(df, conv, items, subject, show_classes=True):
    lookup, levels, raw_max = conv
    final = pd.to_numeric(df['Final'], errors='coerce')
    sdf, f = df[final.notna()], final.dropna()
    if f.empty:
        return ''
    cut     = next((lo for n, lo, _ in levels if n == 3), 65)
    mastery = next((lo for n, lo, _ in levels if n == 5), 85)
    near    = int(((f >= cut - 5) & (f < cut)).sum())
    missing = [it['label'] for it in items if not it['entered']]
    warn = ('<div class="callout callout-warn"><div class="callout-title">Written scores not entered yet</div>'
            f'<p style="margin:0">No scores are in the data for {", ".join(missing)}, so these scale scores '
            'count only what has been entered. Treat them as a floor, not final results.</p></div>'
            if missing else '')
    kpis = warn + ('<div class="kpi-row">'
            + _kpi(f'{f.mean():.0f}', 'Average scale score', LEVEL_COLORS.get(_level_of(round(f.mean()), levels), C['blue']),
                   f'median {f.median():.0f}')
            + _kpi(f'{(f >= cut).mean():.0%}', f'Passing ({cut}+)', C['teal'] if (f >= cut).mean() >= 0.65 else C['orange'],
                   f'{int((f >= cut).sum())} of {len(f)} students')
            + _kpi(f'{(f >= mastery).mean():.0%}', f'Mastery ({mastery}+)', LEVEL_COLORS[5],
                   f'{int((f >= mastery).sum())} students')
            + _kpi(str(near), 'Within 5 of passing', C['orange'], f'scored {cut - 5}–{cut - 1}')
            + '</div>')

    bins = list(range(0, 100, 5))
    counts = [int(((f >= b) & (f < b + 5 if b < 95 else f <= 100)).sum()) for b in bins]
    fig = go.Figure(go.Bar(
        x=[b + 2.5 for b in bins], y=counts, width=4.4,
        marker_color=[LEVEL_COLORS.get(_level_of(b + 2, levels), C['gray']) for b in bins],
        hovertext=[f'{b}–{b + 4 if b < 95 else 100}: {c} students' for b, c in zip(bins, counts)],
        hoverinfo='text', showlegend=False,
    ))
    for n, lo, hi in levels:
        fig.add_vrect(x0=lo, x1=hi + 1, fillcolor=LEVEL_COLORS[n], opacity=0.06, line_width=0,
                      annotation_text=f'Level {n}', annotation_position='top',
                      annotation_font=dict(size=11, color=LEVEL_COLORS[n]))
    fig.add_vline(x=cut, line_dash='dot', line_color=C['navy'], opacity=0.5, line_width=1.5)
    fig.update_layout(**LAYOUT, title='Scale Score Distribution', height=340,
        xaxis=dict(title='Scale score', range=[0, 101], tickvals=list(range(0, 101, 10)),
                   showgrid=False, zeroline=False),
        yaxis=dict(title='Number of students', showgrid=True, zeroline=False),
    )
    html = kpis + to_div(fig)

    classes = _views(sdf, show_classes)[1:]
    if classes:
        rates = [float((pd.to_numeric(sdf[sdf['Class'] == c]['Final'], errors='coerce') >= cut).mean())
                 for c in classes]
        ns = [int((sdf['Class'] == c).sum()) for c in classes]
        cfig = go.Figure(go.Bar(
            x=classes, y=rates, marker_color=[CLASS_COLORS.get(c, C['blue']) for c in classes],
            text=[f'{r:.0%}' for r in rates], textposition='outside', cliponaxis=False,
            textfont=dict(size=12, color=C['navy']), width=0.5,
            hovertext=[f'{c}: {r:.0%} passing (n={n})' for c, r, n in zip(classes, rates, ns)],
            hoverinfo='text', showlegend=False,
        ))
        cfig.update_layout(**LAYOUT, title=f'Passing Rate ({cut}+) by Class', height=300,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(tickformat='.0%', tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], range=[0, 1.12],
                       showgrid=True, zeroline=False),
        )
        html += to_div(cfig)
    return html + _leverage_html(sdf, conv, items, subject)


# ── HTML helpers ──────────────────────────────────────────────────────────────
def kpi_html(df, nq):
    mean_s  = df['MC_Score'].mean()
    thresh  = round(nq * 0.67)
    pct_thr = (df['MC_Score'] >= thresh).mean()
    n       = len(df)
    n_blank = (df['MC_Blank'] > 0).sum()


    return ('<div class="kpi-row">'
            + _kpi(f'{mean_s:.1f}<small> / {nq}</small>', 'Average score', score_color(round(mean_s), nq),
                   f'{mean_s / nq:.0%} correct on average')
            + _kpi(f'{pct_thr:.0%}', f'Scored {thresh} or more',
                   C['teal'] if pct_thr >= 0.55 else C['orange'], f'{thresh / nq:.0%}+ of questions correct')
            + _kpi(str(n), 'Students tested', C['blue'], f'{df["Class"].nunique()} classes')
            + _kpi(str(n_blank), 'Left a question blank',
                   C['orange'] if n_blank > n * 0.10 else C['teal'], f'{n_blank / n:.0%} of students')
            + '</div>')


def subgroup_notes_html(df):
    lines = []
    for flag, name in [('Is_ELL', 'ELL'), ('Is_IEP', 'IEP'), (None, 'Both ELL & IEP')]:
        mask = (df['Is_ELL'] & df['Is_IEP']) if flag is None else df[flag]
        n_g  = int(mask.sum())
        if n_g > 0:
            avg_g = df[mask]['MC_Score'].mean()
            avg_o = df[~mask]['MC_Score'].mean()
            gap   = avg_o - avg_g
            note  = f'{gap:+.1f} pt gap' if abs(gap) >= 0.5 else 'minimal gap'
            lines.append(
                f'<li><b>{name}</b> <span class="muted">(n={n_g})</span> — averaged {avg_g:.1f} '
                f'vs. {avg_o:.1f} for other students <span class="pill">{note}</span></li>')
    if not lines:
        return ('<div class="callout"><div class="callout-title">Subgroup notes</div>'
                '<p class="muted" style="margin:0">No students are flagged ELL or IEP in this data.</p></div>')
    return (f'<div class="callout"><div class="callout-title">Subgroup notes</div>'
            f'<ul>{"".join(lines)}</ul></div>')


def takeaways_html(df, q, std_name, nq, show_classes=True):
    container_id = 'tk-' + uuid.uuid4().hex[:8]
    views = ['Overall'] + (sorted(df['Class'].unique().tolist()) if show_classes else [])

    def std_items_for(fdf):
        std_groups = defaultdict(list)
        for _, row in q.iterrows():
            qn   = int(row['Q_Num'])
            raw  = row.get('Content_Std', '')
            code = '' if (raw is None or str(raw).strip().lower() in ('nan', 'none', '')) \
                   else str(raw).strip()
            std_groups[code or f'Q{qn}'].append(qn)
        items = []
        for code, qns in std_groups.items():
            avg_p = sum(fdf[f'Q{qn}'].mean() for qn in qns) / len(qns)
            label = std_name(code, 50) if code and not code.startswith('Q') else ''
            q_label = ', '.join(f'Q{n}' for n in sorted(qns))
            items.append((q_label, avg_p, label))
        return items

    def panel(title, subtitle, items, color, bg):
        rows = ''.join(
            f'<li><div class="tk-text">{lbl or ql}'
            + (f'<span class="tk-q">{ql}</span>' if lbl else '')
            + f'</div><div class="tk-pct" style="color:{color}">{p:.0%}</div>'
            f'<div class="tk-bar"><i style="width:{p:.0%};background:{color}"></i></div></li>'
            for ql, p, lbl in items
        ) or '<li class="tk-empty">Nothing in this range</li>'
        return (f'<div class="tk-panel" style="--accent:{color};--tint:{bg}">'
                f'<div class="tk-head"><span class="tk-title">{title}</span>'
                f'<span class="tk-count">{len(items)}</span></div>'
                f'<div class="tk-sub">{subtitle}</div><ul>{rows}</ul></div>')

    def panels_html(fdf):
        items   = std_items_for(fdf)
        reteach = sorted([(ql,p,lbl) for ql,p,lbl in items if p < 0.50],     key=lambda x: x[1])
        review  = sorted([(ql,p,lbl) for ql,p,lbl in items if 0.50<=p<0.70], key=lambda x: x[1])
        strong  = sorted([(ql,p,lbl) for ql,p,lbl in items if p >= 0.70],    key=lambda x: -x[1])
        return ('<div class="tk-grid">'
                + panel('Reteach', 'Below 50% correct', reteach, C['red'],    '#FDF1F2')
                + panel('Review',  '50–70% correct',    review,  C['orange'], '#FEF7EC')
                + panel('Strong',  '70% or more correct', strong, C['teal'],  '#EEF9F6')
                + '</div>')

    view_divs = ''
    for i, view in enumerate(views):
        fdf     = df if view == 'Overall' else df[df['Class'] == view]
        display = 'block' if i == 0 else 'none'
        view_divs += (f'<div id="{container_id}-{i}" style="display:{display}">'
                      + panels_html(fdf) + '</div>')

    btns = ''.join(
        f'<button class="q-btn{"  q-btn-active" if i == 0 else ""}" '
        f'data-vi="{i}" data-ctr="{container_id}">{view}</button>'
        for i, view in enumerate(views)
    )

    if len(views) == 1:
        return panels_html(df)

    return f'''<div class="q-btns"><span class="btn-label">View</span>{btns}</div>
{view_divs}
<script>
(function(){{
  var n={len(views)};
  document.querySelectorAll('.q-btn[data-ctr="{container_id}"]').forEach(function(btn){{
    btn.addEventListener('click',function(){{
      var vi=parseInt(this.dataset.vi);
      for(var i=0;i<n;i++){{
        document.getElementById('{container_id}-'+i).style.display=(i===vi?'block':'none');
      }}
      document.querySelectorAll('.q-btn[data-ctr="{container_id}"]').forEach(function(b){{
        b.classList.toggle('q-btn-active',b===btn);
      }});
    }});
  }});
}})();
</script>'''


# ── Report assembly ───────────────────────────────────────────────────────────
def build_html(df, q, written, subject, conv, std_name, nq, title,
               home_link='../../index.html', show_classes=True):
    today = date.today().strftime('%B %d, %Y')

    figs = {
        'hist':          fig_histogram(df, nq),
        'cp_bar':        fig_class_bar(df, nq)        if show_classes else None,
        'cp_dist':       fig_class_dist(df, nq)       if show_classes else None,
        'cp_grade':      fig_grade_bar(df, nq)        if show_classes else None,
        'grp':           fig_groups(df, nq),
        'box':           fig_boxplot(df, nq)           if show_classes else None,
        'oi':            fig_oi(df, q, nq),
        'task_model':    fig_task_model(df, q, nq),
        'skill':         fig_skill(df, q, nq),
        'stimulus_type': fig_stimulus_type(df, q, nq),
    }

    has_written = bool(written)
    has_scale   = (conv is not None and 'Final' in df.columns
                   and pd.to_numeric(df['Final'], errors='coerce').notna().any())

    def _opt(fig):
        return to_div(fig) if fig is not None else ''

    # (anchor, nav label, section title, subtitle, content) — numbered in order below
    blocks = [
        ('overview', 'Overview', 'At a Glance',
         'Multiple-choice results for every student who took the exam. '
         'Bars are shaded by performance band; the dotted line marks the average.',
         kpi_html(df, nq) + to_div(figs['hist'])),
    ]

    if has_scale:
        blocks.append(('scale', 'Scale Score', 'Regents Scale Scores',
            'Final scale scores from the conversion chart, which combines raw credits (multiple choice '
            'and short answers) with the essay score. Shading shows performance Levels 1–5.',
            scale_section_html(df, conv, written, subject, show_classes)))

    if show_classes:
        blocks.append(('class-period', 'Classes', 'Class & Period Performance',
            'Average score for each period, and how scores spread out within each class.',
            (
                '<div class="chart-grid">'
                + to_div(figs['cp_bar']) + to_div(figs['cp_grade'])
                + '</div>'
                + to_div(figs['cp_dist'])
            ) if figs['cp_grade'] is not None else (
                '<div class="chart-grid">'
                + to_div(figs['cp_bar']) + to_div(figs['cp_dist'])
                + '</div>'
            )))

    blocks += [
        ('difficulty', 'Difficulty', 'Which Questions Were Hardest?',
         'Percent of students who answered each question correctly. '
         'Hover a cell to see the question, its standard, and the correct answer.',
         f'<div class="scroll-x"><div style="min-width:{max(520, nq * 34)}px">'
         + difficulty_html(df, q, std_name, nq, show_classes) + '</div></div>'),

        ('distractor', 'Answer Choices', 'What Did Students Select?',
         'How students split across the four choices on each question. Pick a question number to switch.',
         distractor_html(df, q, nq)),
    ]

    if has_written:
        blocks.append(('written', 'Written', 'Written Response',
            written_subtitle(written),
            written_section_html(df, written, show_classes)))

    blocks += [
        ('groups', 'Groups', 'Student Group Comparison',
         'Average score for general education, ELL, and IEP students, compared with the overall average.',
         to_div(figs['grp']) + subgroup_notes_html(df)),

        ('takeaways', 'Takeaways', 'Key Takeaways',
         'Content standards sorted by how many students answered their questions correctly, to guide reteaching.',
         takeaways_html(df, q, std_name, nq, show_classes)),
    ]

    sections = [section(t, a, c, subtitle=s, num=i + 1) for i, (a, _, t, s, c) in enumerate(blocks)]
    sections.append(section('Appendix', 'appendix',
        (_opt(figs['box']) if show_classes else '')
        + '<div class="divider"></div>'
        + '<h3 class="sub-head" style="border-top:0;padding-top:0;margin-top:6px">Question quality '
          '<span>difficulty vs. point-biserial</span></h3>'
        + pb_guide_html(len(df))
        + scatter_html(df, q, std_name, nq, show_classes)
        + pb_flags_html(df, q, std_name, nq)
        + '<div class="divider"></div>'
        + '<div class="chart-grid-2">'
        + _opt(figs['oi'])
        + _opt(figs['stimulus_type'])
        + _opt(figs['task_model'])
        + _opt(figs['skill'])
        + '</div>',
        subtitle='Item statistics and results broken down by question type.', num='+'))

    nav_links = ''.join(f'<a href="#{a}">{lbl}</a>' for a, lbl, *_ in blocks) \
        + '<a href="#appendix">Appendix</a>'

    n_classes = df['Class'].nunique()
    meta = [f'{len(df)} students', f'{nq} multiple-choice questions']
    if has_written:
        meta.append(f'{len(written)} written')
    if show_classes:
        meta.append(f'{n_classes} class{"es" if n_classes != 1 else ""}')
    meta_html = ''.join(f'<span>{m}</span>' for m in meta)

    body = '\n'.join(sections)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="icon" type="image/webp" href="https://www.westinghousehs.org/images/index-page/gwhs-logo.webp">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
  <style>
    :root {{
      --navy: {C["navy"]}; --ink: #2B3448; --muted: #667085; --line: #E6E8EC;
      --bg: #F5F6F8; --card: #FFFFFF; --teal: {C["teal"]}; --accent-ink: #1F7F6D;
      --radius: 16px; --nav-h: 52px;
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      font-family: {FONT}; font-feature-settings: 'cv11', 'tnum';
      background: var(--bg); color: var(--ink); margin: 0; line-height: 1.55;
      -webkit-font-smoothing: antialiased;
    }}

    /* ── Hero ── */
    .hero {{
      background: radial-gradient(1200px 300px at 85% -40%, rgba(53,174,151,.35), transparent 60%),
                  linear-gradient(135deg, #1F2A44 0%, #273557 100%);
      color: white; padding: 22px 32px 30px;
    }}
    .hero-inner {{ max-width: 1080px; margin: 0 auto; }}
    .home-link {{
      display: inline-flex; align-items: center; gap: 6px; color: rgba(255,255,255,.75);
      text-decoration: none; font-size: .82rem; font-weight: 500; margin-bottom: 18px;
    }}
    .home-link:hover {{ color: white; }}
    .eyebrow {{
      font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
      color: #8FE0CF; margin: 0 0 6px;
    }}
    .hero h1 {{ margin: 0; font-size: clamp(1.5rem, 3.2vw, 2.1rem); font-weight: 800; letter-spacing: -.02em; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .meta span {{
      font-size: .8rem; padding: 4px 11px; border-radius: 999px;
      background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.14);
    }}

    /* ── Sticky nav ── */
    nav {{
      position: sticky; top: 0; z-index: 100; height: var(--nav-h);
      background: rgba(255,255,255,.88); backdrop-filter: saturate(180%) blur(10px);
      -webkit-backdrop-filter: saturate(180%) blur(10px);
      border-bottom: 1px solid var(--line);
    }}
    .nav-inner {{
      max-width: 1080px; height: 100%; margin: 0 auto; padding: 0 24px;
      display: flex; align-items: center; gap: 4px; overflow-x: auto; scrollbar-width: none;
    }}
    .nav-inner::-webkit-scrollbar {{ display: none; }}
    nav a {{
      color: var(--muted); text-decoration: none; font-size: .85rem; font-weight: 500;
      padding: 6px 12px; border-radius: 8px; white-space: nowrap; transition: background .15s, color .15s;
    }}
    nav a:hover {{ color: var(--navy); background: #F0F2F5; }}
    nav a.active {{ color: var(--accent-ink); background: #E6F5F1; font-weight: 600; }}

    /* ── Layout ── */
    main {{
      max-width: 1080px; margin: 0 auto; padding: 28px 24px 72px;
      display: flex; flex-direction: column; gap: 24px;
    }}
    section {{
      background: var(--card); border-radius: var(--radius); padding: 26px 30px 24px;
      border: 1px solid var(--line); box-shadow: 0 1px 2px rgba(16,24,40,.04);
      scroll-margin-top: calc(var(--nav-h) + 16px);
    }}
    .sec-head {{
      display: flex; align-items: flex-start; gap: 14px;
      padding-bottom: 16px; margin-bottom: 8px; border-bottom: 1px solid var(--line);
    }}
    .sec-num {{
      flex: none; width: 32px; height: 32px; border-radius: 10px; margin-top: 1px;
      display: grid; place-items: center; font-size: .9rem; font-weight: 700;
      background: #E6F5F1; color: var(--accent-ink);
    }}
    section h2 {{ color: var(--navy); font-size: 1.25rem; font-weight: 700; letter-spacing: -.01em; margin: 0; }}
    section p.subtitle {{ color: var(--muted); font-size: .9rem; margin: 3px 0 0; max-width: 72ch; }}
    .divider {{ height: 1px; background: var(--line); margin: 22px 0 10px; }}
    .note, .muted {{ color: var(--muted); }}
    .note {{ font-size: .85rem; margin: 4px 0 0; }}

    .plotly-graph-div {{ width: 100% !important; }}
    .chart-grid .plotly-graph-div {{ height: 320px !important; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
    .chart-grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
    .scroll-x {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
    .scroll-x > * {{ min-width: 520px; }}

    /* ── KPI tiles ── */
    .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 18px 0 10px; }}
    .kpi {{
      position: relative; border: 1px solid var(--line); border-radius: 12px;
      padding: 16px 18px 14px 20px; background: white; overflow: hidden;
    }}
    .kpi::before {{ content: ''; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--accent); }}
    .kpi-label {{ font-size: .78rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
    .kpi-value {{ font-size: 2rem; font-weight: 800; color: var(--navy); line-height: 1.15; margin-top: 4px; letter-spacing: -.02em; }}
    .kpi-value small {{ font-size: 1rem; font-weight: 600; color: var(--muted); }}
    .kpi-hint {{ font-size: .8rem; color: var(--muted); margin-top: 2px; }}

    /* ── Toggle buttons ── */
    .q-btns {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 14px 0 6px; }}
    .q-btns-center {{ justify-content: flex-start; }}
    .btn-label {{ font-size: .75rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-right: 4px; }}
    .q-btn {{
      font: inherit; font-size: .82rem; font-weight: 500; min-width: 34px; padding: 5px 11px;
      border-radius: 8px; border: 1px solid var(--line); background: white; cursor: pointer;
      color: var(--ink); transition: background .12s, border-color .12s, color .12s;
    }}
    .q-btn:hover {{ background: #F0F2F5; border-color: #D0D5DD; }}
    .q-btn:focus-visible {{ outline: 2px solid var(--teal); outline-offset: 2px; }}
    .q-btn-active {{ background: var(--navy) !important; color: white !important; border-color: var(--navy) !important; }}

    /* ── Distractor question card ── */
    .q-card {{
      display: flex; gap: 14px; align-items: flex-start; margin: 12px 0 6px;
      background: #F8F9FB; border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px;
    }}
    .q-card-num {{
      flex: none; font-weight: 700; font-size: .85rem; color: white; background: var(--navy);
      border-radius: 8px; padding: 3px 9px; margin-top: 1px;
    }}
    .q-card p {{ margin: 0; font-size: .98rem; color: var(--navy); line-height: 1.5; }}
    .dis-legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: .8rem; color: var(--muted); margin: 10px 0 0; }}
    .dis-legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}

    /* ── Callout ── */
    .callout {{ margin-top: 16px; background: #F4F7FE; border: 1px solid #DCE5FB; border-radius: 12px; padding: 14px 18px; }}
    .callout-warn {{ background: #FEF7EC; border-color: #F7DDB0; margin: 16px 0 4px; }}
    .callout-title {{ font-weight: 700; color: var(--navy); margin-bottom: 6px; font-size: .92rem; }}
    .callout ul {{ margin: 0; padding-left: 18px; font-size: .92rem; }}
    .callout li + li {{ margin-top: 4px; }}
    .pb-guide {{ display: grid; grid-template-columns: 1.5fr 1fr; gap: 20px; margin: 10px 0 14px;
      background: #F4F7FE; border: 1px solid #DCE5FB; border-radius: 12px; padding: 16px 20px; }}
    .pb-guide p {{ margin: 6px 0 8px; font-size: .92rem; }}
    .pb-guide ul {{ margin: 0; padding-left: 18px; font-size: .9rem; }}
    .pb-guide li + li {{ margin-top: 4px; }}
    .pb-bands {{ list-style: none; padding: 0 !important; display: flex; flex-direction: column; gap: 8px; }}
    .pb-bands li {{ display: grid; grid-template-columns: 86px 1fr; gap: 2px 10px; align-items: center; margin: 0 !important; }}
    .pb-chip {{ color: white; font-size: .75rem; font-weight: 700; padding: 2px 9px; border-radius: 999px; text-align: center; }}
    .pb-range {{ font-weight: 700; font-size: .85rem; color: var(--navy); }}
    .pb-mean {{ grid-column: 2; font-size: .8rem; color: var(--muted); line-height: 1.35; }}
    .pb-quad {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-bottom: 8px; }}
    .pb-quad div {{ font-size: .82rem; color: var(--muted); border: 1px solid var(--line); border-radius: 10px; padding: 9px 12px; }}
    .pb-quad b {{ display: block; color: var(--navy); font-size: .8rem; margin-bottom: 2px; }}
    .pb-flags {{ margin: 0; padding-left: 18px; font-size: .9rem; }}
    .pb-flags li + li {{ margin-top: 8px; }}
    .pb-reading {{ display: block; font-size: .82rem; color: var(--muted); }}
    @media (max-width: 720px) {{ .pb-guide {{ grid-template-columns: 1fr; }} }}
    .pill {{ display: inline-block; font-size: .75rem; font-weight: 600; padding: 1px 8px; border-radius: 999px; background: white; border: 1px solid #DCE5FB; margin-left: 4px; }}

    /* ── Takeaways ── */
    .tk-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 12px; }}
    .tk-panel {{ border: 1px solid var(--line); border-top: 4px solid var(--accent); border-radius: 12px; padding: 14px 16px 10px; background: linear-gradient(var(--tint), white 90px); }}
    .tk-head {{ display: flex; align-items: center; justify-content: space-between; }}
    .tk-title {{ font-size: 1.05rem; font-weight: 800; color: var(--accent); letter-spacing: -.01em; }}
    .tk-count {{ font-size: .78rem; font-weight: 700; color: var(--accent); background: white; border: 1px solid var(--line); border-radius: 999px; padding: 1px 9px; }}
    .tk-sub {{ font-size: .8rem; color: var(--muted); margin: 2px 0 8px; }}
    .tk-panel ul {{ list-style: none; margin: 0; padding: 0; }}
    .tk-panel li {{ display: grid; grid-template-columns: 1fr auto; gap: 2px 12px; padding: 9px 0; border-top: 1px solid var(--line); }}
    .tk-text {{ font-size: .9rem; color: var(--navy); line-height: 1.35; }}
    .tk-q {{ display: block; font-size: .75rem; color: var(--muted); margin-top: 2px; }}
    .tk-pct {{ font-weight: 800; font-size: .95rem; text-align: right; }}
    .tk-bar {{ grid-column: 1 / -1; height: 4px; border-radius: 4px; background: #EEF0F3; overflow: hidden; }}
    .tk-bar i {{ display: block; height: 100%; border-radius: 4px; }}
    .tk-panel li.tk-empty {{ display: block; color: var(--muted); font-style: italic; font-size: .88rem; }}

    /* ── Sub-sections & projection table ── */
    h3.sub-head {{
      font-size: 1rem; font-weight: 700; color: var(--navy); margin: 26px 0 4px;
      padding-top: 18px; border-top: 1px solid var(--line);
    }}
    h3.sub-head span {{ font-size: .82rem; font-weight: 500; color: var(--muted); margin-left: 8px; }}
    .lev-table {{ width: 100%; border-collapse: collapse; font-size: .9rem; margin-top: 10px; min-width: 520px; }}
    .lev-table th {{ text-align: left; font-size: .75rem; font-weight: 600; color: var(--muted);
      text-transform: uppercase; letter-spacing: .05em; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
    .lev-table td {{ padding: 10px; border-bottom: 1px solid var(--line); color: var(--navy); vertical-align: top; }}
    .lev-table td span {{ display: block; font-size: .8rem; color: var(--muted); }}
    .lev-table td:not(:first-child), .lev-table th:not(:first-child) {{ text-align: right; white-space: nowrap; }}

    footer {{ text-align: center; padding: 8px 24px 40px; font-size: .8rem; color: #98A2B3; }}

    @media (max-width: 720px) {{
      .hero {{ padding: 16px 16px 22px; }}
      .nav-inner {{ padding: 0 12px; }}
      main {{ padding: 16px 12px 48px; gap: 16px; }}
      section {{ padding: 18px 16px 16px; border-radius: 12px; }}
      .chart-grid-2 {{ grid-template-columns: 1fr; }}
      .kpi-value {{ font-size: 1.6rem; }}
    }}
    @media print {{
      nav, .q-btns {{ display: none; }}
      body {{ background: white; }}
      .hero {{ background: none; color: var(--navy); padding: 0 0 12px; }}
      .home-link {{ display: none; }}
      .eyebrow {{ color: var(--accent-ink); }}
      .meta span {{ border-color: var(--line); background: none; }}
      section {{ break-inside: avoid; box-shadow: none; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-inner">
      <a class="home-link" href="{home_link}">← All reports</a>
      <p class="eyebrow">GWHS Social Studies · Assessment Report</p>
      <h1>{title}</h1>
      <div class="meta">{meta_html}</div>
    </div>
  </header>
  <nav><div class="nav-inner">{nav_links}</div></nav>
  <main>{body}</main>
  <footer>Generated {today} · George Westinghouse CTE High School</footer>
  <script>
    setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 50);
    setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 300);
    // Highlight the nav link for the section currently in view
    (function() {{
      var links = {{}};
      document.querySelectorAll('nav a[href^="#"]').forEach(function(a) {{ links[a.getAttribute('href').slice(1)] = a; }});
      if (!('IntersectionObserver' in window)) return;
      var obs = new IntersectionObserver(function(entries) {{
        entries.forEach(function(e) {{
          if (!e.isIntersecting) return;
          Object.values(links).forEach(function(a) {{ a.classList.remove('active'); }});
          var a = links[e.target.id];
          if (a) {{ a.classList.add('active'); var n = a.parentNode; n.scrollLeft = a.offsetLeft - (n.clientWidth - a.offsetWidth) / 2; }}
        }});
      }}, {{ rootMargin: '-30% 0px -60% 0px' }});
      document.querySelectorAll('main section[id]').forEach(function(s) {{ obs.observe(s); }});
    }})();
  </script>
</body>
</html>'''


# ── Entry point ───────────────────────────────────────────────────────────────
def run(config):
    """Generate a report from a config dict (keys listed in the module docstring)."""
    global CLASS_COLORS
    if 'class_colors' in config:
        CLASS_COLORS = {**CLASS_COLORS, **config['class_colors']}

    file         = config['file']
    title        = config['title']
    output       = config['output']
    home_link    = config.get('home_link', '../../index.html')
    show_classes = config.get('show_classes', True)

    print('Loading data...')
    df, q, written, subject, std_lookup, std_name, answer_cols, correct_cols, NQ = \
        load_data(file, config.get('subject'))
    conv = load_conversion(file)
    n_short = sum(1 for it in written if it['kind'] == 'short')
    n_essay = len(written) - n_short
    print(f'  {len(df)} students, {df["Class"].nunique()} classes, {NQ} MC, '
          f'{n_short} short-answer + {n_essay} essay ({subject or "subject unknown"})'
          + ('; conversion chart found' if conv else ''))

    print('Building report...')
    html = build_html(df, q, written, subject, conv, std_name, NQ, title, home_link, show_classes)

    with open(output, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = len(html.encode('utf-8')) / 1024
    print(f'Done → {output}  ({size_kb:.0f} KB)')
