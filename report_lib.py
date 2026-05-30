"""
Shared library for GWHS exam report generation.

Each exam's generate_report.py imports this and calls lib.run(config).

Config keys:
  file          – path to the Excel data file (relative to CWD or absolute)
  title         – report title shown in the header and <title>
  output        – path to write the HTML output
  home_link     – href for the ← Home nav link (default: '../../index.html')
  class_colors  – dict mapping class name → hex color (optional)
                  defaults to Bermejo=blue, Sourial=blue, Dushin=orange, Kovelsky=teal
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
    red='#E84855', orange='#F4A261', teal='#44BBA4', green='#06D6A0',
    blue='#3A86FF', navy='#2D3142', gray='#888888', lgray='#D8D8D8',
)

# Default class colors — overridden by config['class_colors'] in run()
CLASS_COLORS = {'Bermejo': C['blue'], 'Sourial': C['blue'], 'Dushin': C['orange'], 'Kovelsky': C['teal']}

LAYOUT = dict(
    plot_bgcolor='white', paper_bgcolor='white',
    font=dict(family='system-ui, sans-serif', size=12, color=C['navy']),
    hoverlabel=dict(bgcolor='white', font_size=12, font_family='system-ui, sans-serif'),
    margin=dict(l=20, r=20, t=50, b=20),
    autosize=True,
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


def to_div(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={'displayModeBar': False, 'responsive': True})


def section(title, anchor, content_html, subtitle=''):
    sub = f'<p class="subtitle">{subtitle}</p>' if subtitle else ''
    return f'<section id="{anchor}"><h2>{title}</h2>{sub}{content_html}</section>'


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(file):
    scores_raw = pd.read_excel(file, sheet_name='Scores')
    questions  = pd.read_excel(file, sheet_name='Questions')
    std_sheet  = pd.read_excel(file, sheet_name='Content Standards')

    NQ = len(questions)

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

    rename = {scores_raw.columns[k]: v for k, v in enumerate(
        ['Class', 'Grade', 'Period', 'Student', 'ELL', 'IEP'])}
    for i, c in enumerate(answer_cols):
        rename[scores_raw.columns[10 + i]] = c
    for i, c in enumerate(correct_cols):
        rename[scores_raw.columns[10 + NQ + i]] = c

    df = scores_raw.rename(columns=rename).dropna(subset=['Class']).copy()
    df[correct_cols] = df[correct_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    df['MC_Score'] = df[correct_cols].sum(axis=1)
    df['MC_Pct']   = df['MC_Score'] / NQ
    df['MC_Blank'] = df[answer_cols].isna().sum(axis=1)
    df['Is_ELL']   = df['ELL'].notna()
    df['Is_IEP']   = df['IEP'].notna()

    q = questions.copy()
    q = q.rename(columns={
        'CMA #':         'Q_Num',    # CMA sheets
        '#':             'Q_Num',    # Final exam sheets
        'Source #':      'Source_Num',
        'Full Answer':   'Full_Answer',
        'Stimulus Type': 'Stimulus_Type',
        'Source Type':   'Source_Type',
        'Task Model':    'Task_Model',
        'Content':       'Content_Std',
        'Need OI?':      'Need_OI',
        'Point Biserial':'Point_Biserial',
        'Option 1': 'Opt1', 'Option 2': 'Opt2',
        'Option 3': 'Opt3', 'Option 4': 'Opt4',
    })
    q = q.sort_values('Q_Num').reset_index(drop=True)

    return df, q, std_lookup, std_name, answer_cols, correct_cols, NQ


# ── Figure builders ───────────────────────────────────────────────────────────
def fig_histogram(df, nq):
    mean_s = df['MC_Score'].mean()
    counts = [(df['MC_Score'] == s).sum() for s in range(nq + 1)]
    b1, b2, b3 = round(nq * 0.34), round(nq * 0.56), round(nq * 0.78)
    bands = [
        ('Needs Support', C['red'],    -0.5,      b1 + 0.5),
        ('Approaching',   C['orange'],  b1 + 0.5, b2 + 0.5),
        ('Meeting Std',   C['teal'],    b2 + 0.5, b3 + 0.5),
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
                      annotation_font=dict(size=9, color=color))
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


def fig_class_dist(df, nq):
    classes = sorted(df['Class'].unique())
    fig = go.Figure()
    for cls in classes:
        cls_df = df[df['Class'] == cls]
        x_pts  = list(range(nq + 1))
        pcts   = np.array([(cls_df['MC_Score'] == s).sum() / len(cls_df) for s in x_pts])
        color  = CLASS_COLORS.get(cls, C['blue'])
        sigma  = max(0.7, nq / 14)
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


def difficulty_html(df, q, std_name, nq):
    """Heatmap with Hardest→Easiest / Question Order toggle using .q-btn buttons."""
    overall_p     = [df[f'Q{i}'].mean() for i in range(1, nq + 1)]
    order         = list(np.argsort(overall_p))
    q_nums_sorted = [order[j] + 1 for j in range(nq)]
    col_labels    = [f'{q_nums_sorted[j]}' for j in range(nq)]
    full_q        = [str(q['Question'].tolist()[order[j]]).strip() for j in range(nq)]
    answers       = [str(q['Full_Answer'].tolist()[order[j]]).strip() for j in range(nq)]
    std_labels    = [std_name(q['Content_Std'].tolist()[order[j]], 35) for j in range(nq)]

    row_names = ['Overall'] + sorted(df['Class'].unique().tolist())
    ns        = []   # sample size per row, reused for both sort orders
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
            f'n={n}'
            for j in range(nq)
        ])

    all_vals = [v for row in z_data for v in row]
    zmin_val, zmax_val = min(all_vals), max(all_vals)

    # Natural order — precompute answers in Q1…Qn order
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
            f'n={ns[ri]}'
            for j in range(nq)
        ])

    show_text = nq <= 20
    fig = go.Figure(go.Heatmap(
        z=z_data, x=col_labels, y=row_names,
        colorscale=[[0, C['red']], [0.5, C['orange']], [1.0, C['teal']]],
        zmin=zmin_val, zmax=zmax_val, xgap=3, ygap=3,
        text=text_data, texttemplate='%{text}' if show_text else '',
        textfont=dict(size=11, color='white'),
        hoverinfo='text', hovertext=hover_data, showscale=True,
        colorbar=dict(title='% Correct', tickformat='.0%',
                      tickvals=[zmin_val, (zmin_val + zmax_val) / 2, zmax_val],
                      ticktext=[f'{zmin_val:.0%}', f'{(zmin_val+zmax_val)/2:.0%}', f'{zmax_val:.0%}']),
    ))
    fig.update_layout(**LAYOUT,
        title='', height=max(240, 80 + 70 * len(row_names)),
        xaxis=dict(side='top', title='', tickfont=dict(size=11), showgrid=False, ticklen=8),
        yaxis=dict(autorange='reversed', tickfont=dict(size=12), showgrid=False, ticklen=8),
    )
    fig.add_shape(type='line', xref='paper', yref='y', x0=0, x1=1, y0=0.5, y1=0.5,
                  line=dict(color='white', width=10), layer='above')
    fig.update_layout(margin=dict(l=95, r=20, t=70, b=20))

    chart_div = fig.to_html(full_html=False, include_plotlyjs=False,
                            config={'displayModeBar': False, 'responsive': True})
    m = _re.search(r'<div id="([^"]+)"', chart_div)
    fig_id = m.group(1) if m else 'plotly-diff'
    bid    = fig_id[-8:]

    hard_js = json.dumps({'x': [col_labels], 'z': [z_data],  'text': [text_data],  'hovertext': [hover_data]})
    nat_js  = json.dumps({'x': [col_nat],    'z': [z_nat],   'text': [txt_nat],    'hovertext': [hov_nat]})

    btn_html = (
        f'<div class="q-btns" style="margin-top:10px">'
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
            marker_color=[C['teal'] if j == ca else C['lgray'] for j in range(1, 5)],
            marker_line_color='white', marker_line_width=1.2,
            text=[f'{p:.0%}' for p in pcts], textposition='outside',
            hovertext=[
                f'<b>{opts[j-1]}</b><br>'
                + ('✓ Correct' if j == ca else '✗ Incorrect')
                for j in range(1, 5)
            ],
            hoverinfo='text', visible=(q_num == 1), showlegend=False, cliponaxis=False,
        ))

    fig.update_layout(**LAYOUT, height=340,
        xaxis=dict(tickvals=[1, 2, 3, 4], title='', showgrid=False, zeroline=False),
        yaxis=dict(tickformat='.0%', title='% of Students', range=[0, 1.18],
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    fig.update_layout(margin=dict(l=0, r=20, t=5, b=20))

    chart_div = fig.to_html(full_html=False, include_plotlyjs=False,
                            config={'displayModeBar': False, 'responsive': True})
    m = _re.search(r'<div id="([^"]+)"', chart_div)
    fig_id = m.group(1) if m else 'plotly-dis'
    txt_id = f'dis-q-text-{fig_id[-8:]}'

    texts_js = '[' + ','.join(json.dumps(q_text_map.get(i, '')) for i in range(1, nq + 1)) + ']'
    btn_items = ''.join(
        f'<button class="q-btn{"  q-btn-active" if q_num == 1 else ""}" '
        f'data-qi="{q_num - 1}" data-fig="{fig_id}">{q_num}</button>'
        for q_num in range(1, nq + 1)
    )

    return f'''
<div id="{txt_id}" style="font-size:13px;color:{C['navy']};
     padding:10px 4px 4px;min-height:20px;line-height:1.5">{q_text_map[1]}</div>
{chart_div}
<div class="q-btns">{btn_items}</div>
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
    fig.add_vline(x=0.70, line_dash='dash', line_color=C['navy'], opacity=0.4,
                  annotation_text='70% target', annotation_position='top right',
                  annotation_font=dict(size=10, color=C['navy']))
    fig.update_layout(**LAYOUT,
        title='Content Standards — Which Need More Practice?',
        height=max(380, 40 * nq + 100),
        legend=dict(orientation='h', y=-0.14, x=0.5, xanchor='center', font=dict(size=11)),
        xaxis=dict(tickformat='.0%', title='% Who Answered Correctly',
                   range=[0, 1.22], showgrid=True, gridcolor='#EEEEEE', zeroline=False),
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
            fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.35)',
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


def fig_scatter(df, q, std_name, nq):
    p_vals  = [df[f'Q{i}'].mean() for i in range(1, nq + 1)]
    pb_vals = [stats.pointbiserialr(df[f'Q{i}'], df['MC_Score'])[0] for i in range(1, nq + 1)]
    topics  = [std_name(q.loc[q['Q_Num'] == i, 'Content_Std'].values[0], 60) for i in range(1, nq + 1)]
    q_texts = [str(q.loc[q['Q_Num'] == i, 'Question'].values[0]).strip()[:80] for i in range(1, nq + 1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p_vals, y=pb_vals, mode='markers+text',
        text=[f'{i}' for i in range(1, nq + 1)], textposition='top right',
        textfont=dict(size=11, color=C['navy']),
        marker=dict(color=[pct_color(p) for p in p_vals], size=14,
                    line=dict(color='white', width=1.5)),
        hovertext=[
            f'<b>{i+1}: {topics[i]}</b><br>{q_texts[i]}…<br><br>'
            f'Difficulty: {p_vals[i]:.0%}<br>Point-biserial: {pb_vals[i]:.2f}'
            for i in range(nq)],
        hoverinfo='text', showlegend=False,
    ))
    for label, color in [('Hard (<50%)', C['red']), ('Medium (50–70%)', C['orange']), ('Easy (>70%)', C['teal'])]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color=color, size=10), name=label, showlegend=True))
    fig.add_hline(y=0.20, line_dash='dash', line_color=C['red'], opacity=0.5,
                  annotation_text='PB = 0.20 (min)', annotation_font=dict(size=10, color=C['red']))
    fig.add_hline(y=0.30, line_dash='dash', line_color=C['orange'], opacity=0.5,
                  annotation_text='PB = 0.30', annotation_font=dict(size=10, color=C['orange']))
    fig.add_vline(x=0.50, line_dash='dot', line_color=C['gray'], opacity=0.35)
    fig.update_layout(**LAYOUT,
        title='Difficulty vs. Discrimination', height=460,
        legend=dict(orientation='h', y=-0.16, x=0.5, xanchor='center', font=dict(size=11)),
        xaxis=dict(tickformat='.0%', title='Difficulty (% correct)',
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False, range=[0, 1.05]),
        yaxis=dict(title='Discrimination (point-biserial)',
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    return fig


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
        yaxis=dict(tickformat='.0%', title='Average % Correct', range=[0, 1.3],
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
        yaxis=dict(tickformat='.0%', title='Average % Correct', range=[0, 1.25],
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    return fig


def fig_skill(df, q, nq):
    sks = q.dropna(subset=['Skill'])
    if sks.empty:
        return None

    def skill_label(s):
        m = _re.match(r'^([A-Z])\.', s.strip())
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
        yaxis=dict(tickformat='.0%', title='Average % Correct', range=[0, 1.25],
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
        yaxis=dict(tickformat='.0%', title='Average % Correct', range=[0, 1.25],
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    return fig


# ── HTML helpers ──────────────────────────────────────────────────────────────
def kpi_html(df, nq):
    mean_s  = df['MC_Score'].mean()
    thresh  = round(nq * 0.67)
    pct_thr = (df['MC_Score'] >= thresh).mean()
    n       = len(df)
    n_blank = (df['MC_Blank'] > 0).sum()

    def card(value, label, color):
        return (f'<div style="background:{color};border-radius:10px;padding:22px 18px;'
                f'text-align:center;color:white;flex:1;min-width:160px">'
                f'<div style="font-size:2rem;font-weight:700;line-height:1.1">{value}</div>'
                f'<div style="font-size:0.85rem;opacity:0.92;margin-top:6px">{label}</div></div>')

    return (f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:18px;margin-bottom:8px">'
            + card(f'{mean_s:.1f} / {nq}', 'Class Average Score', score_color(round(mean_s), nq))
            + card(f'{pct_thr:.0%}', f'Scored {thresh}+ ({thresh/nq:.0%}+)',
                   C['teal'] if pct_thr >= 0.55 else C['orange'])
            + card(str(n), 'Total Students', C['blue'])
            + card(str(n_blank), 'Left a Question Blank',
                   C['orange'] if n_blank > n * 0.10 else C['teal'])
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
                f'<li><b>{name}</b> (n={n_g}): avg {avg_g:.1f} vs {avg_o:.1f} for other students — {note}</li>')
    if not lines:
        return ''
    return (f'<div style="margin-top:18px;background:#f5f7ff;border-radius:8px;padding:14px 18px">'
            f'<div style="font-weight:700;color:{C["navy"]};margin-bottom:6px">Subgroup Notes</div>'
            f'<ul style="margin:0;padding-left:18px;font-size:0.9rem">{"".join(lines)}</ul></div>')


def takeaways_html(df, q, std_name, nq):
    container_id = 'tk-' + uuid.uuid4().hex[:8]
    views = ['Overall'] + sorted(df['Class'].unique().tolist())

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
            f'<tr>'
            f'<td style="padding:5px 8px">{lbl or ql}</td>'
            f'<td style="font-weight:700;color:{color};padding:5px 8px;'
            f'text-align:right;white-space:nowrap">{p:.0%}</td>'
            f'</tr>'
            for ql, p, lbl in items
        ) or f'<tr><td colspan="2" style="color:#888;padding:8px;font-style:italic">None</td></tr>'
        return (f'<div style="background:{bg};border:2px solid {color};border-radius:10px;'
                f'padding:16px 18px;flex:1;min-width:220px">'
                f'<div style="font-size:1.1rem;font-weight:700;color:{color}">{title}</div>'
                f'<div style="font-size:0.82rem;color:#888;margin-bottom:10px;font-style:italic">{subtitle}</div>'
                f'<table style="width:100%;border-collapse:collapse;font-size:0.9rem">{rows}</table></div>')

    def panels_html(fdf):
        items   = std_items_for(fdf)
        reteach = sorted([(ql,p,lbl) for ql,p,lbl in items if p < 0.50],     key=lambda x: x[1])
        review  = sorted([(ql,p,lbl) for ql,p,lbl in items if 0.50<=p<0.70], key=lambda x: x[1])
        strong  = sorted([(ql,p,lbl) for ql,p,lbl in items if p >= 0.70],    key=lambda x: -x[1])
        return (f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:12px">'
                + panel('RETEACH', 'Below 50% correct', reteach, C['red'],    '#fdf0f1')
                + panel('REVIEW',  '50–70% correct',    review,  C['orange'], '#fef8f0')
                + panel('STRONG',  'Above 70% correct', strong,  C['teal'],   '#f0fbf8')
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

    return f'''<div class="q-btns" style="margin-bottom:4px">{btns}</div>
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
def build_html(df, q, std_name, nq, title, home_link='../../index.html'):
    today = date.today().strftime('%B %d, %Y')

    figs = {
        'hist':          fig_histogram(df, nq),
        'cp_bar':        fig_class_bar(df, nq),
        'cp_dist':       fig_class_dist(df, nq),
        'grp':           fig_groups(df, nq),
        'box':           fig_boxplot(df, nq),
        'scatter':       fig_scatter(df, q, std_name, nq),
        'oi':            fig_oi(df, q, nq),
        'task_model':    fig_task_model(df, q, nq),
        'skill':         fig_skill(df, q, nq),
        'stimulus_type': fig_stimulus_type(df, q, nq),
    }

    nav_links = f'<a href="{home_link}">← Home</a>' + ''.join(
        f'<a href="#{a}">{t}</a>'
        for a, t in [
            ('overview',     'Overview'),
            ('class-period', 'Class & Period'),
            ('difficulty',   'Difficulty'),
            ('distractor',   'Distractors'),
            ('groups',       'Groups'),
            ('takeaways',    'Takeaways'),
            ('appendix',     'Appendix'),
        ]
    )

    def _opt(fig):
        return to_div(fig) if fig is not None else ''

    HEADER_H = 72
    body = '\n'.join([
        section('1.  At a Glance', 'overview',
            kpi_html(df, nq) + to_div(figs['hist'])),

        section('2.  Class & Period Performance', 'class-period',
            '<div class="chart-grid">'
            + to_div(figs['cp_bar']) + to_div(figs['cp_dist'])
            + '</div>'),

        section('3.  Which Questions Were Hardest?', 'difficulty',
            f'<div class="scroll-x"><div style="min-width:{max(520, nq * 34)}px">'
            + difficulty_html(df, q, std_name, nq) + '</div></div>'),

        section('4.  What Did Students Select?', 'distractor',
            distractor_html(df, q, nq)),

        section('6.  Student Group Comparison', 'groups',
            to_div(figs['grp']) + subgroup_notes_html(df)),

        section('7.  Key Takeaways', 'takeaways',
            takeaways_html(df, q, std_name, nq)),

        section('Appendix', 'appendix',
            to_div(figs['box']) + to_div(figs['scatter'])
            + '<div class="chart-grid-2">'
            + _opt(figs['oi'])
            + _opt(figs['stimulus_type'])
            + _opt(figs['task_model'])
            + _opt(figs['skill'])
            + '</div>'),
    ])

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
      background: #f0f2f5; color: {C["navy"]}; margin: 0; padding: 0; line-height: 1.5;
    }}
    header {{
      background: {C["navy"]}; color: white;
      padding: 14px 36px; position: sticky; top: 0; z-index: 100;
      display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.25); min-height: {HEADER_H}px;
    }}
    header h1 {{ margin: 0; font-size: 1.15rem; font-weight: 700; }}
    nav {{ display: flex; gap: 5px; flex-wrap: wrap; }}
    nav a {{
      color: white; opacity: 0.82; text-decoration: none;
      font-size: 0.82rem; padding: 4px 11px; border-radius: 20px;
      background: rgba(255,255,255,0.13); transition: opacity .15s, background .15s;
      white-space: nowrap;
    }}
    nav a:hover {{ opacity: 1; background: rgba(255,255,255,0.25); }}
    main {{
      max-width: 1080px; margin: 0 auto; padding: 32px 24px 72px;
      display: flex; flex-direction: column; gap: 28px;
    }}
    section {{
      background: white; border-radius: 14px; padding: 28px 30px 22px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.07), 0 0 0 1px rgba(0,0,0,0.04);
      scroll-margin-top: {HEADER_H + 12}px;
    }}
    section h2 {{
      color: {C["navy"]}; font-size: 1.15rem; font-weight: 700;
      margin: 0 0 3px; padding-bottom: 10px; border-bottom: 3px solid {C["teal"]};
    }}
    section p.subtitle {{ color: #777; font-size: 0.85rem; margin: 6px 0 16px; font-style: italic; }}
    .plotly-graph-div {{ width: 100% !important; }}
    .chart-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;
    }}
    .chart-grid-2 {{
      display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;
    }}
    @media (max-width: 640px) {{
      .chart-grid-2 {{ grid-template-columns: 1fr; }}
    }}
    .scroll-x {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
    .scroll-x > * {{ min-width: 520px; }}
    .q-btns {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; padding: 0 2px; }}
    .q-btn {{
      font-size: 0.78rem; padding: 3px 9px; border-radius: 6px;
      border: 1px solid #CCC; background: white; cursor: pointer;
      color: {C["navy"]}; transition: background .1s, color .1s;
    }}
    .q-btn:hover {{ background: #f0f2f5; }}
    .q-btn-active {{
      background: {C["navy"]} !important; color: white !important;
      border-color: {C["navy"]} !important;
    }}
    footer {{ text-align: center; padding: 28px; font-size: 0.78rem; color: #aaa; }}
    @media (max-width: 640px) {{
      header {{ padding: 12px 16px; flex-direction: column; align-items: flex-start; }}
      main {{ padding: 16px 12px 48px; gap: 18px; }}
      section {{ padding: 20px 16px 16px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <nav>{nav_links}</nav>
  </header>
  <main>{body}</main>
  <footer>Generated {today}</footer>
  <script>
    setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 50);
    setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 300);
  </script>
</body>
</html>'''


# ── Entry point ───────────────────────────────────────────────────────────────
def run(config):
    """
    Generate a report from a config dict. Example:

        lib.run({
            'file':         './US/May Final/US May Final Data.xlsx',
            'title':        'US History Final — May 2026',
            'output':       './US/May Final/us_may_final_report.html',
            'class_colors': {'Bermejo': '#3A86FF', 'Dushin': '#F4A261', 'Kovelsky': '#44BBA4'},
        })
    """
    global CLASS_COLORS
    if 'class_colors' in config:
        CLASS_COLORS = config['class_colors']

    file      = config['file']
    title     = config['title']
    output    = config['output']
    home_link = config.get('home_link', '../../index.html')

    print('Loading data...')
    df, q, std_lookup, std_name, answer_cols, correct_cols, NQ = load_data(file)
    print(f'  {len(df)} students, {df["Class"].nunique()} classes, {NQ} questions')

    print('Building report...')
    html = build_html(df, q, std_name, NQ, title, home_link)

    with open(output, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = len(html.encode('utf-8')) / 1024
    print(f'Done → {output}  ({size_kb:.0f} KB)')
