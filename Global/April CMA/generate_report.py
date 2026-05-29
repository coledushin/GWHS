"""
Generate a self-contained interactive HTML report for any MC exam.

Run from the exam folder:
    python3 generate_report.py

Output:  report.html
Works for any number of MC questions — auto-detected from the Questions sheet.
"""

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from datetime import date

FILE   = './Global/April CMA/Global April CMA Data.xlsx'
TITLE  = 'Global History CMA — April 2026'
OUTPUT = './Global/April CMA/global_april_cma_report.html'

# ── Palette ───────────────────────────────────────────────────────────────────
C = dict(red='#E84855', orange='#F4A261', teal='#44BBA4', green='#06D6A0',
         blue='#3A86FF', navy='#2D3142', gray='#888888', lgray='#D8D8D8')

CLASS_COLORS = {'Bermejo': C['blue'], 'Dushin': C['orange'], 'Kovelsky': C['teal']}

def pct_color(p):
    if p >= 0.70: return C['teal']
    if p >= 0.50: return C['orange']
    return C['red']

def score_color(s, nq):
    """Traffic-light color based on % correct, works for any N."""
    p = s / nq
    if p >= 0.78: return C['green']
    if p >= 0.56: return C['teal']
    if p >= 0.34: return C['orange']
    return C['red']

def hex_to_rgba(hex_color, alpha=0.09):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

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

def add_color_legend(fig):
    for label, color in LEGEND_ITEMS:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color=color, size=10, symbol='square'),
            name=label, showlegend=True))


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    scores_raw = pd.read_excel(FILE, sheet_name='Scores')
    questions  = pd.read_excel(FILE, sheet_name='Questions')
    std_sheet  = pd.read_excel(FILE, sheet_name='Content Standards')

    # Auto-detect question count from Questions sheet
    NQ = len(questions)

    # Rename just the first two columns so extra columns don't break things
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
        ['Class','Grade','Period','Student','ELL','IEP'])}
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
    # Detect if Option columns exist
    # Rename by original column name so any sheet layout works
    q = q.rename(columns={
        'CMA #':        'Q_Num',
        'Source #':     'Source_Num',
        'Full Answer':  'Full_Answer',
        'Stimulus Type':'Stimulus_Type',
        'Source Type':  'Source_Type',
        'Task Model':   'Task_Model',
        'Content':      'Content_Std',
        'Need OI?':     'Need_OI',
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

    # Proportional score bands
    b1 = round(nq * 0.34)
    b2 = round(nq * 0.56)
    b3 = round(nq * 0.78)
    bands = [
        ('Needs Support',  C['red'],    -0.5, b1 + 0.5),
        ('Approaching',    C['orange'],  b1 + 0.5, b2 + 0.5),
        ('Meeting Std',    C['teal'],    b2 + 0.5, b3 + 0.5),
        ('Exceeding',      C['green'],   b3 + 0.5, nq + 0.5),
    ]

    fig = go.Figure()
    def _bc(s):
        if s > b3: return C['green']
        if s > b2: return C['teal']
        if s > b1: return C['orange']
        return C['red']

    fig.add_trace(go.Bar(
        x=list(range(nq + 1)), y=counts,
        marker_color=[_bc(s) for s in range(nq + 1)],
        marker_line_color='white', marker_line_width=1.5,
        hovertemplate='Score %{x}/' + str(nq) + ': %{y} students<extra></extra>',
        showlegend=False,
    ))
    fig.add_vline(x=mean_s, line_dash='dot', line_color='rgba(100,100,100,0.4)',
                  line_width=1.5)
    for label, color, lo, hi in bands:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=color, opacity=0.06,
                      line_width=0, annotation_text=label,
                      annotation_position='top left',
                      annotation_font=dict(size=9, color=color))
    fig.update_layout(**LAYOUT,
        title='How Did Students Score?', height=350,
        xaxis=dict(title=f'Questions Correct (out of {nq})',
                   tickvals=list(range(nq + 1)), showgrid=False, zeroline=False),
        yaxis=dict(title='Number of Students', showgrid=True,
                   gridcolor='#EEEEEE', zeroline=False),
    )
    return fig


def fig_class_bar(df, nq):
    cp = df.groupby(['Class','Period'])['MC_Score'].agg(['mean','count']).reset_index()
    classes = sorted(df['Class'].unique())
    bar_width = 0.22

    fig = go.Figure()
    group_gap = 0.15
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
                hovertemplate=f'{cls} Period {int(row["Period"])}: {row["mean"]:.1f}/{nq} avg (n={int(row["count"])})<extra></extra>',
                showlegend=False, width=bar_width * 0.85,
            ))
        class_centers[cls] = current_x + (n_bars - 1) * bar_width / 2
        current_x += n_bars * bar_width + group_gap

    fig.update_xaxes(tickvals=list(class_centers.values()),
                     ticktext=list(class_centers.keys()),
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
            hovertemplate=f'{cls}: %{{y:.1%}} at score %{{x}}<extra></extra>',
        ))
    fig.update_xaxes(title_text=f'Score (out of {nq})', tickvals=list(range(nq + 1)),
                     showgrid=False, zeroline=False)
    fig.update_yaxes(title_text='Share of Class', tickformat='.0%',
                     showgrid=True, gridcolor='#EEEEEE', zeroline=False)
    fig.update_layout(**LAYOUT, title='Score Distribution by Class', height=320,
                      legend=dict(orientation='h', y=-0.22, x=0.5, xanchor='center'))
    return fig


def fig_difficulty(df, q, std_name, nq):
    overall_p     = [df[f'Q{i}'].mean() for i in range(1, nq + 1)]
    order         = list(np.argsort(overall_p))          # hardest first
    q_nums_sorted = [order[j] + 1 for j in range(nq)]
    col_labels    = [f'Q{q_nums_sorted[j]}' for j in range(nq)]
    full_q        = [str(q['Question'].tolist()[order[j]]).strip() for j in range(nq)]
    answers       = [str(q['Full_Answer'].tolist()[order[j]]).strip() for j in range(nq)]
    std_labels    = [std_name(q['Content_Std'].tolist()[order[j]], 35) for j in range(nq)]

    row_names = ['Overall'] + sorted(df['Class'].unique().tolist())
    z_data, text_data, hover_data = [], [], []
    for row_name in row_names:
        fdf   = df if row_name == 'Overall' else df[df['Class'] == row_name]
        n     = len(fdf)
        row_p = [fdf[f'Q{q_nums_sorted[j]}'].mean() for j in range(nq)]
        z_data.append(row_p)
        text_data.append([f'{p:.0%}' for p in row_p])
        hover_data.append([
            f'<b>Q{q_nums_sorted[j]}: {std_labels[j]}</b><br>'
            f'{full_q[j]}<br><br>'
            f'✓ <b>Correct:</b> {answers[j]}<br>'
            f'{row_name} (n={n}): {row_p[j]:.0%} correct'
            for j in range(nq)
        ])

    all_vals = [v for row in z_data for v in row]
    zmin_val, zmax_val = min(all_vals), max(all_vals)

    show_text = nq <= 20
    fig = go.Figure(go.Heatmap(
        z=z_data, x=col_labels, y=row_names,
        colorscale=[[0, C['red']], [0.5, C['orange']], [1.0, C['teal']]],
        zmin=zmin_val, zmax=zmax_val,
        xgap=3, ygap=3,
        text=text_data,
        texttemplate='%{text}' if show_text else '',
        textfont=dict(size=11, color='white'),
        hoverinfo='text', hovertext=hover_data,
        showscale=True,
        colorbar=dict(title='% Correct', tickformat='.0%',
                      tickvals=[zmin_val, (zmin_val+zmax_val)/2, zmax_val],
                      ticktext=[f'{zmin_val:.0%}', f'{(zmin_val+zmax_val)/2:.0%}', f'{zmax_val:.0%}']),
    ))
    # Also build question-number order version for toggle
    nat_order   = list(range(nq))                          # Q1, Q2, … Qn
    col_nat     = [f'Q{j+1}' for j in nat_order]
    z_nat, txt_nat, hov_nat = [], [], []
    for ri, row_name in enumerate(row_names):
        z_nat.append([z_data[ri][order.index(j)] for j in nat_order])
        txt_nat.append([f'{z_nat[-1][j]:.0%}' for j in range(nq)])
        hov_nat.append([
            f'<b>Q{j+1}: {std_name(q["Content_Std"].tolist()[j], 35)}</b><br>'
            f'{str(q["Question"].tolist()[j]).strip()}<br><br>'
            f'{row_name}: {z_nat[-1][j]:.0%} correct'
            for j in nat_order
        ])

    sort_btns = [
        dict(label='Hardest → Easiest', method='update',
             args=[{'x': [col_labels], 'z': [z_data], 'text': [text_data],
                    'hovertext': [hover_data]}]),
        dict(label='Question Order', method='update',
             args=[{'x': [col_nat],   'z': [z_nat],  'text': [txt_nat],
                    'hovertext': [hov_nat]}]),
    ]

    fig.update_layout(**LAYOUT,
        title='',
        height=max(240, 80 + 70 * len(row_names)),
        xaxis=dict(side='top', title='', tickfont=dict(size=11), showgrid=False,
                   ticklen=8),
        yaxis=dict(autorange='reversed', tickfont=dict(size=12), showgrid=False,
                   ticklen=8),
        updatemenus=[dict(
            type='buttons', buttons=sort_btns, direction='right',
            showactive=True, x=0, xanchor='left', y=-0.1, yanchor='top',
            bgcolor='white', bordercolor='#CCCCCC', font=dict(size=10),
        )],
    )
    # White line between Overall row (y=0) and first class row (y=1)
    fig.add_shape(type='line', xref='paper', yref='y',
                  x0=0, x1=1, y0=0.5, y1=0.5,
                  line=dict(color='white', width=10), layer='above')
    fig.update_layout(margin=dict(l=95, r=20, t=70, b=60))
    return fig


def distractor_html(df, q, nq):
    """Returns self-contained HTML: question text div + chart + buttons + JS listener.
    Layout (top to bottom): question text → chart → Q-buttons."""
    import re as _re
    has_opts = 'Opt1' in q.columns
    q_text_map = {int(r['Q_Num']): str(r['Question']).strip() for _, r in q.iterrows()}

    fig = go.Figure()
    for i in range(nq):
        q_num = i + 1
        qrow  = q.loc[q['Q_Num'] == q_num].iloc[0]
        ca    = int(qrow['Answer'])
        opts  = ([str(qrow[f'Opt{j}']).strip() for j in range(1, 5)]
                 if has_opts else [str(j) for j in range(1, 5)])
        answered = df[f'Q{q_num}_ans'].dropna().astype(int)
        total    = len(answered)
        counts   = answered.value_counts().reindex([1, 2, 3, 4], fill_value=0)
        pcts     = [counts[j] / total if total > 0 else 0 for j in range(1, 5)]
        fig.add_trace(go.Bar(
            x=[1, 2, 3, 4], y=pcts,
            marker_color=[C['teal'] if j == ca else C['lgray'] for j in range(1, 5)],
            marker_line_color='white', marker_line_width=1.2,
            text=[f'{p:.0%}' for p in pcts], textposition='outside',
            hovertext=[
                f'<b>Choice {j}: {opts[j-1]}</b><br>{pcts[j-1]:.0%} selected<br>'
                + ('✓ <b>Correct answer</b>' if j == ca else '✗ Incorrect')
                for j in range(1, 5)
            ],
            hoverinfo='text', visible=(q_num == 1), showlegend=False, cliponaxis=False,
        ))

    fig.update_layout(**LAYOUT,
        height=340,
        xaxis=dict(tickvals=[1, 2, 3, 4], title='',
                   showgrid=False, zeroline=False),
        yaxis=dict(tickformat='.0%', title='% of Students', range=[0, 1.18],
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    fig.update_layout(margin=dict(l=0, r=20, t=5, b=20))

    chart_div = fig.to_html(full_html=False, include_plotlyjs=False,
                            config={'displayModeBar': False, 'responsive': True})

    m = _re.search(r'<div id="([^"]+)"', chart_div)
    fig_id = m.group(1) if m else 'plotly-dis'
    txt_id = f'dis-q-text-{fig_id[-8:]}'

    texts_js = '[' + ','.join(
        json.dumps(q_text_map.get(i, '')) for i in range(1, nq + 1)
    ) + ']'

    btn_items = ''.join(
        f'<button class="q-btn{"  q-btn-active" if q_num == 1 else ""}" '
        f'data-qi="{q_num - 1}" data-fig="{fig_id}">Q{q_num}</button>'
        for q_num in range(1, nq + 1)
    )
    btn_html = f'<div class="q-btns">{btn_items}</div>'

    return f'''
<div id="{txt_id}" style="font-size:13px;color:{C['navy']};
     padding:10px 4px 4px;min-height:20px;line-height:1.5">{q_text_map[1]}</div>
{chart_div}
{btn_html}
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
        'IEP only':       (~df['Is_ELL']) & df['Is_IEP'],
        'ELL only':       df['Is_ELL'] & (~df['Is_IEP']),
        'Both ELL & IEP': df['Is_ELL'] & df['Is_IEP'],
    }
    groups = {n: m for n, m in masks.items() if m.sum() > 0}
    palette = [C['blue'], C['teal'], C['orange'], C['red']]

    fig = go.Figure()
    for (name, mask), color in zip(groups.items(), palette):
        mean_val = df[mask]['MC_Score'].mean()
        n = int(mask.sum())
        fig.add_trace(go.Bar(
            x=[name], y=[mean_val],
            name=f'{name}  (n={n})',
            marker_color=color, marker_line_color='white', marker_line_width=1.5,
            text=[f'{mean_val:.1f}'], textposition='outside',
            width=0.45,
            hoverinfo='skip',   # value is already on the bar
        ))
    fig.add_hline(y=df['MC_Score'].mean(), line_dash='dot', line_color=C['navy'],
                  opacity=0.55,
                  annotation_text=f'Overall: {df["MC_Score"].mean():.1f}',
                  annotation_position='top right',
                  annotation_font=dict(size=10, color=C['navy']))
    fig.update_layout(**LAYOUT,
        title='Average Score by Student Group',
        height=380, showlegend=True,
        legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
        barmode='group',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title=f'Average Score (out of {nq})', range=[0, nq * 1.28],
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    return fig


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
    return f'''
    <div style="margin-top:18px;background:#f5f7ff;border-radius:8px;padding:14px 18px">
      <div style="font-weight:700;color:{C["navy"]};margin-bottom:6px">Subgroup Notes</div>
      <ul style="margin:0;padding-left:18px;font-size:0.9rem">{''.join(lines)}</ul>
    </div>'''


def fig_boxplot(df, nq):
    class_list = sorted(df['Class'].unique())
    palette    = [CLASS_COLORS.get(c, C['blue']) for c in class_list]

    fig = go.Figure()
    for cls, color in zip(class_list, palette):
        vals = df[df['Class'] == cls]['MC_Score'].tolist()
        fig.add_trace(go.Box(
            y=vals, name=cls,
            marker_color=color,
            line_color=color,
            fillcolor=color.replace('#', '') and  # use rgba for fill
                f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.35)',
            boxpoints='all',
            jitter=0.35,
            pointpos=0,
            marker=dict(size=5, opacity=0.4, color=color),
            hovertemplate=f'{cls}: %{{y}}/{nq}<extra></extra>',
        ))

    fig.update_layout(**LAYOUT,
        title='Score Distribution by Class',
        height=380,
        showlegend=False,
        yaxis=dict(title=f'MC Score (out of {nq})', showgrid=True,
                   gridcolor='#EEEEEE', zeroline=False, range=[-0.5, nq + 0.5]),
        xaxis=dict(showgrid=False, zeroline=False),
    )
    return fig


def fig_scatter(df, q, std_name, nq):
    p_vals  = [df[f'Q{i}'].mean() for i in range(1, nq + 1)]
    pb_vals = [stats.pointbiserialr(df[f'Q{i}'], df['MC_Score'])[0] for i in range(1, nq + 1)]
    topics  = [std_name(q.loc[q['Q_Num']==i,'Content_Std'].values[0], 60) for i in range(1, nq + 1)]
    q_texts = [str(q.loc[q['Q_Num']==i,'Question'].values[0]).strip()[:80] for i in range(1, nq + 1)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p_vals, y=pb_vals, mode='markers+text',
        text=[f'Q{i}' for i in range(1, nq + 1)], textposition='top right',
        textfont=dict(size=11, color=C['navy']),
        marker=dict(color=[pct_color(p) for p in p_vals], size=14,
                    line=dict(color='white', width=1.5)),
        hovertext=[
            f'<b>Q{i+1}: {topics[i]}</b><br>{q_texts[i]}…<br><br>'
            f'Difficulty: {p_vals[i]:.0%} correct<br>Discrimination (PB): {pb_vals[i]:.2f}'
            for i in range(nq)],
        hoverinfo='text', showlegend=False,
    ))
    for label, color in [('Hard (<50%)', C['red']),
                          ('Medium (50–70%)', C['orange']),
                          ('Easy (>70%)', C['teal'])]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color=color, size=10), name=label, showlegend=True))
    fig.add_hline(y=0.20, line_dash='dash', line_color=C['red'], opacity=0.5,
                  annotation_text='PB = 0.20 (min)',
                  annotation_font=dict(size=10, color=C['red']))
    fig.add_hline(y=0.30, line_dash='dash', line_color=C['orange'], opacity=0.5,
                  annotation_text='PB = 0.30',
                  annotation_font=dict(size=10, color=C['orange']))
    fig.add_vline(x=0.50, line_dash='dot', line_color=C['gray'], opacity=0.35)
    fig.update_layout(**LAYOUT,
        title='Difficulty vs. Discrimination',
        height=460,
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
        s = str(val).strip().lower()
        return s not in ('', 'no', 'n', 'false', '0', 'nan', 'none')

    q_copy = q.copy()
    q_copy['OI_Required'] = q_copy['Need_OI'].apply(is_oi)
    oi_yes = q_copy[q_copy['OI_Required']]['Q_Num'].tolist()
    oi_no  = q_copy[~q_copy['OI_Required']]['Q_Num'].tolist()

    categories = [(lbl, qns) for lbl, qns in
                  [('Requires OI', oi_yes), ('No OI Needed', oi_no)] if qns]
    if len(categories) < 2:
        return None

    groups  = ['Overall'] + sorted(df['Class'].unique().tolist())
    palette = [C['navy']] + [CLASS_COLORS.get(c, C['blue']) for c in sorted(df['Class'].unique())]

    fig = go.Figure()
    for grp, color in zip(groups, palette):
        fdf  = df if grp == 'Overall' else df[df['Class'] == grp]
        n    = len(fdf)
        vals = [float(np.mean([fdf[f'Q{qn}'].mean() for qn in qns])) for _, qns in categories]
        q_counts = [len(qns) for _, qns in categories]
        fig.add_trace(go.Bar(
            x=[lbl for lbl, _ in categories], y=vals, name=grp,
            marker_color=color, marker_line_color='white', marker_line_width=1,
            text=[f'{v:.0%}' for v in vals], textposition='outside',
            cliponaxis=False,
            hovertemplate='<b>' + grp + f' (n={n})</b><br>%{{x}}<br>%{{y:.0%}} avg correct (%{{customdata}} Qs)<extra></extra>',
            customdata=q_counts,
        ))

    fig.update_layout(**LAYOUT,
        title='Performance: Outside Information Required vs. Not',
        height=380,
        barmode='group',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(tickformat='.0%', title='Average % Correct', range=[0, 1.3],
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
        legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center'),
    )
    return fig


def fig_task_model(df, q, nq):
    import re as _re
    tms = q.dropna(subset=['Task_Model'])
    if tms.empty:
        return None

    def tm_sort_key(tm):
        m = _re.match(r'^(\d+)', tm.strip())
        return int(m.group(1)) if m else float('inf')

    def tm_num(tm):
        m = _re.match(r'^(\d+)', tm.strip())
        return m.group(1) if m else tm[:3]

    task_models = sorted(tms['Task_Model'].astype(str).str.strip().unique().tolist(),
                         key=tm_sort_key)

    labels, vals, hover_texts = [], [], []
    for tm in task_models:
        tm_qs  = q[q['Task_Model'].astype(str).str.strip() == tm]['Q_Num'].tolist()
        avg    = float(np.mean([df[f'Q{qn}'].mean() for qn in tm_qs]))
        q_list = ', '.join(f'Q{qn}' for qn in sorted(tm_qs))
        labels.append(tm_num(tm))
        vals.append(avg)
        hover_texts.append(f'<b>{tm}</b><br>Questions: {q_list}<br>Avg % correct: {avg:.0%}')

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=[pct_color(v) for v in vals],
        marker_line_color='white', marker_line_width=1.5,
        text=[f'{v:.0%}' for v in vals], textposition='outside',
        cliponaxis=False,
        hovertext=hover_texts, hoverinfo='text',
        showlegend=False,
        width=0.5,
    ))
    fig.update_layout(**LAYOUT,
        title='Performance by Task Model',
        height=380,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=13)),
        yaxis=dict(tickformat='.0%', title='Average % Correct', range=[0, 1.25],
                   showgrid=True, gridcolor='#EEEEEE', zeroline=False),
    )
    return fig


def fig_skill(df, q, nq):
    import re as _re
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
        q_list = ', '.join(f'Q{qn}' for qn in sorted(sk_qs))
        labels.append(skill_label(sk))
        vals.append(avg)
        hover_texts.append(f'<b>{sk}</b><br>Questions: {q_list}<br>Avg % correct: {avg:.0%}')

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=[pct_color(v) for v in vals],
        marker_line_color='white', marker_line_width=1.5,
        text=[f'{v:.0%}' for v in vals], textposition='outside',
        cliponaxis=False,
        hovertext=hover_texts, hoverinfo='text',
        showlegend=False,
        width=0.4,
    ))
    fig.update_layout(**LAYOUT,
        title='Performance by Skill',
        height=380,
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
            + card(f'{mean_s:.1f} / {nq}', 'Class Average Score',
                   score_color(round(mean_s), nq))
            + card(f'{pct_thr:.0%}', f'Scored {thresh}+ ({thresh/nq:.0%}+)',
                   C['teal'] if pct_thr >= 0.55 else C['orange'])
            + card(str(n), 'Total Students', C['blue'])
            + card(str(n_blank), 'Left a Question Blank',
                   C['orange'] if n_blank > n * 0.10 else C['teal'])
            + '</div>')


def takeaways_html(df, q, std_name, nq):
    # Group questions by content standard — average p-values so each standard appears once
    from collections import defaultdict
    std_groups = defaultdict(list)
    for _, row in q.iterrows():
        qn   = int(row['Q_Num'])
        raw  = row.get('Content_Std', '')
        code = '' if (raw is None or str(raw).strip().lower() in ('nan', 'none', ''))                else str(raw).strip()
        std_groups[code or f'Q{qn}'].append(qn)

    std_items = []
    for code, qns in std_groups.items():
        avg_p = sum(df[f'Q{qn}'].mean() for qn in qns) / len(qns)
        label = std_name(code, 50) if code and not code.startswith('Q') else ''
        # Show which question numbers are grouped
        q_label = ', '.join(f'Q{n}' for n in sorted(qns))
        std_items.append((q_label, avg_p, label))

    reteach = sorted([(ql,p,lbl) for ql,p,lbl in std_items if p < 0.50],      key=lambda x:x[1])
    review  = sorted([(ql,p,lbl) for ql,p,lbl in std_items if 0.50<=p<0.70],  key=lambda x:x[1])
    strong  = sorted([(ql,p,lbl) for ql,p,lbl in std_items if p >= 0.70],     key=lambda x:-x[1])

    def panel(title, subtitle, items, color, bg):
        rows = ''.join(
            f'<tr>'
            f'<td style="padding:5px 8px">{lbl or ql}</td>'
            f'<td style="font-weight:700;color:{color};padding:5px 8px;text-align:right;white-space:nowrap">{p:.0%}</td>'
            f'</tr>'
            for ql, p, lbl in items
        ) or f'<tr><td colspan="2" style="color:#888;padding:8px;font-style:italic">None</td></tr>'
        return (f'<div style="background:{bg};border:2px solid {color};border-radius:10px;'
                f'padding:16px 18px;flex:1;min-width:220px">'
                f'<div style="font-size:1.1rem;font-weight:700;color:{color}">{title}</div>'
                f'<div style="font-size:0.82rem;color:#888;margin-bottom:10px;font-style:italic">{subtitle}</div>'
                f'<table style="width:100%;border-collapse:collapse;font-size:0.9rem">{rows}</table></div>')

    return (f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:20px">'
            + panel('RETEACH', 'Below 50% correct', reteach, C['red'],    '#fdf0f1')
            + panel('REVIEW',  '50–70% correct',    review,  C['orange'], '#fef8f0')
            + panel('STRONG',  'Above 70% correct', strong,  C['teal'],   '#f0fbf8')
            + '</div>')


def to_div(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={'displayModeBar': False, 'responsive': True})


def section(title, anchor, content_html, subtitle=''):
    sub = f'<p class="subtitle">{subtitle}</p>' if subtitle else ''
    return f'<section id="{anchor}"><h2>{title}</h2>{sub}{content_html}</section>'


# ── Report assembly ───────────────────────────────────────────────────────────
def build_html(df, q, std_name, nq):
    mean_s = df['MC_Score'].mean()
    today  = date.today().strftime('%B %d, %Y')

    figs = {
        'hist':       fig_histogram(df, nq),
        'cp_bar':     fig_class_bar(df, nq),
        'cp_dist':    fig_class_dist(df, nq),
        'diff':       fig_difficulty(df, q, std_name, nq),
        'grp':        fig_groups(df, nq),
        'box':        fig_boxplot(df, nq),
        'scatter':    fig_scatter(df, q, std_name, nq),
        'oi':         fig_oi(df, q, nq),
        'task_model': fig_task_model(df, q, nq),
        'skill':      fig_skill(df, q, nq),
    }

    nav_links = '<a href="../../index.html">← Home</a>' + ''.join(
        f'<a href="#{a}">{t}</a>'
        for a, t in [
            ('overview','Overview'), ('class-period','Class & Period'),
            ('difficulty','Difficulty'), ('distractor','Distractors'),
            ('groups','Groups'),
            ('takeaways','Takeaways'), ('technical','Technical'),
        ]
    )

    body = '\n'.join([
        section('1.  At a Glance', 'overview',
            kpi_html(df, nq) + to_div(figs['hist'])),

        section('2.  Class & Period Performance', 'class-period',
            '<div class="chart-grid">'
            + to_div(figs['cp_bar']) + to_div(figs['cp_dist'])
            + '</div>'),

        section('3.  Which Questions Were Hardest?', 'difficulty',
            f'<div class="scroll-x"><div style="min-width:{max(520, NQ * 34)}px">'
            + to_div(figs['diff']) + '</div></div>'),

        section('4.  What Did Students Select?', 'distractor',
            distractor_html(df, q, nq),
),


        section('6.  Student Group Comparison', 'groups',
            to_div(figs['grp']) + subgroup_notes_html(df)),

        section('7.  Key Takeaways', 'takeaways',
            takeaways_html(df, q, std_name, nq)),

        section('Appendix', 'appendix',
            to_div(figs['box']) + to_div(figs['scatter'])
            + (to_div(figs['oi']) if figs['oi'] is not None else '')
            + (to_div(figs['task_model']) if figs['task_model'] is not None else '')
            + (to_div(figs['skill']) if figs['skill'] is not None else ''),
            subtitle=''),
    ])

    HEADER_H = 72
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{TITLE}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
      background: #f0f2f5;
      color: {C["navy"]};
      margin: 0; padding: 0; line-height: 1.5;
    }}
    header {{
      background: {C["navy"]}; color: white;
      padding: 14px 36px; position: sticky; top: 0; z-index: 100;
      display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.25); min-height: {HEADER_H}px;
    }}
    header h1 {{ margin: 0; font-size: 1.15rem; font-weight: 700; }}
    header p  {{ margin: 2px 0 0; font-size: 0.78rem; opacity: 0.72; }}
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
      margin: 0 0 3px; padding-bottom: 10px;
      border-bottom: 3px solid {C["teal"]};
    }}
    section p.subtitle {{
      color: #777; font-size: 0.85rem; margin: 6px 0 16px; font-style: italic;
    }}
    .plotly-graph-div {{ width: 100% !important; }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 16px;
    }}
    .scroll-x {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
    .scroll-x > * {{ min-width: 520px; }}
    .q-btns {{
      display: flex; flex-wrap: wrap; gap: 5px;
      margin-top: 10px; padding: 0 2px;
    }}
    .q-btn {{
      font-size: 0.78rem; padding: 3px 9px; border-radius: 6px;
      border: 1px solid #CCC; background: white; cursor: pointer;
      color: {C["navy"]}; transition: background .1s, color .1s;
    }}
    .q-btn:hover {{ background: #f0f2f5; }}
    .q-btn-active {{ background: {C["navy"]} !important; color: white !important; border-color: {C["navy"]} !important; }}
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
    <div>
      <h1>{TITLE}</h1>
    </div>
    <nav>{nav_links}</nav>
  </header>
  <main>{body}</main>
  <footer>Generated {today}</footer>
  <script>
    // Plotly computes chart sizes before CSS grid finishes layout.
    // Re-fire resize once the grid cells have their final dimensions.
    setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 50);
    setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 300);
  </script>
</body>
</html>'''


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading data...')
    df, q, std_lookup, std_name, answer_cols, correct_cols, NQ = load_data()
    print(f'  {len(df)} students, {df["Class"].nunique()} classes, {NQ} questions')

    print('Building report...')
    html = build_html(df, q, std_name, NQ)

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = len(html.encode('utf-8')) / 1024
    print(f'Done → {OUTPUT}  ({size_kb:.0f} KB)')
    print()
    print('To host on GitHub Pages:')
    print('  1. Commit april_cma_report.html to your repo')
    print('  2. Settings → Pages → Deploy from branch → main / root')
    print('  3. Share: https://<username>.github.io/<repo>/april_cma_report.html')
