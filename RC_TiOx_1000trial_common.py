#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared geometry, processing and plotting for the two 1000-trial processing.
"""
from __future__ import annotations
import ast
import json
import re
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu
N_VALUES = (2, 3, 4, 5)
DIGIT_METRICS = ('accuracy', 'recall', 'precision', 'f1')
COMPARATORS = ('state-average', 'ensemble', 'ridge CV')
DIGIT_DELTA_I = {0: 158.0, 1: 130.8, 2: 153.2, 3: 160.1}
MG_DELTA_I = {
    0: 153.2,
    1: 147.7,
    2: 154.1,
    3: 155.5,
    4: 154.9,
    5: 160.1,
    6: 154.0,
    7: 158.0,
    8: 130.8,
}
DIGIT_DIST_MIN_L2_CUTOFF = 2.1
MG_DIST_MIN_L2_CUTOFF = 3.0
CONVEX_ZERO_TOL = 1e-05
# Manuscript plotting palette
STRIP = {
    'g1': (121 / 255, 182 / 255, 101 / 255),
    'g2': (119 / 255, 176 / 255, 120 / 255),
    't1': (116 / 255, 169 / 255, 141 / 255),
    't2': (115 / 255, 163 / 255, 161 / 255),
    'b1': (113 / 255, 156 / 255, 189 / 255),
    'b2': (112 / 255, 150 / 255, 211 / 255),
}
COMP_COLORS = {
    'state-average': STRIP['b2'],
    'ensemble': STRIP['b1'],
    'ridge CV': STRIP['t2'],
}
COMP_LABELS = {
    'state-average': 'state-avg.',
    'ensemble': 'ensemble',
    'ridge CV': 'ridge CV',
}
METRIC_COLORS = {
    'accuracy': STRIP['g2'],
    'recall': STRIP['t1'],
    'precision': STRIP['b1'],
    'f1': STRIP['b2'],
}
METRIC_LABELS = {
    'accuracy': 'Acc.',
    'recall': 'Recall',
    'precision': 'Prec.',
    'f1': 'F1',
}


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def apply_style() -> None:
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.linewidth': 0.5,
        'lines.linewidth': 0.8,
    })


def save_figure(
    fig: plt.Figure,
    out_base: Union[str, Path],
    dpi: int=300,
) -> List[str]:
    out_base = Path(out_base)
    ensure_dir(out_base.parent)
    paths = []
    for ext, kwargs in (('png', {'dpi': dpi}), ('pdf', {}), ('svg', {'dpi': dpi})):
        path = out_base.with_suffix('.' + ext)
        fig.savefig(path, bbox_inches='tight', facecolor='white', **kwargs)
        paths.append(str(path))
    return paths


def tuple_state(x) -> Tuple[int, ...]:
    if isinstance(x, (tuple, list, np.ndarray)):
        return tuple((int(v) for v in x))
    s = str(x).strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, int):
            return (int(val),)
        if isinstance(val, (tuple, list)):
            return tuple((int(v) for v in val))
    except Exception:
        pass
    nums = re.findall('-?\\d+', s)
    if not nums:
        raise ValueError(f'Cannot parse state: {x!r}')
    return tuple((int(v) for v in nums))


def parse_combo(x) -> Tuple[
    Tuple[int, ...],
    ...,
]:
    if isinstance(x, (list, tuple)) and x and isinstance(
        x[0],
        (list, tuple, np.ndarray),
    ):
        return tuple((tuple_state(v) for v in x))
    s = str(x).strip()
    if s.startswith('['):
        try:
            obj = json.loads(s)
            if obj and isinstance(obj[0], (list, tuple)):
                return tuple((tuple_state(v) for v in obj))
        except Exception:
            pass
    return tuple((tuple_state(part) for part in s.split('|') if part.strip()))


def set_string(state: Sequence[int]) -> str:
    return '(' + ','.join(map(str, map(int, state))) + ')'


def combo_string(combo: Sequence[Sequence[int]]) -> str:
    return '|'.join((set_string(s) for s in combo))


def state_vector(
    state: Sequence[int],
    value_map: Dict[int, float],
) -> np.ndarray:
    return np.asarray([value_map[int(v)] for v in state], dtype=float)


def convex_hull_distance(
    target: np.ndarray,
    sources: np.ndarray,
) -> Tuple[float, np.ndarray]:
    target = np.asarray(target, dtype=float)
    sources = np.asarray(sources, dtype=float)
    n = len(sources)
    x0 = np.ones(n, dtype=float) / n

    def objective(lam):
        d = lam @ sources - target
        return float(d @ d)
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=[(0.0, 1.0)] * n,
        constraints=({'type': 'eq', 'fun': lambda lam: np.sum(lam) - 1.0},),
        options={'ftol': 1e-12, 'maxiter': 500, 'disp': False},
    )
    if res.success:
        lam = np.asarray(res.x, dtype=float)
    else:
        lam = np.clip(np.asarray(res.x, dtype=float), 0, 1)
        lam = x0 if lam.sum() <= 0 else lam / lam.sum()
    return (float(np.linalg.norm(lam @ sources - target)), lam)


# Broad geometric rule helpers
def geometry(
    source_combo,
    target_set,
    value_map: Dict[int, float],
    cutoff: float,
) -> Dict:
    combo = parse_combo(source_combo)
    target = tuple_state(target_set)
    S = np.vstack([state_vector(s, value_map) for s in combo])
    T = state_vector(target, value_map)
    dists = np.linalg.norm(S - T[None, :], axis=1)
    conv, lam = convex_hull_distance(T, S)
    pairwise = [
        float(np.linalg.norm(S[i] - S[j]))
        for i in range(len(S))
        for j in range(i + 1, len(S))
    ]
    selected = (
        abs(conv) <= CONVEX_ZERO_TOL
        or float(dists.min()) <= float(cutoff) + 1e-12
    )
    return {
        'geometry_source_combo': combo_string(combo),
        'geometry_target_set': set_string(target),
        'conv_dist_L2': conv,
        'dist_min_L2': float(dists.min()),
        'dist_mean_L2': float(dists.mean()),
        'dist_max_L2': float(dists.max()),
        'lambda_weights': '|'.join((f'{x:.8g}' for x in lam)),
        'source_pairwise_mean_L2': float(np.mean(pairwise)) if pairwise else np.nan,
        'source_pairwise_min_L2': float(np.min(pairwise)) if pairwise else np.nan,
        'source_pairwise_max_L2': float(np.max(pairwise)) if pairwise else np.nan,
        'is_convex': bool(abs(conv) <= CONVEX_ZERO_TOL),
        'selected_broad': bool(selected),
    }


def add_geometry(
    df: pd.DataFrame,
    task: str,
    per_n: bool=False,
) -> pd.DataFrame:
    value_map = MG_DELTA_I if task == 'MG' else DIGIT_DELTA_I
    cutoff = MG_DIST_MIN_L2_CUTOFF if task == 'MG' else DIGIT_DIST_MIN_L2_CUTOFF
    n_col = 'n_source_sets' if 'n_source_sets' in df.columns else 'N_source_sets'
    source_col = 'source_combo'
    target_col = 'target_set'
    key_cols = ['base_case_id', n_col] if per_n else ['base_case_id']
    tmp = df[df[n_col].astype(int).isin(N_VALUES)].copy()
    if not per_n:
        method_col = 'method' if task == 'MG' else 'display_method'
        tmp = tmp[(tmp[n_col].astype(int) == 3) & (tmp[method_col].astype(str) == 'TS')]
    tmp = tmp.drop_duplicates(key_cols)
    rows = []
    for row in tmp.itertuples(index=False):
        g = geometry(
            getattr(row, source_col),
            getattr(row, target_col),
            value_map,
            cutoff,
        )
        out = {'base_case_id': int(getattr(row, 'base_case_id')), **g}
        if per_n:
            out[n_col] = int(getattr(row, n_col))
        rows.append(out)
    return pd.DataFrame(rows)


def normalize_mg_method(x: str) -> str:
    s = str(x)
    mapping = {
        'ridge-CV': 'ridge CV',
        'ridge_cv': 'ridge CV',
        'source-average': 'state-average',
        'self': 'self-training',
        'single_source_1': 'classical',
    }
    return mapping.get(s, s)


def digit_case_average(fold_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [f'{m}_percent' for m in DIGIT_METRICS]
    group = [
        'base_case_id',
        'case_source',
        'N_source_sets',
        'display_method',
        'source_combo',
        'target_set',
        'n3_source_combo',
        'first_source_set_N3',
    ]
    group = [c for c in group if c in fold_df.columns]
    agg = {m: 'mean' for m in metrics}
    agg.update({
        'selected_alpha': 'first',
        'n_train_rows': 'first',
        'n_features': 'first',
    })
    return fold_df.groupby(group, dropna=False).agg(agg).reset_index()


# MG data processing
def process_mg_results(
    results_csv: Union[str, Path],
    out_dir: Union[str, Path],
    expected_selected: int=141,
) -> Dict:
    out_dir = ensure_dir(out_dir)
    df = pd.read_csv(results_csv)
    df['method'] = df['method'].map(normalize_mg_method)
    geom = add_geometry(df, 'MG', per_n=False)
    selected_ids = set(geom.loc[geom.selected_broad, 'base_case_id'])
    if expected_selected is not None and len(selected_ids) != expected_selected:
        raise ValueError(
            f'MG broad rule selected {len(selected_ids)}, expected {expected_selected}',
        )
    selected = df[df.base_case_id.isin(selected_ids)].merge(
        geom,
        on='base_case_id',
        how='left',
    )
    selected = selected.rename(
        columns={'method': 'display_method', 'n_source_sets': 'N_source_sets'},
    )
    baseline = selected[(selected.N_source_sets == 3) & selected.display_method.isin([
        'TS',
        'ensemble',
        'state-average',
    ]) | (selected.N_source_sets == 1) & selected.display_method.eq('ridge CV')]
    num = selected[selected.N_source_sets.between(2, 5) & selected.display_method.isin([
        'TS',
        'ensemble',
        'state-average',
    ])]
    og = []
    for case_id, sub in selected.groupby('base_case_id'):

        def one(method, n):
            x = sub[(sub.display_method == method) & (sub.N_source_sets == n)]
            return None if x.empty else x.iloc[0]
        c, s, t = (one('classical', 1), one('self-training', 1), one('TS', 3))
        if c is None or s is None or t is None:
            continue
        og.append({
            'base_case_id': case_id,
            'classical': c.test_nrmse,
            'self_training': s.test_nrmse,
            'TS': t.test_nrmse,
            'opportunity': c.test_nrmse - s.test_nrmse,
            'gain': c.test_nrmse - t.test_nrmse,
            'recovered_gain': c.test_nrmse - t.test_nrmse,
            'source_combo': t.source_combo,
            'target_set': t.target_set,
            'conv_dist_L2': t.conv_dist_L2,
            'dist_min_L2': t.dist_min_L2,
        })
    og = pd.DataFrame(og)
    paths = {
        'geometry': out_dir / 'MG_broad_rule_N3_geometry_all1000.csv',
        'selected': out_dir / 'MG_broad_rule_selected_all_rows.csv',
        'baseline': out_dir / 'MG_Fig5a_baseline_N3_broad.csv',
        'num': out_dir / 'MG_Fig5c_num_source_sets_N2toN5_broad.csv',
        'og': out_dir / 'MG_Fig5e_opportunity_gain_N3_broad.csv',
    }
    for obj, key in (
        (geom, 'geometry'),
        (selected, 'selected'),
        (baseline, 'baseline'),
        (num, 'num'),
        (og, 'og'),
    ):
        obj.to_csv(paths[key], index=False)
    per_geom = add_geometry(df, 'MG', per_n=True).rename(
        columns={'n_source_sets': 'N_source_sets'},
    )
    per_geom['range_class'] = np.where(
        per_geom.selected_broad,
        'in-range',
        'out-of-range',
    )
    ts = df[(df.method == 'TS') & df.n_source_sets.isin(N_VALUES)][[
        'base_case_id',
        'n_source_sets',
        'test_nrmse',
    ]].drop_duplicates()
    ts = ts.rename(columns={'n_source_sets': 'N_source_sets'}).merge(
        per_geom,
        on=['base_case_id', 'N_source_sets'],
    )
    ts['task'] = 'MG'
    ts['metric'] = 'NRMSE'
    ts['metric_value'] = ts.test_nrmse
    ts.to_csv(out_dir / 'MG_broad_rule_perN_inout_TS_NRMSE.csv', index=False)
    stats = mwu_summary(ts)
    stats.to_csv(out_dir / 'MG_broad_rule_perN_inout_MWU_summary.csv', index=False)
    return {
        'n_selected': len(selected_ids),
        **{k: str(v) for k, v in paths.items()},
    }


# Digit data processing
def process_digit_results(
    caseavg_csv: Union[str, Path],
    out_dir: Union[str, Path],
    expected_selected: int=88,
) -> Dict:
    out_dir = ensure_dir(out_dir)
    df = pd.read_csv(caseavg_csv)
    geom = add_geometry(df, 'Digit', per_n=False)
    selected_ids = set(geom.loc[geom.selected_broad, 'base_case_id'])
    if expected_selected is not None and len(selected_ids) != expected_selected:
        raise ValueError(
            'Digit broad rule selected '
            f'{len(selected_ids)}, expected {expected_selected}',
        )
    selected = df[df.base_case_id.isin(selected_ids)].merge(
        geom,
        on='base_case_id',
        how='left',
    )
    baseline = selected[(selected.N_source_sets == 3) & selected.display_method.isin([
        'TS',
        'ensemble',
        'state-average',
    ]) | (selected.N_source_sets == 1) & selected.display_method.eq('ridge CV')]
    num = selected[selected.N_source_sets.between(2, 5) & selected.display_method.isin([
        'TS',
        'ensemble',
        'state-average',
    ])]
    og = []
    for case_id, sub in selected.groupby('base_case_id'):

        def one(method, n):
            x = sub[(sub.display_method == method) & (sub.N_source_sets == n)]
            return None if x.empty else x.iloc[0]
        c, s, t = (one('classical', 1), one('self-training', 1), one('TS', 3))
        if c is None or s is None or t is None:
            continue
        for metric in DIGIT_METRICS:
            cv = float(c[f'{metric}_percent']) / 100
            sv = float(s[f'{metric}_percent']) / 100
            tv = float(t[f'{metric}_percent']) / 100
            og.append({
                'base_case_id': case_id,
                'metric': metric,
                'classical': cv,
                'self_training': sv,
                'TS': tv,
                'opportunity': sv - cv,
                'gain': tv - cv,
                'recovered_gain': tv - cv,
                'source_combo': t.source_combo,
                'target_set': t.target_set,
                'conv_dist_L2': t.conv_dist_L2,
                'dist_min_L2': t.dist_min_L2,
            })
    og = pd.DataFrame(og)
    paths = {
        'geometry': out_dir / 'Digit_broad_rule_N3_geometry_all1000.csv',
        'selected': out_dir / 'Digit_broad_rule_selected_all_rows.csv',
        'baseline': out_dir / 'Digit_Fig5b_baseline_N3_broad_allmetrics.csv',
        'num': out_dir / 'Digit_Fig5d_num_source_sets_N2toN5_broad_allmetrics.csv',
        'og': out_dir / 'Digit_Fig5f_opportunity_gain_N3_broad_allmetrics.csv',
    }
    for obj, key in (
        (geom, 'geometry'),
        (selected, 'selected'),
        (baseline, 'baseline'),
        (num, 'num'),
        (og, 'og'),
    ):
        obj.to_csv(paths[key], index=False)
    per_geom = add_geometry(df, 'Digit', per_n=True)
    per_geom['range_class'] = np.where(
        per_geom.selected_broad,
        'in-range',
        'out-of-range',
    )
    ts = df[
        (df.display_method == 'TS')
        & df.N_source_sets.isin(N_VALUES)
    ].drop_duplicates([
        'base_case_id',
        'N_source_sets',
    ])
    wide = ts.merge(per_geom, on=['base_case_id', 'N_source_sets'])
    parts = []
    for col, name in (
        ('accuracy_percent', 'accuracy_recall'),
        ('precision_percent', 'precision'),
        ('f1_percent', 'f1'),
    ):
        p = wide.copy()
        p['task'] = 'Digit'
        p['metric'] = name
        p['metric_value'] = p[col]
        parts.append(p)
    per = pd.concat(parts, ignore_index=True)
    per.to_csv(out_dir / 'Digit_broad_rule_perN_inout_TS_metrics.csv', index=False)
    stats = mwu_summary(per)
    stats.to_csv(out_dir / 'Digit_broad_rule_perN_inout_MWU_summary.csv', index=False)
    return {
        'n_selected': len(selected_ids),
        **{k: str(v) for k, v in paths.items()},
    }


def mwu_summary(processed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, metric, N), sub in processed.groupby([
        'task',
        'metric',
        'N_source_sets',
    ]):
        a = sub.loc[
            sub.range_class == 'in-range',
            'metric_value',
        ].dropna().to_numpy(float)
        b = sub.loc[
            sub.range_class == 'out-of-range',
            'metric_value',
        ].dropna().to_numpy(float)
        if len(a) and len(b):
            res = mannwhitneyu(a, b, alternative='two-sided')
            u = float(res.statistic)
            p = float(res.pvalue)
        else:
            u = p = np.nan
        rows.append({
            'task': task,
            'metric': metric,
            'N_source_sets': int(N),
            'n_in': len(a),
            'n_out': len(b),
            'median_in': float(np.median(a)) if len(a) else np.nan,
            'median_out': float(np.median(b)) if len(b) else np.nan,
            'mean_in': float(np.mean(a)) if len(a) else np.nan,
            'mean_out': float(np.mean(b)) if len(b) else np.nan,
            'mannwhitney_u': u,
            'p_two_sided': p,
            'significance': p_stars(p),
        })
    return pd.DataFrame(rows)


def p_stars(p: float) -> str:
    if not np.isfinite(p):
        return 'NA'
    if p < 0.0001:
        return '****'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def style_boxplot(bp, colors) -> None:
    for patch, c in zip(bp['boxes'], colors):
        patch.set(facecolor=c, edgecolor='black', linewidth=0.5, alpha=0.92)
    for key in ('whiskers', 'caps'):
        for item in bp[key]:
            item.set(color='black', linewidth=0.5)
    for item in bp['medians']:
        item.set(color='black', linewidth=0.7)
    for item in bp.get('fliers', []):
        item.set(
            marker='o',
            markersize=2.1,
            markerfacecolor='none',
            markeredgecolor='black',
            markeredgewidth=0.45,
            alpha=0.68,
        )


def finite_ylim(
    arrays: Iterable[np.ndarray],
    include_zero=True,
) -> Tuple[float, float]:
    arrays = list(arrays)
    valid = [np.asarray(a, float)[np.isfinite(a)] for a in arrays if len(a)]
    vals = np.concatenate(valid) if valid else np.array([0.0, 1.0])
    lo, hi = (float(vals.min()), float(vals.max()))
    if include_zero:
        lo = min(lo, 0)
        hi = max(hi, 0)
    span = max(hi - lo, 1e-06)
    return (lo - 0.12 * span, hi + 0.18 * span)


def boxplot_visible_bounds(values: np.ndarray) -> Tuple[float, float]:
    """Return the Tukey-whisker-visible low/high values, excluding fliers."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = float(q3 - q1)
    lo_bound = float(q1 - 1.5 * iqr)
    hi_bound = float(q3 + 1.5 * iqr)
    visible = arr[(arr >= lo_bound) & (arr <= hi_bound)]
    if visible.size == 0:
        visible = arr
    return (float(np.min(visible)), float(np.max(visible)))


def add_pair_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
) -> None:
    """Draw an in-axis significance bracket.

    Keeping the bracket inside the axes prevents tight-bbox from creating
    excessive whitespace.
    """
    ax.plot(
        [x1, x1, x2, x2],
        [y - h, y, y, y - h],
        color='black',
        linewidth=0.55,
        clip_on=True,
        zorder=5,
    )
    ax.text(
        (x1 + x2) / 2.0,
        y + h * 0.18,
        text,
        ha='center',
        va='bottom',
        fontsize=5.5,
        clip_on=True,
        zorder=6,
    )


# Plotting helpers
def plot_baseline_advantage(
    selected_csv: Union[str, Path],
    task: str,
    out_dir: Union[str, Path],
) -> List[str]:
    apply_style()
    df = pd.read_csv(selected_csv)
    out_dir = ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.15), sharey=task == 'MG')
    for ax, N in zip(axes, N_VALUES):
        sub = df[df.N_source_sets.isin([1, N])]
        if task == 'MG':
            wide = sub.pivot_table(
                index='base_case_id',
                columns='display_method',
                values='test_nrmse',
                aggfunc='first',
            )
            data = [(wide[c] - wide['TS']).dropna().to_numpy() for c in COMPARATORS]
            bp = ax.boxplot(
                data,
                positions=range(3),
                widths=0.55,
                patch_artist=True,
                showfliers=False,
            )
            style_boxplot(bp, [COMP_COLORS[c] for c in COMPARATORS])
            ax.set_ylabel('$A$ in NRMSE reduction' if N == 2 else '')
        else:
            centers = np.arange(3)
            offsets = np.linspace(-0.27, 0.27, 4)
            all_data = []
            for mi, m in enumerate(DIGIT_METRICS):
                wide = sub.pivot_table(
                    index='base_case_id',
                    columns='display_method',
                    values=f'{m}_percent',
                    aggfunc='first',
                )
                data = [(wide['TS'] - wide[c]).dropna().to_numpy() for c in COMPARATORS]
                all_data += data
                bp = ax.boxplot(
                    data,
                    positions=centers + offsets[mi],
                    widths=0.12,
                    patch_artist=True,
                    showfliers=False,
                )
                style_boxplot(bp, [METRIC_COLORS[m]] * 3)
            ax.set_ylabel('$A$ in score (%)' if N == 2 else '')
            ax.set_ylim(-50, 100)
        ax.axhline(0, ls='--', lw=0.6, color='.35')
        ax.set_xticks(range(3))
        ax.set_xticklabels([COMP_LABELS[c] for c in COMPARATORS], rotation=15)
        ax.set_title(f'$n={N}$', fontsize=7)
        ax.tick_params(direction='in')
    if task == 'Digit':
        axes[0].legend(
            handles=[
                Patch(
                    facecolor=METRIC_COLORS[metric],
                    edgecolor='black',
                    label=METRIC_LABELS[metric],
                )
                for metric in DIGIT_METRICS
            ],
            frameon=False,
            ncol=4,
            fontsize=5.2,
            loc='lower left',
            bbox_to_anchor=(0, 1.05),
        )
    fig.tight_layout(pad=0.5, w_pad=0.8)
    paths = save_figure(fig, out_dir / f'{task}_TS_advantage_vs_baselines_N2toN5')
    plt.close(fig)
    return paths


def plot_ts_vs_n(
    selected_csv: Union[str, Path],
    task: str,
    out_dir: Union[str, Path],
) -> List[str]:
    apply_style()
    df = pd.read_csv(selected_csv)
    out_dir = ensure_dir(out_dir)
    metrics = ('nrmse',) if task == 'MG' else DIGIT_METRICS
    if task == 'MG':
        fig, axes = plt.subplots(
            1,
            1,
            figsize=(2.8, 2.15),
        )
    else:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(5.2, 4.0),
        )
    axes = np.atleast_1d(axes).ravel()
    for ax, metric in zip(axes, metrics):
        if task == 'MG':
            classical_rows = df[
                (df.N_source_sets == 1)
                & (df.display_method == 'classical')
            ]
            c = classical_rows.set_index('base_case_id').test_nrmse
            data = []
            for N in N_VALUES:
                ts_rows = df[
                    (df.N_source_sets == N)
                    & (df.display_method == 'TS')
                ]
                ts = ts_rows.set_index('base_case_id').test_nrmse
                data.append((c - ts).dropna().to_numpy())
            ylabel = '$A$ in NRMSE reduction'
        else:
            classical_rows = df[
                (df.N_source_sets == 1)
                & (df.display_method == 'classical')
            ]
            c = classical_rows.set_index('base_case_id')[f'{metric}_percent']
            data = []
            for N in N_VALUES:
                ts_rows = df[
                    (df.N_source_sets == N)
                    & (df.display_method == 'TS')
                ]
                ts = ts_rows.set_index('base_case_id')[f'{metric}_percent']
                data.append((ts - c).dropna().to_numpy())
            ylabel = f'$A$ in {METRIC_LABELS[metric]} (%)'
        pos = np.arange(4)
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
        )
        style_boxplot(bp, [STRIP['g2']] * 4)
        for patch in bp['boxes']:
            patch.set(zorder=1)
        for key in ('whiskers', 'caps', 'medians'):
            for artist in bp[key]:
                artist.set(zorder=2)
        med = [np.median(x) for x in data]
        mean = [np.mean(x) for x in data]
        ax.plot(
            pos,
            med,
            color='.1',
            lw=1.0,
            zorder=7,
            label='Median',
        )
        ax.scatter(
            pos,
            med,
            color='white',
            edgecolors='.1',
            s=22,
            linewidths=0.9,
            zorder=8,
        )
        ax.plot(
            pos,
            mean,
            color=STRIP['b2'],
            lw=1.0,
            ls='--',
            zorder=7,
            label='Mean',
        )
        ax.scatter(
            pos,
            mean,
            color='white',
            edgecolors=STRIP['b2'],
            marker='s',
            s=22,
            linewidths=0.9,
            zorder=8,
        )
        ax.axhline(0, ls='--', lw=0.6, color='.35')
        ax.set_xticks(pos)
        ax.set_xticklabels(N_VALUES)
        ax.set_xlabel('$n$ (device sets)')
        ax.set_ylabel(ylabel)
        ax.tick_params(direction='in')
        ax.legend(
            frameon=False,
            fontsize=5.2,
            ncol=2,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout(pad=0.55, w_pad=0.8, h_pad=1.0)
    paths = save_figure(fig, out_dir / f'{task}_TS_vs_classical_by_N')
    plt.close(fig)
    return paths


def plot_opportunity_gain(
    og_csv: Union[str, Path],
    task: str,
    out_dir: Union[str, Path],
) -> List[str]:
    apply_style()
    df = pd.read_csv(og_csv)
    out_dir = ensure_dir(out_dir)
    metrics = (None,) if task == 'MG' else DIGIT_METRICS
    fig, axes = plt.subplots(
        1,
        1,
        figsize=(2.55, 2.25),
    ) if task == 'MG' else plt.subplots(2, 2, figsize=(5.1, 4.3))
    axes = np.atleast_1d(axes).ravel()
    for ax, metric in zip(axes, metrics):
        sub = df if metric is None else df[df.metric == metric]
        x = sub.opportunity.to_numpy(float)
        y = sub.gain.to_numpy(float)
        pos = y > 0
        ax.scatter(x[~pos], y[~pos], s=10, color=STRIP['g2'], edgecolors='none')
        ax.scatter(x[pos], y[pos], s=10, color=STRIP['b2'], edgecolors='none')
        lo = min(np.nanmin(x), np.nanmin(y), 0)
        hi = max(np.nanmax(x), np.nanmax(y), 0)
        span = max(hi - lo, 0.02)
        lo -= 0.08 * span
        hi += 0.08 * span
        ax.plot([lo, hi], [lo, hi], ls='--', lw=0.7, color='.55')
        ax.axhline(0, ls='--', lw=0.5, color='black')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('Opportunity')
        ax.set_ylabel('Gain')
        if metric:
            ax.set_title(METRIC_LABELS[metric], fontsize=7)
        ax.tick_params(direction='in')
    fig.tight_layout(pad=0.6, w_pad=0.8, h_pad=0.8)
    paths = save_figure(fig, out_dir / f'{task}_opportunity_gain')
    plt.close(fig)
    return paths


def prepare_fewshot_delta(raw: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == 'MG':
        metric_cols = ['test_nrmse']
    else:
        metric_cols = [f'{metric}_percent' for metric in DIGIT_METRICS]
    rows = []
    for case_id, sub in raw.groupby('base_case_id'):
        zero = sub[(sub.method == 'TS-zeroshot') & (sub.fewshot == 0)]
        if zero.empty:
            continue
        for col in metric_cols:
            z = float(zero.iloc[0][col])
            metric = 'nrmse' if task == 'MG' else col.replace('_percent', '')
            for shot in sorted(set(sub.fewshot) - {0}):
                cur = sub[sub.fewshot == shot]
                ts = cur[cur.method == 'TS-fewshot']
                cl = cur[cur.method == 'classical-fewshot']
                if ts.empty or cl.empty:
                    continue
                tv = float(ts.iloc[0][col])
                cv = float(cl.iloc[0][col])
                d1 = cv - tv
                d2 = cv - z
                for typ, val in (
                    ('delta1_classical_fewshot_minus_TS_fewshot', d1),
                    ('delta2_classical_fewshot_minus_TS_zeroshot', d2),
                ):
                    advantage = val if task == 'MG' else -val
                    rows.append({
                        'task': task,
                        'base_case_id': case_id,
                        'fewshot': shot,
                        'metric': metric,
                        'delta_type': typ,
                        'delta_value': val,
                        'advantage_value': advantage,
                        'classical_fewshot_value': cv,
                        'TS_fewshot_value': tv,
                        'TS_zeroshot_value': z,
                    })
    return pd.DataFrame(rows)


def plot_fewshot_statistics(
    delta_csv: Union[str, Path],
    task: str,
    out_dir: Union[str, Path],
) -> List[str]:
    apply_style()
    df = pd.read_csv(delta_csv)
    out_dir = ensure_dir(out_dir)
    metrics = ('nrmse',) if task == 'MG' else ('accuracy', 'precision', 'f1')
    if task == 'MG':
        fig, axes = plt.subplots(
            1,
            1,
            figsize=(2.9, 2.15),
        )
    else:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(7.4, 2.25),
        )
    axes = np.atleast_1d(axes).ravel()
    types = (
        'delta1_classical_fewshot_minus_TS_fewshot',
        'delta2_classical_fewshot_minus_TS_zeroshot',
    )
    colors = (STRIP['t1'], STRIP['b1'])
    labels = (r'$\Delta_{\rm TS+fs}$', r'$\Delta_{\rm TS-zero}$')
    flierprops = dict(
        marker='o',
        markersize=1.55,
        markerfacecolor='black',
        markeredgecolor='black',
        markeredgewidth=0.35,
        alpha=0.82,
    )
    for ax, metric in zip(axes, metrics):
        sub = df[df.metric == metric]
        shots = sorted(sub.fewshot.unique())
        centers = np.arange(len(shots))
        offsets = (-0.16, 0.16)
        arrays = []
        for ti, typ in enumerate(types):
            data = [
                sub[
                    (sub.fewshot == shot)
                    & (sub.delta_type == typ)
                ].advantage_value.to_numpy(float)
                for shot in shots
            ]
            arrays += data
            bp = ax.boxplot(
                data,
                positions=centers + offsets[ti],
                widths=0.27,
                patch_artist=True,
                showfliers=True,
                manage_ticks=False,
                flierprops=flierprops,
            )
            style_boxplot(bp, [colors[ti]] * len(data))
            med = [
                np.median(arr)
                if len(np.asarray(arr, float)[np.isfinite(arr)]) else np.nan
                for arr in data
            ]
            line_x = centers + offsets[ti]
            ax.plot(
                line_x,
                med,
                color=colors[ti],
                lw=0.95,
                zorder=7,
            )
            ax.scatter(
                line_x,
                med,
                color='white',
                edgecolors=colors[ti],
                s=17,
                linewidths=0.85,
                zorder=8,
            )
        ax.axhline(0, ls='--', lw=0.6, color='.35')
        ax.set_xticks(centers)
        ax.set_xticklabels(shots)
        ax.set_xlabel('Few-shot count' if task == 'MG' else 'Shots per digit')
        ax.set_ylabel(
            (
                'Advantage in NRMSE'
                if task == 'MG'
                else f'Advantage in {METRIC_LABELS[metric]} (%)'
            ),
        )
        if task == 'MG':
            ax.set_ylim(-0.06, 0.16)
        else:
            ax.set_ylim(*finite_ylim(arrays))
        ax.tick_params(direction='in')
        ax.legend(
            handles=[
                Patch(
                    facecolor=color,
                    edgecolor='black',
                    label=label,
                )
                for color, label in zip(colors, labels)
            ],
            frameon=False,
            fontsize=5.3,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
        )
    fig.tight_layout(pad=0.55, w_pad=0.9)
    paths = save_figure(fig, out_dir / f'{task}_fewshot_delta_statistics')
    plt.close(fig)
    return paths


def plot_fewshot_demo(
    demo_csv: Union[str, Path],
    task: str,
    out_dir: Union[str, Path],
) -> List[str]:
    apply_style()
    df = pd.read_csv(demo_csv)
    out_dir = ensure_dir(out_dir)
    metrics = ('test_nrmse',) if task == 'MG' else tuple(
        (f'{m}_percent' for m in DIGIT_METRICS),
    )
    fig, axes = plt.subplots(
        1,
        1,
        figsize=(2.65, 1.95),
    ) if task == 'MG' else plt.subplots(2, 2, figsize=(4.7, 3.65))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, metrics):
        for method, color in (
            ('TS-fewshot', STRIP['g2']),
            ('classical-fewshot', STRIP['b2']),
        ):
            g = df[df.method == method].sort_values('fewshot')
            ax.plot(
                g.fewshot,
                g[col],
                marker='o',
                ms=2.5,
                lw=0.8,
                color=color,
                label=method,
            )
        z = df[(df.method == 'TS-zeroshot') & (df.fewshot == 0)]
        if not z.empty:
            ax.axhline(
                float(z.iloc[0][col]),
                ls='--',
                lw=0.7,
                color=STRIP['t2'],
                label='TS-zeroshot',
            )
        ax.set_xlabel('Few-shot count' if task == 'MG' else 'Shots per digit')
        ax.set_ylabel(
            'NRMSE' if task == 'MG' else METRIC_LABELS[col.replace(
                '_percent',
                '',
            )] + ' (%)',
        )
        ax.tick_params(direction='in')
        ax.legend(frameon=False, fontsize=5.1)
    fig.tight_layout(pad=0.5, w_pad=0.75, h_pad=0.85)
    paths = save_figure(fig, out_dir / f'{task}_fewshot_demo')
    plt.close(fig)
    return paths


def plot_inout(
    processed_csv: Union[str, Path],
    stats_csv: Union[str, Path],
    task: str,
    out_dir: Union[str, Path],
) -> List[str]:
    apply_style()
    data = pd.read_csv(processed_csv)
    stats = pd.read_csv(stats_csv)
    out_dir = ensure_dir(out_dir)
    metrics = ('NRMSE',) if task == 'MG' else ('accuracy_recall', 'precision', 'f1')
    if task == 'MG':
        fig, axes = plt.subplots(
            len(metrics),
            1,
            figsize=(8.4, 2.1 * len(metrics)),
            squeeze=False,
        )
    else:
        fig, axes = plt.subplots(
            len(metrics),
            1,
            figsize=(9.2, 2.28 * len(metrics)),
            squeeze=False,
        )
    axes = axes.ravel()
    legend_handles = [
        Patch(facecolor=STRIP['g2'], edgecolor='black', label='In-range'),
        Patch(facecolor=STRIP['b2'], edgecolor='black', label='Out-of-range'),
    ]
    for ax, metric in zip(axes, metrics):
        pair_data = []
        lows = []
        highs = []
        for N in N_VALUES:
            sub = data[(data.metric == metric) & (data.N_source_sets == N)]
            a = sub[sub.range_class == 'in-range'].metric_value.to_numpy(float)
            b = sub[sub.range_class == 'out-of-range'].metric_value.to_numpy(float)
            pair_data.append((N, a, b))
            for arr in (a, b):
                lo, hi = boxplot_visible_bounds(arr)
                if np.isfinite(lo):
                    lows.append(lo)
                if np.isfinite(hi):
                    highs.append(hi)
        if task == 'MG':
            if lows and highs:
                lo = min(lows)
                hi = max(highs)
                span = max(hi - lo, 1e-06)
                y0 = max(0.0, lo - 0.08 * span)
                y1 = hi + 0.24 * span
            else:
                y0, y1 = (0.0, 1.0)
        else:
            y0, y1 = (0.0, 106.0)
        ax.set_ylim(y0, y1)
        yr = max(y1 - y0, 1e-09)
        for i, (N, a, b) in enumerate(pair_data, 1):
            pos = [i - 0.17, i + 0.17]
            bp = ax.boxplot(
                [a, b],
                positions=pos,
                widths=0.24,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )
            style_boxplot(bp, [STRIP['g2'], STRIP['b2']])
            st = stats[(stats.metric == metric) & (stats.N_source_sets == N)]
            if not st.empty:
                _, hia = boxplot_visible_bounds(a)
                _, hib = boxplot_visible_bounds(b)
                finite_pair = [v for v in (hia, hib) if np.isfinite(v)]
                if finite_pair:
                    pair_hi = max(finite_pair)
                else:
                    pair_hi = y0 + 0.55 * yr
                if task == 'MG':
                    bracket_y = min(pair_hi + 0.07 * yr, y1 - 0.055 * yr)
                    bracket_h = 0.018 * yr
                else:
                    bracket_y = min(pair_hi + 0.032 * yr, y1 - 0.028 * yr)
                    bracket_h = 0.010 * yr
                add_pair_bracket(
                    ax,
                    pos[0],
                    pos[1],
                    bracket_y,
                    bracket_h,
                    str(st.iloc[0].significance),
                )
        ax.set_xlim(0.45, 4.55)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels([f'$n={n}$' for n in N_VALUES])
        if task == 'Digit':
            ax.set_yticks([10, 40, 70, 100])
        ax.set_ylabel(
            'MG TS NRMSE' if task == 'MG' else {
                'accuracy_recall': 'Digit TS acc. / rec. (%)',
                'precision': 'Digit TS precision (%)',
                'f1': 'Digit TS F1 (%)',
            }[metric],
        )
        ax.tick_params(direction='in')
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=2,
        fontsize=5.8,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.01),
    )
    axes[-1].set_xlabel('Number of source sets used in transfer ($n$)')
    if task == 'MG':
        fig.tight_layout(pad=0.65, h_pad=0.7, rect=(0.0, 0.0, 1.0, 0.94))
    else:
        fig.tight_layout(pad=0.72, h_pad=0.95, rect=(0.0, 0.0, 1.0, 0.965))
    paths = save_figure(fig, out_dir / f'{task}_broad_rule_inout_perN')
    plt.close(fig)
    return paths


