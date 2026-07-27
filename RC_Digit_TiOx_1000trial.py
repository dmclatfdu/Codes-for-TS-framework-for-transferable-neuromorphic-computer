#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete three-channel TiOx spoken-digit 1000-trial pipeline.

Stages: plan -> compute -> merge -> process -> fewshot -> plot.
Plans are regenerated from fixed seeds.
10 parallel processing are generated automatically.
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import numpy as np
import pandas as pd
from RC_Voice_Exp_TiOx import (
    TRIAL_DEFAULT_ALPHA as DEFAULT_ALPHA,
    TRIAL_DEVICE_SET as DEVICE_SET,
    TiOx1000VoiceFoldCache as VoiceFoldCache,
    trial_evaluate_family as evaluate_family,
    trial_evaluate_fewshot as evaluate_fewshot,
    trial_load_or_create_cv_split as load_or_create_cv_split,
    trial_load_state_bank as load_state_bank,
    trial_make_cv_folds as make_cv_folds,
)
from RC_TiOx_1000trial_common import (
    DIGIT_DELTA_I,
    combo_string,
    digit_case_average,
    ensure_dir,
    plot_baseline_advantage,
    plot_fewshot_demo,
    plot_fewshot_statistics,
    plot_inout,
    plot_opportunity_gain,
    plot_ts_vs_n,
    prepare_fewshot_delta,
    process_digit_results,
    set_string,
)
N3_SEED = 20260601
EXPAND_SEED = 20260624
FOLD_SEED = 0
N_CASES = 1000
SHOTS = (1, 2, 3, 5, 10)
DEFAULT_OUT = './Data/TiOx_1000trial/Digit'
DEFAULT_FIG = './Figure/TiOx_1000trial/Digit'
# PyCharm one-click configuration
PYCHARM_MODE = 'auto'
PYCHARM_OUT_DIR = DEFAULT_OUT
PYCHARM_FIGURE_DIR = DEFAULT_FIG
PYCHARM_PROJECT_ROOT = '.'
PYCHARM_N_SHARDS = 10
PYCHARM_MAX_PARALLEL = 10
PYCHARM_RECOMPUTE = False
PYCHARM_REBUILD_STATE_BANK = False
# Fixed recommended N=3 cases used by the manuscript workflow
FIXED_20_CASES = (
    (((1, 2, 2), (2, 1, 1), (3, 1, 2)), (0, 1, 2)),
    (((3, 1, 2), (3, 2, 2), (1, 3, 2)), (0, 1, 2)),
    (((1, 2, 2), (2, 1, 1), (3, 1, 2)), (2, 1, 2)),
    (((1, 2, 2), (3, 1, 2), (3, 2, 2)), (2, 1, 2)),
    (((3, 1, 2), (3, 2, 2), (1, 3, 2)), (1, 0, 2)),
    (((1, 2, 2), (2, 1, 1), (2, 1, 2)), (0, 1, 2)),
    (((1, 2, 2), (2, 1, 2), (1, 3, 2)), (1, 0, 2)),
    (((1, 2, 2), (3, 1, 2), (3, 2, 2)), (0, 1, 2)),
    (((1, 2, 2), (2, 1, 2), (3, 2, 2)), (0, 1, 2)),
    (((2, 1, 1), (3, 1, 2), (1, 3, 2)), (0, 1, 2)),
    (((2, 1, 2), (3, 2, 2), (1, 3, 2)), (1, 0, 2)),
    (((1, 2, 2), (3, 1, 2), (2, 1, 2)), (0, 1, 2)),
    (((2, 1, 1), (3, 1, 2), (3, 2, 2)), (0, 1, 2)),
    (((2, 1, 1), (3, 1, 2), (2, 1, 2)), (0, 1, 2)),
    (((1, 2, 2), (2, 1, 1), (3, 2, 2)), (2, 1, 2)),
    (((0, 1, 3), (1, 0, 2), (2, 3, 2)), (2, 1, 2)),
    (((2, 1, 1), (2, 1, 2), (1, 3, 2)), (0, 1, 2)),
    (((1, 2, 2), (3, 1, 2), (1, 3, 2)), (0, 1, 2)),
    (((3, 1, 2), (2, 1, 2), (1, 3, 2)), (1, 0, 2)),
    (((3, 1, 2), (2, 1, 2), (1, 3, 2)), (0, 1, 2)),
)


def tuple3(x):
    t = tuple((int(v) for v in x))
    if len(t) != 3:
        raise ValueError(f'Expected three-channel set, got {x}')
    return t


def pool(n_devices=4):
    return [tuple3(s) for s in itertools.product(
        range(n_devices),
        repeat=3,
    )]


def ordered_key(combo, target):
    return combo_string(combo) + '->' + set_string(target)


def classify_inout(combo, target) -> Dict:
    combo = tuple((tuple3(s) for s in combo))
    target = tuple3(target)
    char = np.asarray([DIGIT_DELTA_I[i] for i in range(4)])
    out = []
    ranges = []
    index = []
    for ch in range(3):
        ids = [s[ch] for s in combo]
        vals = char[ids]
        tid = target[ch]
        tv = float(char[tid])
        lo = float(vals.min())
        hi = float(vals.max())
        index.append(f'ch{ch + 1}:idx{min(ids)}-{max(ids)}/t{tid}')
        ranges.append(f'ch{ch + 1}:{lo:.1f}-{hi:.1f}/t{tv:.1f}')
        if tv < lo - 1e-12 or tv > hi + 1e-12:
            out.append(ch + 1)
    s = 'none' if not out else ','.join(map(str, out))
    q = len(out)
    return {
        'inout_code': q,
        'n_out_channels': q,
        'out_channels': s,
        'channel_ranges': '; '.join(ranges),
        'index_ranges_for_debug': '; '.join(index),
        'inout_print': f"in={q} ({s}; {'; '.join(ranges)})",
    }


def validate(combo, target):
    combo = tuple((tuple3(s) for s in combo))
    target = tuple3(target)
    if len(combo) != 3 or len(set(combo)) != 3 or target in combo:
        raise ValueError(f'Invalid N3 case: {combo}->{target}')


# Plan generation
def generate_n3(n_total=N_CASES, seed=N3_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    sets = pool()
    rows = []
    used = set()
    for combo, target in FIXED_20_CASES:
        combo = tuple((tuple3(s) for s in combo))
        target = tuple3(target)
        validate(combo, target)
        key = ordered_key(combo, target)
        if key in used:
            raise ValueError('Duplicate fixed case')
        used.add(key)
        rows.append({
            'case_id': len(rows) + 1,
            'case_source': 'fixed_recommended20',
            'source_combo': combo_string(combo),
            'source_combo_json': json.dumps([list(s) for s in combo]),
            'target_set': set_string(target),
            'target_set_json': json.dumps(list(target)),
            'unique_train_test_key': key,
            **classify_inout(combo, target),
        })
    while len(rows) < int(n_total):
        combo = tuple(
            (sets[int(i)] for i in rng.choice(
                np.arange(len(sets)),
                size=3,
                replace=False,
            )),
        )
        candidates = [s for s in sets if s not in combo]
        target = candidates[int(rng.integers(0, len(candidates)))]
        key = ordered_key(combo, target)
        if key in used:
            continue
        validate(combo, target)
        used.add(key)
        rows.append({
            'case_id': len(rows) + 1,
            'case_source': 'random_broad_search',
            'source_combo': combo_string(combo),
            'source_combo_json': json.dumps([list(s) for s in combo]),
            'target_set': set_string(target),
            'target_set_json': json.dumps(list(target)),
            'unique_train_test_key': key,
            **classify_inout(combo, target),
        })
    return pd.DataFrame(rows)


def choose_extra(
    rng,
    sets,
    existing,
    target,
):
    candidates = [s for s in sets if s not in set(existing) and s != target]
    return tuple3(candidates[int(rng.integers(0, len(candidates)))])


def expand_plan(n3: pd.DataFrame, seed=EXPAND_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    sets = pool()
    rows = []
    for row in n3.sort_values('case_id').itertuples(index=False):
        cid = int(row.case_id)
        combo3 = tuple((tuple3(s) for s in json.loads(row.source_combo_json)))
        target = tuple3(json.loads(row.target_set_json))
        validate(combo3, target)
        remove = int(rng.integers(0, 3))
        combo2 = tuple((s for i, s in enumerate(combo3) if i != remove))
        add4 = choose_extra(rng, sets, combo3, target)
        combo4 = combo3 + (add4,)
        add5 = choose_extra(rng, sets, combo4, target)
        combo5 = combo4 + (add5,)
        for N, combo in {2: combo2, 3: combo3, 4: combo4, 5: combo5}.items():
            rows.append({
                'base_case_id': cid,
                'expanded_case_id': f'case{cid:04d}_N{N}',
                'case_source': row.case_source,
                'N_source_sets': N,
                'source_combo': combo_string(combo),
                'source_combo_json': json.dumps([list(s) for s in combo]),
                'target_set': set_string(target),
                'target_set_json': json.dumps(list(target)),
                'n3_source_combo': combo_string(combo3),
                'n3_source_combo_json': json.dumps([list(s) for s in combo3]),
                'unique_train_test_key': ordered_key(combo, target),
                'removed_source_pos_for_N2': remove + 1,
                'removed_source_set_for_N2': set_string(combo3[remove]),
                'added_source_set_for_N4': set_string(add4) if N >= 4 else '',
                'added_source_set_for_N5': set_string(add5) if N >= 5 else '',
                **classify_inout(combo, target),
            })
    out = pd.DataFrame(rows)
    out['duplicate_unique_train_test_key'] = out.unique_train_test_key.duplicated(
        keep=False,
    )
    return out


def verify(
    df: pd.DataFrame,
    ref: Path,
    name: str,
) -> Dict:
    report = {'reference_checked': False}
    if ref.exists():
        old = pd.read_csv(ref)
        generated = df.copy()
        common = [c for c in old.columns if c in generated.columns]
        tmp = Path(ref.parent) / f".__verify_{name.replace(' ', '_')}.csv"
        generated.to_csv(tmp, index=False)
        generated = pd.read_csv(tmp)
        tmp.unlink(missing_ok=True)
        ok = len(old) == len(generated) and old[common].astype(str).equals(
            generated[common].astype(str),
        )
        report = {
            'reference_checked': True,
            'reference_path': str(ref.resolve()),
            'exact_match': bool(ok),
            'common_columns': common,
        }
        if not ok:
            raise ValueError(f'Generated {name} plan does not match {ref}')
    return report


def generate_plan(
    out_dir: Union[str, Path],
    n3_seed=N3_SEED,
    expand_seed=EXPAND_SEED,
    n_cases=N_CASES,
    n3_reference: Optional[Union[str, Path]]=None,
    expanded_reference: Optional[Union[str, Path]]=None,
):
    out = ensure_dir(Path(out_dir) / 'plan')
    n3 = generate_n3(n_cases, n3_seed)
    expanded = expand_plan(n3, expand_seed)
    n3_path = out / 'Digit_N3_combo_broad_search_plan_1000.csv'
    ex_path = out / 'Digit_N2345_matched_candidate_plan.csv'
    n3.to_csv(n3_path, index=False)
    expanded.to_csv(ex_path, index=False)
    r1 = verify(
        n3,
        Path(n3_reference) if n3_reference else Path(
            'Digit_N3_combo_broad_search_plan_1000.csv',
        ),
        'Digit N3',
    )
    r2 = verify(
        expanded,
        Path(expanded_reference) if expanded_reference else Path(
            'Digit_N2345_matched_candidate_plan.csv',
        ),
        'Digit expanded',
    )
    (out / 'plan_verification.json').write_text(
        json.dumps(
            {'n3_seed': n3_seed, 'expand_seed': expand_seed, 'n3': r1, 'expanded': r2},
            indent=2,
        ),
        encoding='utf-8',
    )
    return (n3, expanded)


def append_rows(path: Path, rows: List[Dict]):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode='a', header=not path.exists(), index=False)


def complete_pairs(path: Path, expected_rows: int) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=['base_case_id', 'fold', 'display_method'])
    counts = df.groupby(['base_case_id', 'fold']).size()
    return set(((int(a), int(b)) for (a, b), n in counts.items() if n >= expected_rows))


# Baseline computation
def compute(
    out_dir: Union[str, Path],
    project_root: Union[str, Path]='.',
    shard_id=0,
    n_shards=1,
    resume=True,
    rebuild_state_bank=False,
):
    out = Path(out_dir)
    plan_path = out / 'plan' / 'Digit_N2345_matched_candidate_plan.csv'
    if not plan_path.exists():
        generate_plan(out)
    plan = pd.read_csv(plan_path)
    grouped = {int(cid): g for cid, g in plan.groupby('base_case_id')}
    assigned = sorted((cid for cid in grouped if cid % int(n_shards) == int(shard_id)))
    worker = ensure_dir(
        out / 'workers',
    )/ f'Digit_foldlevel_shard{shard_id:02d}_of_{n_shards:02d}.csv'
    if not resume and worker.exists():
        worker.unlink()
    done = complete_pairs(worker, 15) if resume else set()
    bank = load_state_bank(project_root, rebuild=rebuild_state_bank)
    split = load_or_create_cv_split(project_root, FOLD_SEED)
    folds = make_cv_folds(split)
    for fold_info in folds:
        fold_index = int(fold_info['fold'])
        fold = VoiceFoldCache(
            bank,
            fold_info['train_set'],
            fold_info['test_set'],
            project_root,
        )
        for cid in assigned:
            if (cid, fold_index) in done:
                continue
            g = grouped[cid]
            combos = {
                N: tuple(
                    tuple3(source_set)
                    for source_set in json.loads(
                        g[g.N_source_sets == N]
                        .iloc[0]
                        .source_combo_json
                    )
                )
                for N in (2, 3, 4, 5)
            }
            base = g[g.N_source_sets == 3].iloc[0]
            target = tuple3(json.loads(base.target_set_json))
            rows = []
            for r in evaluate_family(fold, combos, target, DEFAULT_ALPHA):
                N = int(r['N_source_sets'])
                combo = combos[N] if N in combos else (combos[3][0],)
                clean = {k: v for k, v in r.items() if k not in {
                    'confusion_matrix',
                    'train_confusion_matrix',
                    'N_source_sets',
                    'display_method',
                }}
                rows.append({
                    'task': 'Digit_TiOx_three_channel_N2345_1000trial',
                    'base_case_id': cid,
                    'case_source': base.case_source,
                    'fold': fold_index,
                    'N_source_sets': N,
                    'display_method': r['display_method'],
                    'source_combo': combo_string(combo),
                    'source_combo_json': json.dumps([list(s) for s in combo]),
                    'target_set': set_string(target),
                    'target_set_json': json.dumps(list(target)),
                    'n3_source_combo': combo_string(combos[3]),
                    'first_source_set_N3': set_string(combos[3][0]),
                    'alpha_policy': '1e-16_except_ridge_CV',
                    'accuracy_percent': r['accuracy'],
                    'recall_percent': r['recall'],
                    'precision_percent': r['precision'],
                    'f1_percent': r['f1'],
                    **clean,
                })
            append_rows(worker, rows)
            print(f'[Digit shard {shard_id}] fold {fold_index} case {cid} complete')
    return worker


def merge(
    out_dir: Union[str, Path],
    n_shards=1,
):
    out = Path(out_dir)
    files = [
        out
        / 'workers'
        / f'Digit_foldlevel_shard{i:02d}_of_{n_shards:02d}.csv'
        for i in range(n_shards)
    ]
    if any((not p.exists() for p in files)):
        raise FileNotFoundError('Some Digit worker files are missing')
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True).drop_duplicates(
        ['base_case_id', 'fold', 'N_source_sets', 'display_method'],
        keep='last',
    )
    fold_path = (
        out
        / 'Digit_N2345_combo_expand_foldlevel_allmetrics_with_single_refs.csv'
    )
    df.sort_values([
        'base_case_id',
        'fold',
        'N_source_sets',
        'display_method',
    ]).to_csv(fold_path, index=False)
    if len(df) != N_CASES * 10 * 15:
        raise ValueError(f'Digit merged rows={len(df)}, expected {N_CASES * 10 * 15}')
    avg = digit_case_average(df)
    avg_path = out / 'Digit_N2345_combo_expand_caseavg_allmetrics_with_single_refs.csv'
    avg.to_csv(avg_path, index=False)
    return (fold_path, avg_path)


def process(out_dir: Union[str, Path]):
    out = Path(out_dir)
    avg = out / 'Digit_N2345_combo_expand_caseavg_allmetrics_with_single_refs.csv'
    if not avg.exists():
        _, avg = merge(out, 1)
    return process_digit_results(avg, ensure_dir(out / 'plotdata'), 88)


def fewshot_complete(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=['base_case_id', 'fold', 'method'])
    counts = df.groupby(['base_case_id', 'fold']).size()
    return set(((int(a), int(b)) for (a, b), n in counts.items() if n >= 11))


def fewshot(
    out_dir: Union[str, Path],
    project_root: Union[str, Path]='.',
    shard_id=0,
    n_shards=1,
    resume=True,
):
    out = Path(out_dir)
    geom = out / 'plotdata' / 'Digit_broad_rule_N3_geometry_all1000.csv'
    if not geom.exists():
        process(out)
    selected = set(
        pd.read_csv(geom).query('selected_broad == True').base_case_id.astype(int),
    )
    plan = pd.read_csv(out / 'plan' / 'Digit_N2345_matched_candidate_plan.csv')
    n3 = plan[plan.N_source_sets == 3].set_index('base_case_id')
    assigned = sorted((cid for cid in selected if cid % int(n_shards) == int(shard_id)))
    worker = ensure_dir(
        out / 'fewshot_workers',
    )/ f'Digit_fewshot_foldlevel_shard{shard_id:02d}_of_{n_shards:02d}.csv'
    if not resume and worker.exists():
        worker.unlink()
    done = fewshot_complete(worker) if resume else set()
    bank = load_state_bank(project_root)
    split = load_or_create_cv_split(project_root, FOLD_SEED)
    for fold_info in make_cv_folds(split):
        fi = int(fold_info['fold'])
        fold = VoiceFoldCache(
            bank,
            fold_info['train_set'],
            fold_info['test_set'],
            project_root,
        )
        for cid in assigned:
            if (cid, fi) in done:
                continue
            row = n3.loc[cid]
            combo = tuple((tuple3(s) for s in json.loads(row.source_combo_json)))
            target = tuple3(json.loads(row.target_set_json))
            rows = []
            for r in evaluate_fewshot(fold, combo, target, SHOTS, DEFAULT_ALPHA):
                clean = {k: v for k, v in r.items() if k not in {
                    'confusion_matrix',
                    'train_confusion_matrix',
                    'accuracy',
                    'recall',
                    'precision',
                    'f1',
                }}
                rows.append({
                    'task': 'Digit',
                    'base_case_id': cid,
                    'fold': fi,
                    'method': r['method'],
                    'fewshot': r['fewshot'],
                    'source_combo': combo_string(combo),
                    'target_set': set_string(target),
                    'accuracy_percent': r['accuracy'],
                    'recall_percent': r['recall'],
                    'precision_percent': r['precision'],
                    'f1_percent': r['f1'],
                    **clean,
                })
            append_rows(worker, rows)
            print(f'[Digit few-shot shard {shard_id}] fold {fi} case {cid} complete')
    return worker


def fewshot_average(df: pd.DataFrame) -> pd.DataFrame:
    group = ['task', 'base_case_id', 'method', 'fewshot', 'source_combo', 'target_set']
    metrics = [f'{m}_percent' for m in ('accuracy', 'recall', 'precision', 'f1')]
    return df.groupby(group, dropna=False)[metrics].mean().reset_index()


def merge_fewshot(
    out_dir: Union[str, Path],
    n_shards=1,
):
    out = Path(out_dir)
    files = [
        out
        / 'fewshot_workers'
        / f'Digit_fewshot_foldlevel_shard{i:02d}_of_{n_shards:02d}.csv'
        for i in range(n_shards)
    ]
    if any((not p.exists() for p in files)):
        raise FileNotFoundError('Some Digit few-shot shards are missing')
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True).drop_duplicates(
        ['base_case_id', 'fold', 'method', 'fewshot'],
        keep='last',
    )
    fold_path = out / 'Digit_fewshot_foldlevel.csv'
    df.to_csv(fold_path, index=False)
    if len(df) != 88 * 10 * 11:
        raise ValueError(f'Digit few-shot rows={len(df)}, expected {88 * 10 * 11}')
    avg = fewshot_average(df)
    avg_path = out / 'Digit_fewshot_caseavg.csv'
    avg.to_csv(avg_path, index=False)
    delta = prepare_fewshot_delta(avg, 'Digit')
    delta.to_csv(out / 'Digit_fewshot_delta_long.csv', index=False)
    return (fold_path, avg_path)


def demo(
    out_dir: Union[str, Path],
    project_root: Union[str, Path]='.',
):
    combo = ((1, 2, 2), (2, 1, 1), (3, 1, 2))
    target = (0, 1, 2)
    bank = load_state_bank(project_root)
    split = load_or_create_cv_split(project_root, FOLD_SEED)
    rows = []
    for fold_info in make_cv_folds(split):
        fi = int(fold_info['fold'])
        fold = VoiceFoldCache(
            bank,
            fold_info['train_set'],
            fold_info['test_set'],
            project_root,
        )
        for r in evaluate_fewshot(fold, combo, target, SHOTS, DEFAULT_ALPHA):
            rows.append({
                'task': 'Digit',
                'base_case_id': 'demo',
                'fold': fi,
                'method': r['method'],
                'fewshot': r['fewshot'],
                'source_combo': combo_string(combo),
                'target_set': set_string(target),
                'accuracy_percent': r['accuracy'],
                'recall_percent': r['recall'],
                'precision_percent': r['precision'],
                'f1_percent': r['f1'],
            })
    avg = fewshot_average(pd.DataFrame(rows))
    path = Path(out_dir) / 'Digit_fewshot_demo.csv'
    avg.to_csv(path, index=False)
    return path


# Plot-only entry point
def plot(
    out_dir: Union[str, Path],
    fig_dir: Union[str, Path],
):
    out = Path(out_dir)
    fig = ensure_dir(fig_dir)
    pd_dir = out / 'plotdata'
    if not (pd_dir / 'Digit_broad_rule_selected_all_rows.csv').exists():
        process(out)
    plot_baseline_advantage(
        pd_dir / 'Digit_broad_rule_selected_all_rows.csv',
        'Digit',
        fig,
    )
    plot_ts_vs_n(pd_dir / 'Digit_broad_rule_selected_all_rows.csv', 'Digit', fig)
    plot_opportunity_gain(
        pd_dir / 'Digit_Fig5f_opportunity_gain_N3_broad_allmetrics.csv',
        'Digit',
        fig,
    )
    plot_inout(
        pd_dir / 'Digit_broad_rule_perN_inout_TS_metrics.csv',
        pd_dir / 'Digit_broad_rule_perN_inout_MWU_summary.csv',
        'Digit',
        fig,
    )
    if (out / 'Digit_fewshot_delta_long.csv').exists():
        plot_fewshot_statistics(out / 'Digit_fewshot_delta_long.csv', 'Digit', fig)
    if (out / 'Digit_fewshot_demo.csv').exists():
        plot_fewshot_demo(out / 'Digit_fewshot_demo.csv', 'Digit', fig)


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(sum((1 for _ in open(path, 'rb'))) - 1)
    except Exception:
        return 0


def _run_one_subprocess(
    cmd: List[str],
    log_path: Path,
    cwd: Path,
) -> Tuple[int, str, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in (
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        env[name] = '1'
    with open(str(log_path), 'w', encoding='utf-8', errors='replace') as log:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return (
        int(cmd[cmd.index('--shard-id') + 1]),
        str(log_path),
        int(completed.returncode),
    )


def run_parallel_shards(
    stage: str,
    out_dir: Union[str, Path],
    figure_dir: Union[str, Path],
    project_root: Union[str, Path],
    n_shards: int=10,
    max_parallel: int=10,
    n3_seed: int=N3_SEED,
    expand_seed: int=EXPAND_SEED,
    n_cases: int=N_CASES,
    recompute: bool=False,
) -> None:
    """Launch independent shard processes.

    Every worker uses the same Python interpreter selected by PyCharm.
    """
    if stage not in ('compute', 'fewshot'):
        raise ValueError('Parallel stage must be compute or fewshot')
    script = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    log_dir = ensure_dir(Path(out_dir) / 'parallel_logs' / stage)
    commands = []
    for shard_id in range(int(n_shards)):
        cmd = [
            sys.executable,
            str(script),
            '--mode',
            stage,
            '--out-dir',
            str(out_dir),
            '--figure-dir',
            str(figure_dir),
            '--project-root',
            str(project_root),
            '--n3-seed',
            str(int(n3_seed)),
            '--expand-seed',
            str(int(expand_seed)),
            '--n-cases',
            str(int(n_cases)),
            '--shard-id',
            str(shard_id),
            '--n-shards',
            str(int(n_shards)),
        ]
        if recompute:
            cmd.append('--recompute')
        commands.append((
            cmd,
            log_dir / f'{stage}_shard{shard_id:02d}_of_{int(n_shards):02d}.log',
        ))
    workers = max(1, min(int(max_parallel), int(n_shards)))
    print(
        f'[Digit auto] Launching {n_shards} {stage} shards '
        f'with max_parallel={workers}.'
    )
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_one_subprocess,
                cmd,
                log,
                cwd,
            )
            for cmd, log in commands
        ]
        for future in as_completed(futures):
            shard_id, log_path, returncode = future.result()
            if returncode == 0:
                print(
                    f'[Digit auto] {stage} shard {shard_id} finished. '
                    f'Log: {log_path}'
                )
            else:
                failures.append((shard_id, log_path, returncode))
                print(f'[Digit auto] {stage} shard {shard_id} FAILED. Log: {log_path}')
    if failures:
        message = '\n'.join(
            (f'shard={sid}, returncode={rc}, log={log}' for sid, log, rc in failures),
        )
        raise RuntimeError(f'Digit parallel {stage} failed:\n{message}')


def auto_run(
    out_dir: Union[str, Path],
    figure_dir: Union[str, Path],
    project_root: Union[str, Path],
    n3_seed: int=N3_SEED,
    expand_seed: int=EXPAND_SEED,
    n_cases: int=N_CASES,
    n_shards: int=10,
    max_parallel: int=10,
    recompute: bool=False,
    rebuild_state_bank: bool=False,
    n3_reference: Optional[Union[str, Path]]=None,
    expanded_reference: Optional[Union[str, Path]]=None,
) -> None:
    """Complete one-click PyCharm workflow with automatic 10-way parallel execution."""
    out = Path(out_dir)
    plan_file = out / 'plan' / 'Digit_N2345_matched_candidate_plan.csv'
    if recompute or not plan_file.exists():
        generate_plan(
            out,
            n3_seed,
            expand_seed,
            n_cases,
            n3_reference,
            expanded_reference,
        )
    merged = out / 'Digit_N2345_combo_expand_foldlevel_allmetrics_with_single_refs.csv'
    baseline_complete = _csv_row_count(merged) == int(n_cases) * 10 * 15
    if recompute or not baseline_complete:
        load_state_bank(project_root, rebuild=rebuild_state_bank)
        load_or_create_cv_split(project_root, FOLD_SEED)
        run_parallel_shards(
            'compute',
            out,
            figure_dir,
            project_root,
            n_shards,
            max_parallel,
            n3_seed,
            expand_seed,
            n_cases,
            recompute,
        )
        merge(out, n_shards)
    else:
        print('[Digit auto] Complete baseline data found; skip baseline workers.')
    process(out)
    fewshot_fold = out / 'Digit_fewshot_foldlevel.csv'
    fewshot_complete_flag = _csv_row_count(fewshot_fold) == 88 * 10 * 11
    if recompute or not fewshot_complete_flag:
        run_parallel_shards(
            'fewshot',
            out,
            figure_dir,
            project_root,
            n_shards,
            max_parallel,
            n3_seed,
            expand_seed,
            n_cases,
            recompute,
        )
        merge_fewshot(out, n_shards)
    else:
        print('[Digit auto] Complete few-shot data found; skip few-shot workers.')
    if recompute or not (out / 'Digit_fewshot_demo.csv').exists():
        demo(out, project_root)
    plot(out, figure_dir)


def main():
    p = argparse.ArgumentParser(
        description=(
            'Digit TiOx 1000-trial pipeline. '
            'PyCharm AUTO mode runs 10 shards in parallel.'
        ),
    )
    p.add_argument(
        '--mode',
        choices=(
            'auto',
            'plan',
            'compute',
            'merge',
            'process',
            'fewshot',
            'merge-fewshot',
            'demo',
            'plot',
            'all',
        ),
        default=PYCHARM_MODE,
    )
    p.add_argument('--out-dir', default=PYCHARM_OUT_DIR)
    p.add_argument('--figure-dir', default=PYCHARM_FIGURE_DIR)
    p.add_argument('--project-root', default=PYCHARM_PROJECT_ROOT)
    p.add_argument('--n3-seed', type=int, default=N3_SEED)
    p.add_argument('--expand-seed', type=int, default=EXPAND_SEED)
    p.add_argument('--n-cases', type=int, default=N_CASES)
    p.add_argument('--shard-id', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=PYCHARM_N_SHARDS)
    p.add_argument('--max-parallel', type=int, default=PYCHARM_MAX_PARALLEL)
    p.add_argument('--n3-reference', default=None)
    p.add_argument('--expanded-reference', default=None)
    p.add_argument('--recompute', action='store_true', default=PYCHARM_RECOMPUTE)
    p.add_argument(
        '--rebuild-state-bank',
        action='store_true',
        default=PYCHARM_REBUILD_STATE_BANK,
    )
    a = p.parse_args()
    out = Path(a.out_dir)
    if a.mode == 'auto':
        auto_run(
            out,
            a.figure_dir,
            a.project_root,
            a.n3_seed,
            a.expand_seed,
            a.n_cases,
            a.n_shards,
            a.max_parallel,
            a.recompute,
            a.rebuild_state_bank,
            a.n3_reference,
            a.expanded_reference,
        )
        return
    if a.mode in ('plan', 'all'):
        generate_plan(
            out,
            a.n3_seed,
            a.expand_seed,
            a.n_cases,
            a.n3_reference,
            a.expanded_reference,
        )
    if a.mode in ('compute', 'all'):
        compute(
            out,
            a.project_root,
            a.shard_id,
            a.n_shards,
            not a.recompute,
            a.rebuild_state_bank,
        )
    if a.mode in ('merge', 'all'):
        merge(out, a.n_shards)
    if a.mode in ('process', 'all'):
        process(out)
    if a.mode in ('fewshot', 'all'):
        fewshot(out, a.project_root, a.shard_id, a.n_shards, not a.recompute)
    if a.mode in ('merge-fewshot', 'all'):
        merge_fewshot(out, a.n_shards)
    if a.mode in ('demo', 'all'):
        demo(out, a.project_root)
    if a.mode in ('plot', 'all'):
        plot(out, a.figure_dir)
if __name__ == '__main__':
    main()
