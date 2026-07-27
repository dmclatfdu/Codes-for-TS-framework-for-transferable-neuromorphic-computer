#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete three-channel TiOx MG 1000-trial pipeline.

Stages: plan -> compute -> merge -> process -> fewshot -> plot.
All plans are regenerated from fixed seeds.
"""
from __future__ import annotations
import argparse
import itertools
import json
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
from RC_MG import (
    DEFAULT_ALPHA,
    MG_DEVICE_CODE,
    MG_DEVICE_IDS,
    MG_FEWSHOT_STOP,
    MGEvaluator,
    load_state_bank,
)
from RC_TiOx_1000trial_common import (
    MG_DELTA_I,
    combo_string,
    ensure_dir,
    geometry,
    parse_combo,
    plot_baseline_advantage,
    plot_fewshot_demo,
    plot_fewshot_statistics,
    plot_inout,
    plot_opportunity_gain,
    plot_ts_vs_n,
    prepare_fewshot_delta,
    process_mg_results,
    set_string,
)
PLAN_SEED = 20260316
N_CASES = 1000
FEWSHOTS = (5, 10, 20, 40, 80)
DEFAULT_OUT = './Data/TiOx_1000trial/MG'
DEFAULT_FIG = './Figure/TiOx_1000trial/MG'
# PyCharm one-click configuration
PYCHARM_MODE = 'auto'
PYCHARM_OUT_DIR = DEFAULT_OUT
PYCHARM_FIGURE_DIR = DEFAULT_FIG
PYCHARM_DATA_ROOT = './Data/MG readin'
PYCHARM_N_SHARDS = 1
PYCHARM_RECOMPUTE = False
PYCHARM_REBUILD_STATE_BANK = False


def set_serial_str(s):
    return '(' + ','.join((str(int(x) + 1) for x in s)) + ')'


def combo_serial_str(c):
    return '|'.join((set_serial_str(s) for s in c))


def device_set_code(s):
    return '-'.join((MG_DEVICE_CODE[int(x)] for x in s))


def combo_code(c):
    return '|'.join((device_set_code(s) for s in c))


def delta_i(i):
    return float(MG_DELTA_I[int(i)])


def physical_summary(source_combo, target_set) -> Dict:
    row = {}
    ranges = []
    out = []
    for ch in range(3):
        ids = [int(s[ch]) for s in source_combo]
        vals = [delta_i(i) for i in ids]
        tid = int(target_set[ch])
        tv = delta_i(tid)
        lo, hi = (float(min(vals)), float(max(vals)))
        if tv < lo - 1e-12:
            direction = 'low'
        elif tv > hi + 1e-12:
            direction = 'high'
        else:
            direction = 'inside'
        if direction != 'inside':
            out.append(ch + 1)
        name = f'ch{ch + 1}'
        ranges.append(f'{name}:{lo:.1f}-{hi:.1f}/t{tv:.1f}')
        row.update({
            f'source_{name}_device_ids': ','.join(map(str, ids)),
            f'source_{name}_device_serials': ','.join((str(x + 1) for x in ids)),
            f'source_{name}_device_codes': ','.join((MG_DEVICE_CODE[x] for x in ids)),
            f'source_{name}_deltaI_values_uA': ','.join((f'{v:.1f}' for v in vals)),
            f'source_{name}_deltaI_min_uA': lo,
            f'source_{name}_deltaI_max_uA': hi,
            f'source_{name}_deltaI_span_uA': hi - lo,
            f'target_{name}_device_id': tid,
            f'target_{name}_device_serial': tid + 1,
            f'target_{name}_device_code': MG_DEVICE_CODE[tid],
            f'target_{name}_deltaI_uA': tv,
            f'target_{name}_outside_source_deltaI_range': direction != 'inside',
            f'target_{name}_outside_direction': direction,
        })
    out_s = 'none' if not out else ','.join(map(str, out))
    n = len(out)
    row.update({
        'inout_code': n,
        'n_out_channels': n,
        'n_in_channels': 3 - n,
        'out_channels': out_s,
        'channel_ranges': '; '.join(ranges),
        'inout_print': f"out={n} ({out_s}; {'; '.join(ranges)})",
        'mean_source_deltaI_span_uA': float(
            np.mean([row[f'source_ch{i}_deltaI_span_uA'] for i in (1, 2, 3)]),
        ),
        'max_source_deltaI_span_uA': float(
            np.max([row[f'source_ch{i}_deltaI_span_uA'] for i in (1, 2, 3)]),
        ),
    })
    return row


def device_pool() -> List[Tuple[
    int,
    int,
    int,
]]:
    return [tuple(map(int, s)) for s in itertools.product(
        MG_DEVICE_IDS,
        repeat=3,
    )]


def choose_extra(
    rng,
    pool,
    existing,
    target,
):
    used = {tuple(s) for s in existing}
    target = tuple(target)
    candidates = [s for s in pool if s not in used and s != target]
    return tuple(candidates[int(rng.integers(0, len(candidates)))])


# Plan generation
def generate_plan(
    out_dir: Union[str, Path],
    seed: int=PLAN_SEED,
    n_cases: int=N_CASES,
    reference_csv: Optional[Union[str, Path]]=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = ensure_dir(Path(out_dir) / 'plan')
    rng = np.random.default_rng(int(seed))
    pool = device_pool()
    seen = set()
    wide_rows = []
    long_rows = []
    while len(wide_rows) < int(n_cases):
        target = tuple(pool[int(rng.integers(0, len(pool)))])
        candidates = [s for s in pool if s != target]
        combo3 = tuple(
            (candidates[int(i)] for i in rng.choice(
                len(candidates),
                size=3,
                replace=False,
            )),
        )
        if (combo3, target) in seen:
            continue
        seen.add((combo3, target))
        case_id = len(wide_rows)
        remove = int(rng.integers(0, 3))
        combo2 = tuple((s for i, s in enumerate(combo3) if i != remove))
        add4 = choose_extra(rng, pool, combo3, target)
        combo4 = combo3 + (add4,)
        add5 = choose_extra(rng, pool, combo4, target)
        combo5 = combo4 + (add5,)
        combos = {2: combo2, 3: combo3, 4: combo4, 5: combo5}
        wide_rows.append({
            'base_case_id': case_id,
            'seed_master': int(seed),
            'target_set_json': json.dumps(list(target)),
            'target_set': set_string(target),
            'target_set_serial': set_serial_str(target),
            'target_set_code': device_set_code(target),
            **{
                f'source_combo_N{N}_json': json.dumps(
                    [list(source_set) for source_set in combos[N]]
                )
                for N in (2, 3, 4, 5)
            },
            **{f'source_combo_N{N}': combo_string(combos[N]) for N in (2, 3, 4, 5)},
            **{f'source_combo_N{N}_serial': combo_serial_str(combos[N]) for N in (
                2,
                3,
                4,
                5,
            )},
            **{f'source_combo_N{N}_code': combo_code(combos[N]) for N in (2, 3, 4, 5)},
            'removed_source_index_for_N2_from_N3': remove,
            'added_source_set_N4_json': json.dumps(list(add4)),
            'added_source_set_N5_json': json.dumps(list(add5)),
            'first_source_set_N3_json': json.dumps(list(combo3[0])),
            'first_source_set_N3': set_string(combo3[0]),
            'first_source_set_N3_serial': set_serial_str(combo3[0]),
            'first_source_set_N3_code': device_set_code(combo3[0]),
        })
        for N in (2, 3, 4, 5):
            combo = combos[N]
            long_rows.append({
                'base_case_id': case_id,
                'n_source_sets': N,
                'source_combo_json': json.dumps([list(s) for s in combo]),
                'target_set_json': json.dumps(list(target)),
                'source_combo': combo_string(combo),
                'target_set': set_string(target),
                'source_combo_serial': combo_serial_str(combo),
                'target_set_serial': set_serial_str(target),
                'source_combo_code': combo_code(combo),
                'target_set_code': device_set_code(target),
                'source_combo_N3_json': json.dumps([list(s) for s in combo3]),
                'source_combo_N3': combo_string(combo3),
                'first_source_set_N3_json': json.dumps(list(combo3[0])),
                'first_source_set_N3': set_string(combo3[0]),
                'case_source': 'MG_N2345_random_from_N3',
                **physical_summary(combo, target),
            })
    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)
    long_path = out / 'MG_N2345_combo_expand_plan_long.csv'
    wide_path = out / 'MG_N2345_combo_expand_plan_wide.csv'
    long.to_csv(long_path, index=False)
    wide.to_csv(wide_path, index=False)
    (out / 'MG_N2345_combo_expand_config.json').write_text(
        json.dumps({
            'n_cases': n_cases,
            'seed_master': seed,
            'n_values': [2, 3, 4, 5],
            'device_ids': list(MG_DEVICE_IDS),
            'device_codes': list(MG_DEVICE_CODE),
            'delta_I_uA_by_device_id': MG_DELTA_I,
        }, indent=2),
        encoding='utf-8',
    )
    report = {
        'generated_rows': len(long),
        'generated_cases': len(wide),
        'seed': seed,
        'reference_checked': False,
    }
    ref = Path(reference_csv) if reference_csv else Path(
        'MG_N2345_combo_expand_plan_long.csv',
    )
    if ref.exists():
        old = pd.read_csv(ref)
        generated = pd.read_csv(long_path)
        common = [c for c in old.columns if c in generated.columns]
        ok = len(old) == len(generated) and old[common].astype(str).equals(
            generated[common].astype(str),
        )
        report.update({
            'reference_checked': True,
            'reference_path': str(ref.resolve()),
            'exact_match': bool(ok),
        })
        if not ok:
            raise ValueError(f'Generated MG plan does not match {ref}')
    (out / 'plan_verification.json').write_text(
        json.dumps(report, indent=2),
        encoding='utf-8',
    )
    return (wide, long)


def append_rows(path: Path, rows: List[Dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode='a', header=not path.exists(), index=False)


# Baseline computation
def compute(
    out_dir: Union[str, Path],
    data_root: Union[str, Path],
    shard_id: int=0,
    n_shards: int=1,
    resume: bool=True,
    rebuild_state_bank: bool=False,
) -> Path:
    out = Path(out_dir)
    plan_path = out / 'plan' / 'MG_N2345_combo_expand_plan_wide.csv'
    if not plan_path.exists():
        generate_plan(out)
    plan = pd.read_csv(plan_path)
    worker = ensure_dir(
        out / 'workers',
    )/ f'MG_results_shard{shard_id:02d}_of_{n_shards:02d}.csv'
    if not resume and worker.exists():
        worker.unlink()
    done = set()
    if resume and worker.exists():
        old = pd.read_csv(worker)
        done = set(
            old.groupby('base_case_id')
            .filter(lambda group: len(group) >= 15)
            .base_case_id.astype(int),
        )
    bank, target = load_state_bank(data_root, rebuild_state_bank)
    ev = MGEvaluator(bank, target, alpha=DEFAULT_ALPHA)
    assigned = plan[plan.base_case_id.astype(int).mod(int(n_shards)).eq(int(shard_id))]
    for row in assigned.itertuples(index=False):
        cid = int(row.base_case_id)
        if cid in done:
            continue
        combos = {
            N: tuple(
                tuple(int(value) for value in source_set)
                for source_set in json.loads(
                    getattr(row, f'source_combo_N{N}_json')
                )
            )
            for N in (2, 3, 4, 5)
        }
        target_set = tuple(json.loads(row.target_set_json))
        results = ev.evaluate_family(combos, target_set)
        rows = []
        for r in results:
            N = int(r['n_source_sets'])
            combo = combos[N] if N in combos else (combos[3][0],)
            rows.append({
                'task': 'MG_MC_TiOx_1000trial',
                'base_case_id': cid,
                'n_source_sets': N,
                'reference_N_source_sets': r.get('reference_N_source_sets', N),
                'method': r['method'],
                'source_combo': combo_string(combo),
                'source_combo_json': json.dumps([list(s) for s in combo]),
                'source_combo_N3': combo_string(combos[3]),
                'target_set': set_string(target_set),
                'target_set_json': json.dumps(list(target_set)),
                'first_source_set_N3': set_string(combos[3][0]),
                'alpha_policy': '1e-16_except_ridge_CV',
                **{k: v for k, v in r.items() if k not in {
                    'method',
                    'n_source_sets',
                    'reference_N_source_sets',
                }},
            })
        append_rows(worker, rows)
        print(f'[MG shard {shard_id}] case {cid} complete')
    return worker


def merge(
    out_dir: Union[str, Path],
    n_shards: int=1,
) -> Path:
    out = Path(out_dir)
    files = [
        out
        / 'workers'
        / f'MG_results_shard{i:02d}_of_{n_shards:02d}.csv'
        for i in range(n_shards)
    ]
    missing = [p for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing worker files:\n' + '\n'.join(map(
            str,
            missing,
        )))
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True).drop_duplicates(
        ['base_case_id', 'n_source_sets', 'method'],
        keep='last',
    )
    path = out / 'MG_N2345_combo_expand_results_long.csv'
    df.sort_values([
        'base_case_id',
        'n_source_sets',
        'method',
    ]).to_csv(path, index=False)
    expected = N_CASES * 15
    if len(df) != expected:
        raise ValueError(f'Merged MG rows={len(df)}, expected {expected}')
    return path


def process(out_dir: Union[str, Path]) -> Dict:
    out = Path(out_dir)
    result = out / 'MG_N2345_combo_expand_results_long.csv'
    if not result.exists():
        result = merge(out, 1)
    return process_mg_results(result, ensure_dir(out / 'plotdata'), 141)


def fewshot(
    out_dir: Union[str, Path],
    data_root: Union[str, Path],
    shard_id: int=0,
    n_shards: int=1,
    resume: bool=True,
) -> Path:
    out = Path(out_dir)
    geom_path = out / 'plotdata' / 'MG_broad_rule_N3_geometry_all1000.csv'
    if not geom_path.exists():
        process(out)
    selected = set(
        pd.read_csv(geom_path).query('selected_broad == True').base_case_id.astype(int),
    )
    plan = pd.read_csv(out / 'plan' / 'MG_N2345_combo_expand_plan_wide.csv')
    plan = plan[plan.base_case_id.isin(selected)]
    worker = ensure_dir(
        out / 'fewshot_workers',
    )/ f'MG_fewshot_shard{shard_id:02d}_of_{n_shards:02d}.csv'
    if not resume and worker.exists():
        worker.unlink()
    done = set()
    if resume and worker.exists():
        old = pd.read_csv(worker)
        done = set(
            old.groupby('base_case_id')
            .filter(lambda group: len(group) >= 11)
            .base_case_id.astype(int),
        )
    bank, target = load_state_bank(data_root)
    ev = MGEvaluator(bank, target, alpha=DEFAULT_ALPHA)
    assigned_plan = plan[
        plan.base_case_id.astype(int).mod(n_shards).eq(shard_id)
    ]
    for row in assigned_plan.itertuples(index=False):
        cid = int(row.base_case_id)
        if cid in done:
            continue
        combo = tuple((tuple(x) for x in json.loads(row.source_combo_N3_json)))
        target_set = tuple(json.loads(row.target_set_json))
        rows = []
        for r in ev.fewshot(combo, target_set, FEWSHOTS, MG_FEWSHOT_STOP):
            rows.append({
                'task': 'MG',
                'base_case_id': cid,
                'source_combo': combo_string(combo),
                'target_set': set_string(target_set),
                'alpha': DEFAULT_ALPHA,
                **r,
            })
        append_rows(worker, rows)
        print(f'[MG few-shot shard {shard_id}] case {cid} complete')
    return worker


def merge_fewshot(
    out_dir: Union[str, Path],
    n_shards: int=1,
) -> Path:
    out = Path(out_dir)
    files = [
        out
        / 'fewshot_workers'
        / f'MG_fewshot_shard{i:02d}_of_{n_shards:02d}.csv'
        for i in range(n_shards)
    ]
    if any((not p.exists() for p in files)):
        raise FileNotFoundError('Some MG few-shot shards are missing')
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True).drop_duplicates(
        ['base_case_id', 'method', 'fewshot'],
        keep='last',
    )
    path = out / 'MG_fewshot_raw.csv'
    df.to_csv(path, index=False)
    if len(df) != 141 * 11:
        raise ValueError(f'MG few-shot rows={len(df)}, expected {141 * 11}')
    delta = prepare_fewshot_delta(df, 'MG')
    delta.to_csv(out / 'MG_fewshot_delta_long.csv', index=False)
    return path


def demo(
    out_dir: Union[str, Path],
    data_root: Union[str, Path],
) -> Path:
    combo = ((8, 0, 0), (0, 8, 8), (5, 8, 0))
    target_set = (7, 8, 0)
    bank, target = load_state_bank(data_root)
    ev = MGEvaluator(bank, target, alpha=DEFAULT_ALPHA)
    demo_rows = [
        {
            'task': 'MG',
            'base_case_id': 'demo',
            'source_combo': combo_string(combo),
            'target_set': set_string(target_set),
            **result,
        }
        for result in ev.fewshot(
            combo,
            target_set,
            FEWSHOTS,
            MG_FEWSHOT_STOP,
        )
    ]
    df = pd.DataFrame(demo_rows)
    path = Path(out_dir) / 'MG_fewshot_demo.csv'
    df.to_csv(path, index=False)
    return path


# Plot-only entry point
def plot(
    out_dir: Union[str, Path],
    fig_dir: Union[str, Path],
) -> None:
    out = Path(out_dir)
    fig = ensure_dir(fig_dir)
    plotdata = out / 'plotdata'
    if not (plotdata / 'MG_broad_rule_selected_all_rows.csv').exists():
        process(out)
    plot_baseline_advantage(plotdata / 'MG_broad_rule_selected_all_rows.csv', 'MG', fig)
    plot_ts_vs_n(plotdata / 'MG_broad_rule_selected_all_rows.csv', 'MG', fig)
    plot_opportunity_gain(
        plotdata / 'MG_Fig5e_opportunity_gain_N3_broad.csv',
        'MG',
        fig,
    )
    plot_inout(
        plotdata / 'MG_broad_rule_perN_inout_TS_NRMSE.csv',
        plotdata / 'MG_broad_rule_perN_inout_MWU_summary.csv',
        'MG',
        fig,
    )
    if (out / 'MG_fewshot_delta_long.csv').exists():
        plot_fewshot_statistics(out / 'MG_fewshot_delta_long.csv', 'MG', fig)
    if (out / 'MG_fewshot_demo.csv').exists():
        plot_fewshot_demo(out / 'MG_fewshot_demo.csv', 'MG', fig)


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(sum((1 for _ in open(path, 'rb'))) - 1)
    except Exception:
        return 0


def auto_run(
    out_dir: Union[str, Path],
    fig_dir: Union[str, Path],
    data_root: Union[str, Path],
    seed: int=PLAN_SEED,
    n_cases: int=N_CASES,
    n_shards: int=1,
    reference_plan: Optional[Union[str, Path]]=None,
    recompute: bool=False,
    rebuild_state_bank: bool=False,
) -> None:
    """One-click PyCharm workflow. Existing complete data are reused and replotted."""
    out = Path(out_dir)
    plan_file = out / 'plan' / 'MG_N2345_combo_expand_plan_wide.csv'
    if recompute or not plan_file.exists():
        generate_plan(out, seed, n_cases, reference_plan)
    merged = out / 'MG_N2345_combo_expand_results_long.csv'
    baseline_complete = _csv_row_count(merged) == int(n_cases) * 15
    if recompute or not baseline_complete:
        compute(out, data_root, 0, n_shards, not recompute, rebuild_state_bank)
        merge(out, n_shards)
    else:
        print('[MG auto] Complete baseline data found; skip model computation.')
    process(out)
    fewshot_file = out / 'MG_fewshot_raw.csv'
    fewshot_complete = _csv_row_count(fewshot_file) == 141 * 11
    if recompute or not fewshot_complete:
        fewshot(out, data_root, 0, n_shards, not recompute)
        merge_fewshot(out, n_shards)
    else:
        print('[MG auto] Complete few-shot data found; skip few-shot computation.')
    if recompute or not (out / 'MG_fewshot_demo.csv').exists():
        demo(out, data_root)
    plot(out, fig_dir)


def main() -> None:
    p = argparse.ArgumentParser(
        description='MG TiOx 1000-trial pipeline; directly runnable in PyCharm.',
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
    p.add_argument('--data-root', default=PYCHARM_DATA_ROOT)
    p.add_argument('--seed', type=int, default=PLAN_SEED)
    p.add_argument('--n-cases', type=int, default=N_CASES)
    p.add_argument('--shard-id', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=PYCHARM_N_SHARDS)
    p.add_argument('--reference-plan', default=None)
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
            a.data_root,
            a.seed,
            a.n_cases,
            a.n_shards,
            a.reference_plan,
            a.recompute,
            a.rebuild_state_bank,
        )
        return
    if a.mode in ('plan', 'all'):
        generate_plan(out, a.seed, a.n_cases, a.reference_plan)
    if a.mode in ('compute', 'all'):
        compute(
            out,
            a.data_root,
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
        fewshot(out, a.data_root, a.shard_id, a.n_shards, not a.recompute)
    if a.mode in ('merge-fewshot', 'all'):
        merge_fewshot(out, a.n_shards)
    if a.mode in ('demo', 'all'):
        demo(out, a.data_root)
    if a.mode in ('plot', 'all'):
        plot(out, a.figure_dir)
if __name__ == '__main__':
    main()
