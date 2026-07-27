"""
Arrhythmia reservoir-computing simulations.

The file keeps four functions needed for the manuscript workflow:
    1. one arrhythmia task demo with temporal output plots;
    2. reusable three-channel TiOx raw-state cache;
    3. four-baseline comparison with repeated noise realizations;
    4. redraw of the baseline figure from saved plot data.
"""

import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.linear_model import Ridge

from sim_RC_library import *


ARR_3CH_K3_VALUES = np.array([1.08, 1.06, 1.085, 1.085, 1.10, 1.20, 1.12, 1.16, 0.96]) * 1e-5
ARR_3CH_SOURCE_SETS = ((8, 8, 8), (0, 0, 0), (5, 5, 5))
ARR_3CH_TARGET_SET = (7, 7, 7)
ARR_3CH_NUM_NODE = 160
ARR_3CH_CACHE_ROOT = './Data/Arrhythmia/Arrhythmia_3ch_raw_state_bank_npz_160nodes'
ARR_3CH_SAVE_PREFIX = 'Arrhythmia_3ch_TiOx_160nodes_baselines'
ARR_3CH_RIDGE_CV_ALPHAS = (
    0.0, 1e-12, 3e-12, 1e-11, 3e-11, 1e-10, 3e-10,
    1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6,
    3e-6, 1e-5
)


# ==================================================================================================
# Single-task demo
# ==================================================================================================


def ECG_SRC_sim(
        num_node=ARR_3CH_NUM_NODE,
        mask_abs=0.1,
        direct_transfer=False,
        noise_level=1e-6,
        no_pic=False,
        ridge_alpha=3e-9,
        mode='high',
        fig_suffix='',
        C2C_tr=0.01e-5,
        C2C_ts=0.01e-5,
        Ts_k3=1.16e-5,
        mask_seed=1234,
        random_seed=20260315,
        cache_root=ARR_3CH_CACHE_ROOT,
        reuse_cache=True,
        verbose=True
):
    """Run the three-channel arrhythmia demo and draw its train/test outputs.

    ``num_node`` is the number of virtual nodes in each channel.  With the
    default value, the readout receives 3 x 160 = 480 reservoir states.

    The source/target definitions follow the requested manuscript demo:
        direct_transfer=False: train on (0, 0, 0), test on (7, 7, 7);
        direct_transfer=True: train on temporal partitions from
                              (8, 8, 8), (0, 0, 0), (5, 5, 5),
                              and test on (7, 7, 7).

    ``mode`` and ``Ts_k3`` are retained for compatibility with older calls;
    the three-channel device sets now determine the k3 values explicitly.
    """
    if int(num_node) != ARR_3CH_NUM_NODE:
        raise ValueError(
            'The three-channel demo uses 160 nodes per channel. '
            'Please call ECG_SRC_sim(num_node=160, ...).'
        )

    source_single = (5, 5, 5)
    source_sets = ARR_3CH_SOURCE_SETS
    target_set = ARR_3CH_TARGET_SET
    record_len = 50

    cache = prepare_arrhythmia_3ch_state_cache(
        num_node_per_channel=num_node,
        mask_abs=mask_abs,
        device_k3_values=ARR_3CH_K3_VALUES,
        C2C_tr=C2C_tr,
        C2C_ts=C2C_ts,
        mask_seed=mask_seed,
        cache_root=cache_root,
        reuse_cache=reuse_cache,
        verbose=verbose
    )

    Target_tr = np.asarray(cache['Target_tr'], dtype=np.float32)
    target_ts = np.asarray(cache['target_ts'], dtype=np.float32)
    rng = np.random.RandomState(random_seed)

    # Testing always uses the three-channel target set (7, 7, 7).
    State_ts = _arr_add_noise(
        _arr_assemble_3ch_state(cache, target_set, phase='ts'),
        noise_level=noise_level,
        rng=rng
    )

    if not direct_transfer:
        # One three-channel source device set, trained on the complete source sequence.
        State_tr = _arr_add_noise(
            _arr_assemble_3ch_state(cache, source_single, phase='tr'),
            noise_level=noise_level,
            rng=rng
        )
        Demo_target_tr = Target_tr
        source_description = str(source_single)
        method_name = 'classical'
    else:
        # Temporal training: each source device set receives one contiguous third of the sequence.
        block_len = len(Target_tr) // len(source_sets)
        if block_len * len(source_sets) != len(Target_tr):
            raise ValueError('Training length must be divisible by three source sets.')
        if block_len % record_len != 0:
            raise ValueError('Each temporal training block must contain complete ECG records.')

        state_parts = []
        target_parts = []
        for i, source_set in enumerate(source_sets):
            row0 = i * block_len
            row1 = (i + 1) * block_len
            state_source = _arr_add_noise(
                _arr_assemble_3ch_state(cache, source_set, phase='tr'),
                noise_level=noise_level,
                rng=rng
            )
            state_parts.append(state_source[row0:row1])
            target_parts.append(Target_tr[row0:row1])

        State_tr = np.vstack(state_parts)
        Demo_target_tr = np.vstack(target_parts)
        source_description = str(source_sets)
        method_name = 'TS'

    # One linear readout is trained on the selected three-channel source states.
    lin = Ridge(alpha=float(ridge_alpha))
    lin.fit(State_tr, Demo_target_tr)
    Output_tr = lin.predict(State_tr)
    Output_ts = lin.predict(State_ts)

    TH_choice, THS_choice, train_metrics = _arr_select_thresholds(
        Output_tr,
        Demo_target_tr,
        record_len=record_len
    )
    test_metrics = _arr_evaluate(
        Output_ts,
        target_ts,
        TH_choice,
        THS_choice,
        record_len=record_len
    )

    NRMSE_tr = nrmse(Demo_target_tr, Output_tr)
    NRMSE_ts = nrmse(target_ts, Output_ts)

    print('Three-channel state dimension is {}'.format(State_tr.shape[1]))
    print('Training source set(s): {}'.format(source_description))
    print('Testing target set: {}'.format(target_set))
    print('Training Acc is {}'.format(train_metrics['accuracy']))
    print('Test Acc is {}'.format(test_metrics['accuracy']))
    print('TH is {}, THS is {}'.format(TH_choice, THS_choice))
    print('NRMSE tr is {}'.format(NRMSE_tr))
    print('NRMSE ts is {}'.format(NRMSE_ts))

    if no_pic:
        return test_metrics['accuracy'], NRMSE_tr, NRMSE_ts

    os.makedirs('./Data/Arrhythmia', exist_ok=True)
    os.makedirs('./Figure/Arrhythmia', exist_ok=True)

    signal_path = './Data/Arrhythmia/storage_ECG_signal_3ch_{}{}.h5'.format(method_name, fig_suffix)
    with h5py.File(signal_path, 'w') as file:
        file.create_dataset('Target_tr', data=Demo_target_tr)
        file.create_dataset('Output_tr', data=Output_tr)
        file.create_dataset('target_ts', data=target_ts)
        file.create_dataset('Output_ts', data=Output_ts)
        file.create_dataset('TH', data=TH_choice)
        file.create_dataset('THS', data=THS_choice)
        file.create_dataset('num_node_per_channel', data=int(num_node))
        file.create_dataset('total_nodes', data=int(3 * num_node))

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 0.6), sharey='row', sharex='col')

    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['svg.fonttype'] = 'none'

    color_high = np.array([103, 149, 216]) / 255
    color_mid = np.array([110, 167, 151]) / 255
    color_low = np.array([117, 185, 86]) / 255
    color_test = np.array([107, 158, 184]) / 255
    color_target = np.array([200, 200, 200]) / 255
    color_threshold = np.array([218, 69, 131]) / 255
    color_partition_target = np.array([80, 80, 80]) / 255

    # Training signal
    if not direct_transfer:
        plot_len = min(900, len(Demo_target_tr))
        x_train = np.arange(plot_len)
        ax1.plot(x_train, Demo_target_tr[:plot_len, 0], color=color_target)
        ax1.plot(x_train, Output_tr[:plot_len, 0], color=color_high)
    else:
        colors = [color_low, color_mid, color_high]
        block_len = len(Demo_target_tr) // len(source_sets)
        for i, color in enumerate(colors):
            segment_len = min(300, block_len)
            x = np.arange(i * 300, i * 300 + segment_len)
            row0 = i * block_len
            ax1.plot(x, Demo_target_tr[row0:row0 + segment_len, 0], color=color_partition_target)
            ax1.plot(x, Output_tr[row0:row0 + segment_len, 0], color=color)

    ax1.axvline(300, ls='--', color=np.array([180, 180, 180]) / 255)
    ax1.axvline(600, ls='--', color=np.array([180, 180, 180]) / 255)
    ax1.axhline(TH_choice, color=color_threshold, linestyle='--')
    ax1.set_xlim(0, 900)
    ax1.set_xticks([0, 300, 600, 900])
    ax1.set_xlabel('Time step', fontdict={'family': 'arial', 'size': 6}, labelpad=1)
    ax1.set_ylabel('Output value', fontdict={'family': 'arial', 'size': 6})
    ax1.tick_params(axis='both', direction='in', labelsize=6)

    # Testing signal
    test_start = min(1800, max(0, len(target_ts) - 900))
    test_end = min(test_start + 900, len(target_ts))
    x_test = np.arange(900, 900 + test_end - test_start)
    ax2.plot(x_test, target_ts[test_start:test_end, 0], color=color_target)
    ax2.plot(x_test, Output_ts[test_start:test_end, 0], color=color_test)
    ax2.axhline(TH_choice, color=color_threshold, linestyle='--')
    ax2.set_xlim(900, 1800)
    ax2.set_xticks([1200, 1500, 1800])
    ax2.tick_params(axis='both', direction='in', labelsize=6)
    figure.subplots_adjust(wspace=0, hspace=0.1)

    figure_path = './Figure/Arrhythmia/Sim_ECG_3ch_{}{}.svg'.format(method_name, fig_suffix)
    figure.savefig(figure_path, dpi=300, format='svg', transparent=True, bbox_inches='tight')
    plt.show()

    return test_metrics['accuracy'], NRMSE_tr, NRMSE_ts


# ==================================================================================================
# Three-channel raw-state cache
# ==================================================================================================


def _arr_k3_key(k3):
    return ('{:.12e}'.format(float(k3))
            .replace('+', 'p')
            .replace('-', 'm')
            .replace('.', 'p'))


def _arr_unique_k3_values(k3_values):
    unique = []
    for k3 in np.asarray(k3_values, dtype=float):
        if not any(np.isclose(k3, x, rtol=0, atol=1e-18) for x in unique):
            unique.append(float(k3))
    return np.asarray(unique, dtype=float)


def _arr_load_signal_target(max_records=1000):
    data = io.loadmat('./Data/Arrhythmia/ECGdataset.mat')['dataset'][:max_records, :, :]
    inputs = data[:, :, 0]
    scale = np.max(np.abs(inputs), axis=1, keepdims=True)
    scale[scale == 0] = 1
    inputs = inputs / scale
    labels = data[:, :, 1:]
    signal = inputs.reshape(-1, 1)
    target = labels.reshape(-1, 1)
    return signal, target


def _arr_make_channel_masks(num_node_per_channel=ARR_3CH_NUM_NODE, mask_abs=0.1, mask_seed=1234):
    rng_state = np.random.get_state()
    masks = []
    try:
        for ch in range(3):
            np.random.seed(int(mask_seed) + 1009 * ch)
            masks.append(create_mask(num_node_per_channel, abs_value=mask_abs))
    finally:
        np.random.set_state(rng_state)
    return masks


def _arr_state_file(cache_root, channel, k3):
    return os.path.join(
        cache_root,
        'mask_{}'.format(channel + 1),
        '{}.npz'.format(_arr_k3_key(k3))
    )


def _arr_state_file_matches(
        path, channel, k3, num_node_per_channel,
        mask_abs, split, C2C_tr, C2C_ts, mask_seed, max_records
):
    if not os.path.exists(path):
        return False

    try:
        with np.load(path, allow_pickle=False) as data:
            checks = [
                int(data['channel_index']) == int(channel),
                abs(float(data['k3']) - float(k3)) <= 1e-18,
                int(data['num_node_per_channel']) == int(num_node_per_channel),
                abs(float(data['mask_abs']) - float(mask_abs)) <= 1e-15,
                abs(float(data['split']) - float(split)) <= 1e-15,
                abs(float(data['C2C_tr']) - float(C2C_tr)) <= 1e-18,
                abs(float(data['C2C_ts']) - float(C2C_ts)) <= 1e-18,
                int(data['mask_seed']) == int(mask_seed),
                int(data['max_records']) == int(max_records),
                data['state_tr'].shape[1] == int(num_node_per_channel),
                data['state_ts'].shape[1] == int(num_node_per_channel)
            ]
        return all(checks)
    except (KeyError, OSError, ValueError):
        return False


def prepare_arrhythmia_3ch_state_cache(
        num_node_per_channel=ARR_3CH_NUM_NODE,
        mask_abs=0.1,
        split=0.6,
        device_k3_values=ARR_3CH_K3_VALUES,
        C2C_tr=0.01e-5,
        C2C_ts=0.01e-5,
        max_records=1000,
        mask_seed=1234,
        cache_root=ARR_3CH_CACHE_ROOT,
        reuse_cache=True,
        verbose=True
):
    os.makedirs(cache_root, exist_ok=True)

    device_k3_values = np.asarray(device_k3_values, dtype=float)
    unique_k3_values = _arr_unique_k3_values(device_k3_values)
    signal, target = _arr_load_signal_target(max_records=max_records)
    masks = _arr_make_channel_masks(
        num_node_per_channel=num_node_per_channel,
        mask_abs=mask_abs,
        mask_seed=mask_seed
    )

    Target_tr_ref = None
    target_ts_ref = None
    state_files = {}
    SRC = TiOx_SRC()

    for ch, mask in enumerate(masks):
        channel_dir = os.path.join(cache_root, 'mask_{}'.format(ch + 1))
        os.makedirs(channel_dir, exist_ok=True)
        np.save(os.path.join(channel_dir, 'mask.npy'), mask)

        Input_tr, input_ts, Target_tr, target_ts = signal_process(signal, target, mask, split=split)

        if Target_tr_ref is None:
            Target_tr_ref = Target_tr.astype(np.float32, copy=False)
            target_ts_ref = target_ts.astype(np.float32, copy=False)
        elif Target_tr.shape != Target_tr_ref.shape or target_ts.shape != target_ts_ref.shape:
            raise RuntimeError('Unexpected target shape mismatch among channel masks.')

        for k3 in unique_k3_values:
            path = _arr_state_file(cache_root, ch, k3)
            state_files[(ch, _arr_k3_key(k3))] = path

            if reuse_cache and _arr_state_file_matches(
                    path, ch, k3, num_node_per_channel,
                    mask_abs, split, C2C_tr, C2C_ts, mask_seed, max_records
            ):
                if verbose:
                    print('Reusing {}'.format(path))
                continue

            if verbose:
                print('Generating mask {}, k3 = {:.6g}'.format(ch + 1, k3))

            i_tr, g_tr, g0_tr = SRC.iterate_SRC(
                Input_tr, 20e-6,
                k3=float(k3),
                virtual_nodes=num_node_per_channel,
                C2C_strength=C2C_tr,
                clear=True
            )
            state_tr = i_tr.reshape(len(Target_tr_ref), num_node_per_channel).astype(np.float32, copy=False)

            i_ts, g_ts, g0_ts = SRC.iterate_SRC(
                input_ts, 20e-6,
                k3=float(k3),
                virtual_nodes=num_node_per_channel,
                C2C_strength=C2C_ts,
                clear=True
            )
            state_ts = i_ts.reshape(len(target_ts_ref), num_node_per_channel).astype(np.float32, copy=False)

            np.savez_compressed(
                path,
                state_tr=state_tr,
                state_ts=state_ts,
                Target_tr=Target_tr_ref,
                target_ts=target_ts_ref,
                channel_index=np.array(ch, dtype=np.int64),
                k3=np.array(k3, dtype=np.float64),
                num_node_per_channel=np.array(num_node_per_channel, dtype=np.int64),
                mask_abs=np.array(mask_abs, dtype=np.float64),
                split=np.array(split, dtype=np.float64),
                C2C_tr=np.array(C2C_tr, dtype=np.float64),
                C2C_ts=np.array(C2C_ts, dtype=np.float64),
                mask_seed=np.array(mask_seed, dtype=np.int64),
                max_records=np.array(max_records, dtype=np.int64)
            )

            del i_tr, i_ts, state_tr, state_ts

    metadata = {
        'cache_format': 'per_mask_per_k3_npz',
        'num_channels': 3,
        'num_node_per_channel': int(num_node_per_channel),
        'mask_abs': float(mask_abs),
        'split': float(split),
        'C2C_tr': float(C2C_tr),
        'C2C_ts': float(C2C_ts),
        'mask_seed': int(mask_seed),
        'max_records': int(max_records),
        'device_k3_values': [float(x) for x in device_k3_values],
        'unique_k3_values': [float(x) for x in unique_k3_values]
    }
    with open(os.path.join(cache_root, 'state_bank_metadata.json'), 'w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2)

    return {
        'Target_tr': Target_tr_ref,
        'target_ts': target_ts_ref,
        'device_k3_values': device_k3_values,
        'unique_k3_values': unique_k3_values,
        'num_node_per_channel': int(num_node_per_channel),
        'state_files': state_files,
        'cache_root': cache_root
    }


def _arr_assemble_3ch_state(cache, device_set, phase='tr'):
    if len(device_set) != 3:
        raise ValueError('device_set must contain three device indices.')
    if phase not in ('tr', 'ts'):
        raise ValueError("phase must be 'tr' or 'ts'.")

    dataset_name = 'state_tr' if phase == 'tr' else 'state_ts'
    states = []

    for ch, device_index in enumerate(device_set):
        k3 = cache['device_k3_values'][int(device_index)]
        path = cache['state_files'][(ch, _arr_k3_key(k3))]
        with np.load(path, allow_pickle=False) as data:
            states.append(data[dataset_name].astype(np.float32, copy=False))

    return np.hstack(states)


# ==================================================================================================
# Metrics and baseline models
# ==================================================================================================


def _arr_add_noise(state, noise_level, rng):
    state = np.asarray(state, dtype=np.float32)
    if noise_level is None or noise_level <= 0:
        return state.copy()
    noise = rng.normal(0, noise_level, size=state.shape).astype(np.float32)
    return state + noise


def _arr_record_labels(target, record_len=50):
    target = np.asarray(target).reshape(-1)
    n = len(target) // record_len * record_len
    return np.max(target[:n].reshape(-1, record_len), axis=1).astype(int)


def _arr_record_predictions(output, TH, THS, record_len=50):
    output = np.asarray(output).reshape(-1)
    n = len(output) // record_len * record_len
    count = np.sum(output[:n].reshape(-1, record_len) >= TH, axis=1)
    return (count >= THS).astype(int)


def _arr_binary_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    }


def _arr_evaluate(output, target, TH, THS, record_len=50):
    y_true = _arr_record_labels(target, record_len=record_len)
    y_pred = _arr_record_predictions(output, TH, THS, record_len=record_len)
    metrics = _arr_binary_metrics(y_true, y_pred)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return metrics


def _arr_select_thresholds(output, target, record_len=50, TH_box=None, THS_box=None):
    if TH_box is None:
        TH_box = np.arange(0.21, 0.8, 0.01)
    if THS_box is None:
        THS_box = np.arange(1, 6)

    output = np.asarray(output).reshape(-1)
    n = len(output) // record_len * record_len
    output_record = output[:n].reshape(-1, record_len)
    y_true = _arr_record_labels(target, record_len=record_len)
    best = None

    for TH in TH_box:
        count = np.sum(output_record >= TH, axis=1)
        for THS in THS_box:
            metrics = _arr_binary_metrics(y_true, (count >= THS).astype(int))
            center = -abs(float(TH) - 0.5) - 0.01 * abs(float(THS) - 3.0)
            score = (metrics['accuracy'], metrics['macro_f1'], center)
            if best is None or score > best['score']:
                best = {
                    'TH': float(TH),
                    'THS': int(THS),
                    'metrics': metrics,
                    'score': score
                }

    return best['TH'], best['THS'], best['metrics']


def _arr_record_indices(record_indices, record_len=50):
    record_indices = np.asarray(record_indices, dtype=int)
    return (record_indices[:, None] * record_len + np.arange(record_len)).reshape(-1)


def _arr_select_ridge_alpha_cv(
        X, y,
        alphas=ARR_3CH_RIDGE_CV_ALPHAS,
        record_len=50,
        n_folds=5
):
    n_records = len(y) // record_len
    folds = [x for x in np.array_split(np.arange(n_records), n_folds) if len(x) > 0]
    all_records = np.arange(n_records)
    rows = []

    for alpha in alphas:
        fold_accuracy = []
        fold_f1 = []

        for val_records in folds:
            train_records = np.setdiff1d(all_records, val_records)
            train_index = _arr_record_indices(train_records, record_len=record_len)
            val_index = _arr_record_indices(val_records, record_len=record_len)

            lin = Ridge(alpha=float(alpha))
            lin.fit(X[train_index], y[train_index])

            output_train = lin.predict(X[train_index])
            TH, THS, train_metrics = _arr_select_thresholds(
                output_train,
                y[train_index],
                record_len=record_len
            )

            output_val = lin.predict(X[val_index])
            val_metrics = _arr_evaluate(
                output_val,
                y[val_index],
                TH,
                THS,
                record_len=record_len
            )
            fold_accuracy.append(val_metrics['accuracy'])
            fold_f1.append(val_metrics['macro_f1'])

        rows.append({
            'alpha': float(alpha),
            'cv_accuracy_mean': float(np.mean(fold_accuracy)),
            'cv_accuracy_std': float(np.std(fold_accuracy)),
            'cv_macro_f1_mean': float(np.mean(fold_f1)),
            'cv_macro_f1_std': float(np.std(fold_f1))
        })

    cv_df = pd.DataFrame(rows).sort_values(
        ['cv_accuracy_mean', 'cv_macro_f1_mean', 'alpha'],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return float(cv_df.loc[0, 'alpha']), cv_df


def run_arrhythmia_3ch_baselines_once(
        cache,
        source_sets=ARR_3CH_SOURCE_SETS,
        target_set=ARR_3CH_TARGET_SET,
        single_source_set=None,
        noise_level=1e-6,
        ridge_alpha=3e-9,
        ridge_cv_alphas=ARR_3CH_RIDGE_CV_ALPHAS,
        ridge_cv_folds=5,
        record_len=50,
        random_seed=None,
        verbose=False
):
    source_sets = tuple(tuple(int(x) for x in s) for s in source_sets)
    target_set = tuple(int(x) for x in target_set)
    if single_source_set is None:
        single_source_set = source_sets[-1]
    single_source_set = tuple(int(x) for x in single_source_set)

    if len(source_sets) != 3:
        raise ValueError('Three source sets are required.')

    rng = np.random.RandomState(random_seed)
    Target_tr = np.asarray(cache['Target_tr'], dtype=np.float32)
    target_ts = np.asarray(cache['target_ts'], dtype=np.float32)
    block_len = len(Target_tr) // 3

    if block_len * 3 != len(Target_tr) or block_len % record_len != 0:
        raise ValueError('Training samples must form three complete ECG-record blocks.')

    state_cache = {}

    def get_state(device_set, phase):
        key = (tuple(device_set), phase)
        if key not in state_cache:
            raw_state = _arr_assemble_3ch_state(cache, device_set, phase=phase)
            state_cache[key] = _arr_add_noise(raw_state, noise_level, rng)
        return state_cache[key]

    X_target_ts = get_state(target_set, 'ts')
    results = []

    def finish(method, label, output_train, output_test, y_train, alpha, extra=None):
        TH, THS, selected_metrics = _arr_select_thresholds(
            output_train,
            y_train,
            record_len=record_len
        )
        train_metrics = _arr_evaluate(output_train, y_train, TH, THS, record_len=record_len)
        test_metrics = _arr_evaluate(output_test, target_ts, TH, THS, record_len=record_len)

        row = {
            'method': method,
            'method_label': label,
            'alpha': float(alpha),
            'TH': float(TH),
            'THS': int(THS),
            'train_accuracy': train_metrics['accuracy'],
            'train_macro_precision': train_metrics['macro_precision'],
            'train_macro_recall': train_metrics['macro_recall'],
            'train_macro_f1': train_metrics['macro_f1'],
            'test_accuracy': test_metrics['accuracy'],
            'test_macro_precision': test_metrics['macro_precision'],
            'test_macro_recall': test_metrics['macro_recall'],
            'test_macro_f1': test_metrics['macro_f1'],
            'train_nrmse': float(nrmse(y_train, output_train)),
            'test_nrmse': float(nrmse(target_ts, output_test)),
            'confusion_matrix': test_metrics['confusion_matrix'].tolist()
        }
        if extra is not None:
            row.update(extra)
        results.append(row)

    # 1. Temporal-switch training
    X_ts = []
    y_ts = []
    for i, source_set in enumerate(source_sets):
        X_source = get_state(source_set, 'tr')
        X_ts.append(X_source[i * block_len:(i + 1) * block_len])
        y_ts.append(Target_tr[i * block_len:(i + 1) * block_len])

    X_ts = np.vstack(X_ts)
    y_ts = np.vstack(y_ts)
    lin = Ridge(alpha=float(ridge_alpha))
    lin.fit(X_ts, y_ts)
    finish(
        'TS_partition_N3', 'TS',
        lin.predict(X_ts),
        lin.predict(X_target_ts),
        y_ts,
        ridge_alpha,
        {'source_sets': json.dumps(source_sets), 'target_set': json.dumps(target_set)}
    )

    # 2. Output ensemble
    output_train = []
    output_test = []
    for source_set in source_sets:
        X_source = get_state(source_set, 'tr')
        lin = Ridge(alpha=float(ridge_alpha))
        lin.fit(X_source, Target_tr)
        output_train.append(lin.predict(X_source))
        output_test.append(lin.predict(X_target_ts))

    finish(
        'output_ensemble_N3', 'Output ensemble',
        np.mean(np.stack(output_train), axis=0),
        np.mean(np.stack(output_test), axis=0),
        Target_tr,
        ridge_alpha,
        {'source_sets': json.dumps(source_sets), 'target_set': json.dumps(target_set)}
    )

    # 3. Source-state average
    X_average = np.mean(np.stack([get_state(x, 'tr') for x in source_sets]), axis=0)
    lin = Ridge(alpha=float(ridge_alpha))
    lin.fit(X_average, Target_tr)
    finish(
        'source_state_average_N3', 'State average',
        lin.predict(X_average),
        lin.predict(X_target_ts),
        Target_tr,
        ridge_alpha,
        {'source_sets': json.dumps(source_sets), 'target_set': json.dumps(target_set)}
    )

    # 4. Single-source ridge-CV
    X_single = get_state(single_source_set, 'tr')
    alpha_cv, cv_df = _arr_select_ridge_alpha_cv(
        X_single,
        Target_tr,
        alphas=ridge_cv_alphas,
        record_len=record_len,
        n_folds=ridge_cv_folds
    )
    lin = Ridge(alpha=alpha_cv)
    lin.fit(X_single, Target_tr)
    finish(
        'single_source_ridge_cv', 'Ridge-CV',
        lin.predict(X_single),
        lin.predict(X_target_ts),
        Target_tr,
        alpha_cv,
        {
            'single_source_set': json.dumps(single_source_set),
            'target_set': json.dumps(target_set),
            'ridge_cv_best_accuracy': float(cv_df.loc[0, 'cv_accuracy_mean']),
            'ridge_cv_best_macro_f1': float(cv_df.loc[0, 'cv_macro_f1_mean'])
        }
    )

    if verbose:
        for row in results:
            print('{:<18s} acc={:.4f}, F1={:.4f}, alpha={:.3g}'.format(
                row['method_label'], row['test_accuracy'], row['test_macro_f1'], row['alpha']
            ))

    return results, cv_df


# ==================================================================================================
# Baseline comparison and cached plotting
# ==================================================================================================


def _arr_prepare_plot_data(df):
    method_order = ['TS', 'Output ensemble', 'State average', 'Ridge-CV']
    method_short = ['TS', 'ensemble', 'state-avg.', 'ridge CV']
    metric_order = [
        ('test_accuracy', 'Accuracy'),
        ('test_macro_recall', 'Recall'),
        ('test_macro_precision', 'Precision'),
        ('test_macro_f1', 'F1 score')
    ]

    rows = []
    for method_index, method in enumerate(method_order):
        method_df = df[df['method_label'] == method]
        for metric_index, (metric, metric_label) in enumerate(metric_order):
            for _, row in method_df.iterrows():
                rows.append({
                    'method_label': method,
                    'method_short': method_short[method_index],
                    'method_order': method_index,
                    'metric': metric,
                    'metric_label': metric_label,
                    'metric_order': metric_index,
                    'value': float(row[metric]),
                    'value_percent': 100 * float(row[metric]),
                    'round': int(row['round'])
                })

    return pd.DataFrame(rows)


def _arr_plot_baseline_metrics(plot_df, out_path):
    method_order = ['TS', 'Output ensemble', 'State average', 'Ridge-CV']
    method_short = ['TS', 'ensemble', 'state-avg.', 'ridge CV']
    metric_order = [
        ('test_accuracy', 'Accuracy', np.array([119, 176, 120]) / 255),
        ('test_macro_recall', 'Recall', np.array([116, 169, 141]) / 255),
        ('test_macro_precision', 'Precision', np.array([113, 156, 189]) / 255),
        ('test_macro_f1', 'F1 score', np.array([112, 150, 211]) / 255)
    ]

    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42

    figure, ax = plt.subplots(figsize=(4.8, 2))
    centers = np.arange(len(method_order), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(metric_order))
    handles = []

    for metric_index, (metric, label, color) in enumerate(metric_order):
        values = []
        positions = []
        for method_index, method in enumerate(method_order):
            values.append(plot_df.loc[
                (plot_df['method_label'] == method) & (plot_df['metric'] == metric),
                'value_percent'
            ].to_numpy(dtype=float))
            positions.append(centers[method_index] + offsets[metric_index])

        boxplot = ax.boxplot(
            values,
            positions=positions,
            widths=0.12,
            patch_artist=True,
            showfliers=False,
            boxprops={'linewidth': 0.5, 'edgecolor': 'black'},
            whiskerprops={'linewidth': 0.5, 'color': 'black'},
            capprops={'linewidth': 0.5, 'color': 'black'},
            medianprops={'linewidth': 0.7, 'color': 'black'}
        )

        facecolor = np.clip(color + (1 - color) * 0.08, 0, 1)
        for box in boxplot['boxes']:
            box.set_facecolor(facecolor)
            box.set_alpha(0.92)

        handles.append(Patch(facecolor=facecolor, edgecolor='black', linewidth=0.5, label=label))

    ax.set_xticks(centers)
    ax.set_xticklabels(method_short)
    ax.set_ylabel('Score (%)')
    ax.set_ylim(50, 100)
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.tick_params(axis='both', direction='in', labelsize=6)
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=6,
        loc='lower right',
        ncol=2,
        handlelength=1.0,
        columnspacing=0.8,
        handletextpad=0.4
    )

    figure.tight_layout(pad=0.4)
    figure.savefig(out_path, dpi=300, format='svg', transparent=True, bbox_inches='tight')
    root = os.path.splitext(out_path)[0]
    figure.savefig(root + '.png', dpi=600, transparent=True, bbox_inches='tight')
    figure.savefig(root + '.pdf', dpi=300, transparent=True, bbox_inches='tight')
    plt.close(figure)


def plot_arrhythmia_3ch_all_metrics_from_saved(
        save_prefix=ARR_3CH_SAVE_PREFIX,
        data_dir='./Data/Arrhythmia',
        figure_dir='./Figure/Arrhythmia'
):
    os.makedirs(figure_dir, exist_ok=True)

    plotdata_path = os.path.join(data_dir, '{}_all_metrics_plotdata.csv'.format(save_prefix))
    long_path = os.path.join(data_dir, '{}_long.csv'.format(save_prefix))
    figure_path = os.path.join(figure_dir, '{}_all_metrics.svg'.format(save_prefix))

    if os.path.exists(plotdata_path):
        plot_df = pd.read_csv(plotdata_path)
    elif os.path.exists(long_path):
        result_df = pd.read_csv(long_path)
        plot_df = _arr_prepare_plot_data(result_df)
        plot_df.to_csv(plotdata_path, index=False)
    else:
        raise FileNotFoundError('No cached plot data or baseline result file was found.')

    _arr_plot_baseline_metrics(plot_df, figure_path)
    print('Saved {}'.format(figure_path))
    return plot_df


def compare_arrhythmia_3ch_baselines(
        rounds=20,
        num_node_per_channel=ARR_3CH_NUM_NODE,
        mask_abs=0.1,
        noise_level=1e-6,
        ridge_alpha=3e-9,
        source_sets=ARR_3CH_SOURCE_SETS,
        target_set=ARR_3CH_TARGET_SET,
        single_source_set=None,
        C2C_tr=0.01e-5,
        C2C_ts=0.01e-5,
        mask_seed=1234,
        base_noise_seed=20260315,
        ridge_cv_alphas=ARR_3CH_RIDGE_CV_ALPHAS,
        ridge_cv_folds=5,
        cache_root=ARR_3CH_CACHE_ROOT,
        reuse_cache=True,
        save_prefix=ARR_3CH_SAVE_PREFIX,
        verbose=True
):
    os.makedirs('./Data/Arrhythmia', exist_ok=True)
    os.makedirs('./Figure/Arrhythmia', exist_ok=True)

    cache = prepare_arrhythmia_3ch_state_cache(
        num_node_per_channel=num_node_per_channel,
        mask_abs=mask_abs,
        device_k3_values=ARR_3CH_K3_VALUES,
        C2C_tr=C2C_tr,
        C2C_ts=C2C_ts,
        mask_seed=mask_seed,
        cache_root=cache_root,
        reuse_cache=reuse_cache,
        verbose=verbose
    )

    if single_source_set is None:
        single_source_set = source_sets[-1]

    result_rows = []
    cv_rows = []

    for round_index in range(rounds):
        if verbose:
            print('\nArrhythmia 3CH round {}/{}'.format(round_index + 1, rounds))

        rows, cv_df = run_arrhythmia_3ch_baselines_once(
            cache,
            source_sets=source_sets,
            target_set=target_set,
            single_source_set=single_source_set,
            noise_level=noise_level,
            ridge_alpha=ridge_alpha,
            ridge_cv_alphas=ridge_cv_alphas,
            ridge_cv_folds=ridge_cv_folds,
            random_seed=base_noise_seed + round_index,
            verbose=verbose
        )

        for row in rows:
            row['round'] = round_index
            row['num_node_per_channel'] = num_node_per_channel
            row['total_nodes'] = 3 * num_node_per_channel
            row['noise_level'] = noise_level
            row['C2C_tr'] = C2C_tr
            row['C2C_ts'] = C2C_ts
            row['mask_seed'] = mask_seed
            result_rows.append(row)

        cv_df = cv_df.copy()
        cv_df['round'] = round_index
        cv_rows.append(cv_df)

    result_df = pd.DataFrame(result_rows)
    cv_all = pd.concat(cv_rows, ignore_index=True)

    long_path = './Data/Arrhythmia/{}_long.csv'.format(save_prefix)
    cv_path = './Data/Arrhythmia/{}_ridge_cv_details.csv'.format(save_prefix)
    summary_path = './Data/Arrhythmia/{}_summary.csv'.format(save_prefix)
    plotdata_path = './Data/Arrhythmia/{}_all_metrics_plotdata.csv'.format(save_prefix)
    figure_path = './Figure/Arrhythmia/{}_all_metrics.svg'.format(save_prefix)

    result_for_csv = result_df.copy()
    result_for_csv['confusion_matrix'] = result_for_csv['confusion_matrix'].apply(json.dumps)
    result_for_csv.to_csv(long_path, index=False)
    cv_all.to_csv(cv_path, index=False)

    summary = result_df.groupby('method_label')[
        ['test_accuracy', 'test_macro_precision', 'test_macro_recall', 'test_macro_f1', 'test_nrmse']
    ].agg(['mean', 'median', 'std'])
    summary.to_csv(summary_path)

    plot_df = _arr_prepare_plot_data(result_df)
    plot_df.to_csv(plotdata_path, index=False)
    _arr_plot_baseline_metrics(plot_df, figure_path)

    if verbose:
        print('\nSaved {}'.format(long_path))
        print('Saved {}'.format(cv_path))
        print('Saved {}'.format(summary_path))
        print('Saved {}'.format(plotdata_path))
        print('Saved {}'.format(figure_path))
        print('\n{}'.format(summary.to_string()))

    return result_df, cv_all, summary


if __name__ == '__main__':

    # Main baseline comparison. Cached raw states are reused automatically.
    compare_arrhythmia_3ch_baselines(rounds=20)

    # Single-task temporal signal demo. Uncomment one line when the demo figure is needed.
    ECG_SRC_sim(num_node=160, direct_transfer=False)
    ECG_SRC_sim(num_node=160, direct_transfer=True)

    # Redraw the baseline figure from saved CSV data without rerunning the reservoir.
    plot_arrhythmia_3ch_all_metrics_from_saved()
