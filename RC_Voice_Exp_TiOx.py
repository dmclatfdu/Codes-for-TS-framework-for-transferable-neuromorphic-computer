#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spoken digit classification reservoir-computing demos with TS framework.

The file contains the classical, temporal-switch,
state-average, output-ensemble and Ridge-CV experiments under ``if __name__ == '__main__'``.
It also contains supporting functions for the 1000 trials processing.

Zefeng Zhang, Research Institute of Intelligent Complex Systems, Fudan University
"""
import csv
import os
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import Ridge
from Voice_Inputs import *
DEVICE = 'TiOx'
DEVICE_SET = ['15d', '16d', '4u', '11u']
PREFIX = ['15d_', '16d_', '4u_', '11u_']
DT_SET = np.asarray([1, 2, 2, 3], dtype=int)
TS_SETS = np.asarray(
    [[1, 2, 2, 3], [2, 1, 1, 1], [3, 1, 2, 0]],
    dtype=int,
)
MAX_VOICE_STEPS = 25
DEFAULT_NUM_NODE = 40
DEFAULT_DOWNSAMPLING_START = 15
RIDGE_CV_ALPHAS = (
    0.0,
    1e-16,
    1e-14,
    1e-12,
    1e-10,
    1e-08,
    1e-06,
    0.0001,
    0.01,
    1.0,
    100.0,
)
RIDGE_CV_FOLDS = 5


def _voice_root(*parts):
    return os.path.join('.', 'Data', 'Voice', DEVICE, *parts)


def _response_file(
    device_index,
    digit,
    subject,
    set_number,
):
    code = DEVICE_SET[int(device_index)]
    name = '{}0{}f{}set{}.csv'.format(
        PREFIX[int(device_index)],
        int(digit),
        int(subject),
        int(set_number),
    )
    if DEVICE == 'NbOx':
        return _voice_root('response', 'Device {}'.format(code), name)
    return _voice_root('response', code, name)


def _state_bank_path(
    num_node=DEFAULT_NUM_NODE,
    down_sampling_start=DEFAULT_DOWNSAMPLING_START,
):
    cache_dir = _voice_root('cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir,
        'response_statebank_nodes{}_start{}_float64_v2.npy'.format(
            int(num_node),
            int(down_sampling_start),
        ),
    )


def _read_response_state(
    file_path,
    num_node,
    down_sampling_start,
):
    """Read 20,000 numeric current samples after the instrument CSV header."""
    row_count = 20000
    with open(file_path, 'rb') as handle:
        lines = handle.read().splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if b'MeasResult2_value' in line),
        None,
    )
    if header_index is None:
        start_index, current_column = (148, 3)
    else:
        header_fields = lines[header_index].replace(b'\x00', b'').split(b',')
        current_column = next(
            (
                index
                for index, field in enumerate(header_fields)
                if b'MeasResult2_value' in field
            ),
        )
        start_index = header_index + 1
    values = []
    for line in lines[start_index:]:
        fields = line.replace(b'\x00', b'').split(b',')
        if len(fields) <= current_column:
            continue
        token = fields[current_column].strip().strip(b'"')
        try:
            values.append(-float(token))
        except ValueError:
            continue
        if len(values) == row_count:
            break
    if len(values) != row_count:
        raise ValueError(
            (
                'Response {} contains {} valid current samples after the '
                'data header; expected {}.'
            ).format(
                file_path,
                len(values),
                row_count,
            ),
        )
    current = np.asarray(values, dtype=np.float64)
    ratio = int(len(current) / MAX_VOICE_STEPS / int(num_node))
    if ratio <= 0:
        raise ValueError('Invalid down-sampling ratio for {}'.format(file_path))
    required = MAX_VOICE_STEPS * int(num_node)
    sampled = current[int(down_sampling_start)::ratio][:required]
    if sampled.size != required:
        raise ValueError(
            'Response {} produced {} samples; expected {}.'.format(
                file_path,
                sampled.size,
                required,
            ),
        )
    return sampled.reshape(MAX_VOICE_STEPS, int(num_node))


def build_response_state_bank(
    num_node=DEFAULT_NUM_NODE,
    down_sampling_start=DEFAULT_DOWNSAMPLING_START,
    force=False,
    verbose=True,
):
    """Build the reusable response cache from the 2,000 unique raw CSV files."""
    cache_path = _state_bank_path(num_node, down_sampling_start)
    expected_shape = (len(DEVICE_SET), 5, 10, 10, MAX_VOICE_STEPS, int(num_node))
    if os.path.exists(cache_path) and (not force):
        bank = np.load(cache_path, mmap_mode='r')
        if bank.shape != expected_shape:
            raise ValueError(
                (
                    'Cache shape {} does not match expected {}. '
                    'Rebuild with force=True.'
                ).format(
                    bank.shape,
                    expected_shape,
                ),
            )
        return bank
    temp_path = cache_path + '.tmp.npy'
    if os.path.exists(temp_path):
        os.remove(temp_path)
    bank = np.lib.format.open_memmap(
        temp_path,
        mode='w+',
        dtype=np.float64,
        shape=expected_shape,
    )
    total = len(DEVICE_SET) * 5 * 10 * 10
    count = 0
    try:
        for device_index in range(len(DEVICE_SET)):
            for subject in range(1, 6):
                for digit in range(10):
                    for set_number in range(10):
                        file_path = _response_file(
                            device_index,
                            digit,
                            subject,
                            set_number,
                        )
                        bank[
                            device_index,
                            subject - 1,
                            digit,
                            set_number,
                        ] = _read_response_state(
                            file_path,
                            num_node=int(num_node),
                            down_sampling_start=int(down_sampling_start),
                        )
                        count += 1
                        if verbose and (count % 100 == 0 or count == total):
                            print('[{} cache] {}/{}'.format(DEVICE, count, total))
        bank.flush()
    except Exception:
        del bank
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    del bank
    os.replace(temp_path, cache_path)
    return np.load(cache_path, mmap_mode='r')


def load_response_state_bank(
    num_node=DEFAULT_NUM_NODE,
    down_sampling_start=DEFAULT_DOWNSAMPLING_START,
    rebuild=False,
    verbose=True,
):
    return build_response_state_bank(
        num_node=num_node,
        down_sampling_start=down_sampling_start,
        force=rebuild,
        verbose=verbose,
    )


def load_or_create_cv_split(random_seed=0, rebuild=False):
    """Create one reproducible 10-fold split and reuse it for ID, DT and TS."""
    input_dir = _voice_root('inputs')
    os.makedirs(input_dir, exist_ok=True)
    path = os.path.join(input_dir, 'cv_split_seed{}.npy'.format(int(random_seed)))
    if os.path.exists(path) and (not rebuild):
        split = np.load(path)
        if split.shape != (50, 10, 3):
            raise ValueError('Unexpected split shape: {}'.format(split.shape))
        return split
    base = np.empty((50, 10, 3), dtype=np.int16)
    row = 0
    for subject in range(1, 6):
        for set_number in range(10):
            for digit in range(10):
                base[row, digit] = (digit, subject, set_number)
            row += 1
    rng = np.random.default_rng(int(random_seed))
    split = np.empty_like(base)
    for digit in range(10):
        split[:, digit] = base[rng.permutation(50), digit]
    np.save(path, split)
    return split

@lru_cache(maxsize=1)


def _load_steps_rec():
    path = _voice_root('inputs', 'steps_rec.csv')
    return np.loadtxt(path, delimiter=',', usecols=0, dtype=np.int32)


def target_signal_gen(
    words,
    sample_set,
    num_classification,
):
    """Build variable lengths, one-hot frame targets and sample metadata."""
    sample_set = np.asarray(sample_set, dtype=np.int16)
    file_info = sample_set.reshape(-1, 3)
    if len(file_info) != int(words):
        raise ValueError(
            'words={} but sample_set contains {} samples.'.format(
                words,
                len(file_info),
            ),
        )
    digit = file_info[:, 0].astype(int)
    subject = file_info[:, 1].astype(int)
    set_number = file_info[:, 2].astype(int)
    steps = _load_steps_rec()
    lengths = steps[set_number + digit * 10 + (subject - 1) * 100].astype(int)
    labels = np.tile(np.arange(int(num_classification)), sample_set.shape[0])
    frame_labels = np.repeat(labels, lengths)
    target = np.zeros((int(np.sum(lengths)), int(num_classification)), dtype=float)
    target[np.arange(len(frame_labels)), frame_labels] = 1.0
    return (lengths.tolist(), target, file_info)


def _offsets(lengths):
    return np.concatenate(([0], np.cumsum(np.asarray(lengths, dtype=np.int64))))


def _assemble_features(
    state_bank,
    file_info,
    lengths,
    pointers,
    num_node,
):
    """Assemble one fold directly from the memory-mapped response state bank."""
    pointers = np.asarray(pointers, dtype=int)
    if pointers.ndim != 2 or pointers.shape[0] != len(file_info):
        raise ValueError('pointers must have shape (n_words, channels).')
    offsets = _offsets(lengths)
    channels = pointers.shape[1]
    X = np.empty(
        (int(offsets[-1]), int(num_node) * channels),
        dtype=np.float64,
    )
    for word_index, (digit, subject, set_number) in enumerate(file_info.astype(int)):
        row0, row1 = (int(offsets[word_index]), int(offsets[word_index + 1]))
        length = row1 - row0
        for channel, device_index in enumerate(pointers[word_index]):
            col0 = channel * int(num_node)
            col1 = col0 + int(num_node)
            X[row0:row1, col0:col1] = state_bank[
                int(device_index),
                int(subject) - 1,
                int(digit),
                int(set_number),
                :length,
                :int(num_node),
            ]
    return X


def conmat_acc(
    words,
    output,
    VL,
    num_classification,
):
    """Return the word-level confusion matrix and number of correct words."""
    offsets = _offsets(VL)
    confusion = np.zeros((num_classification, num_classification), dtype=float)
    correct = 0
    for word_index in range(int(words)):
        word_output = output[offsets[word_index]:offsets[word_index + 1]]
        predicted = int(np.argmax(np.mean(word_output, axis=0)))
        truth = word_index % int(num_classification)
        confusion[truth, predicted] += 1
        correct += int(predicted == truth)
    return (confusion, correct)


def _train_device_sets(
    parallel,
    channels,
    identical,
    direct_transfer,
):
    channels = int(channels) if parallel else 1
    if channels < 1 or channels > len(DEVICE_SET):
        raise ValueError('channels must be between 1 and {}.'.format(len(DEVICE_SET)))
    test_set = np.arange(
        channels,
        dtype=int,
    ).reshape(1, -1) if parallel else np.array([[0]])
    if identical:
        train_sets = test_set.copy()
        mode_name = 'ID'
    elif direct_transfer:
        train_sets = DT_SET[:channels].reshape(1, -1)
        mode_name = 'DT'
    else:
        train_sets = TS_SETS[:, :channels]
        mode_name = 'TS'
    return (train_sets, test_set, channels, mode_name)


def _word_pointers(device_sets, words):
    device_sets = np.asarray(device_sets, dtype=int)
    if len(device_sets) == 1:
        return np.repeat(device_sets, int(words), axis=0)
    if int(words) % len(device_sets):
        raise ValueError('Number of words must be divisible by the number of switches.')
    return np.repeat(device_sets, int(words) // len(device_sets), axis=0)


def Voice_SRC_exp(
    Device=DEVICE,
    num_classification=10,
    num_node=DEFAULT_NUM_NODE,
    direct_transfer=False,
    identical=False,
    down_sampling_start=DEFAULT_DOWNSAMPLING_START,
    OUTPUT=False,
    parallel=False,
    channels=3,
    suffix='',
    state_bank=None,
    split_matrix=None,
    random_seed=0,
    rebuild_cache=False,
    verbose=True,
):
    if Device != DEVICE:
        raise ValueError('This file is configured for {}.'.format(DEVICE))
    if int(num_classification) != 10:
        raise ValueError('The supplied spoken-digit data uses 10 classes.')
    train_sets, test_set_devices, channels, mode_name = _train_device_sets(
        parallel,
        channels,
        identical,
        direct_transfer,
    )
    if verbose:
        print({
            'ID': 'identical',
            'DT': 'direct transfer',
            'TS': 'temporal switch',
        }[mode_name])
    if state_bank is None:
        state_bank = load_response_state_bank(
            num_node=num_node,
            down_sampling_start=down_sampling_start,
            rebuild=rebuild_cache,
            verbose=verbose,
        )
    if split_matrix is None:
        split_matrix = load_or_create_cv_split(random_seed=random_seed)
    train_con_mat, test_con_mat = ([], [])
    train_acc, test_acc = ([], [])
    train_power, test_power = ([], [])
    for fold in range(10):
        test_rows = np.arange(fold * 5, (fold + 1) * 5)
        train_set = np.delete(split_matrix, test_rows, axis=0)
        test_set = split_matrix[test_rows]
        train_words = 45 * num_classification
        VL_tr, Target_tr, FileInfo_tr = target_signal_gen(
            train_words,
            train_set,
            num_classification,
        )
        train_pointers = _word_pointers(train_sets, train_words)
        X_tr = _assemble_features(
            state_bank,
            FileInfo_tr,
            VL_tr,
            train_pointers,
            num_node,
        )
        model = Ridge(alpha=0.0)
        model.fit(X_tr, Target_tr)
        Output_tr = model.predict(X_tr)
        cm_tr, correct_tr = conmat_acc(
            train_words,
            Output_tr,
            VL_tr,
            num_classification,
        )
        train_con_mat.append(cm_tr)
        train_acc.append(round(correct_tr / train_words * 100, 2))
        test_words = 5 * num_classification
        VL_ts, Target_ts, FileInfo_ts = target_signal_gen(
            test_words,
            test_set,
            num_classification,
        )
        test_pointers = _word_pointers(test_set_devices, test_words)
        X_ts = _assemble_features(
            state_bank,
            FileInfo_ts,
            VL_ts,
            test_pointers,
            num_node,
        )
        Output_ts = model.predict(X_ts)
        cm_ts, correct_ts = conmat_acc(test_words, Output_ts, VL_ts, num_classification)
        test_con_mat.append(cm_ts)
        test_acc.append(round(correct_ts / test_words * 100, 2))
        if verbose:
            print(
                'Fold No.{}, training acc = {:.2f}%, testing acc = {:.2f}%'.format(
                    fold + 1,
                    correct_tr / train_words * 100,
                    correct_ts / test_words * 100,
                ),
            )
    if OUTPUT:
        return (
            train_acc,
            train_con_mat,
            train_power,
            test_acc,
            test_con_mat,
            test_power,
        )
    output_dir = _voice_root('outputs')
    os.makedirs(output_dir, exist_ok=True)
    _save_confusion_matrices(
        os.path.join(output_dir, '{}_train_con_mat{}.csv'.format(mode_name, suffix)),
        train_con_mat,
    )
    _save_confusion_matrices(
        os.path.join(output_dir, '{}_test_con_mat{}.csv'.format(mode_name, suffix)),
        test_con_mat,
    )
    with open(
        os.path.join(output_dir, '{}_both_acc{}.csv'.format(mode_name, suffix)),
        'w',
        newline='',
        encoding='utf-8',
    )as file:
        csv.writer(file).writerows(np.asarray([train_acc, test_acc]))
    return None


def _classification_metrics(
    output,
    target,
    lengths,
    num_classification=10,
):
    """Word-level accuracy and macro recall/precision/F1."""
    offsets = _offsets(lengths)
    confusion = np.zeros((num_classification, num_classification), dtype=int)
    for word_index in range(len(lengths)):
        sl = slice(int(offsets[word_index]), int(offsets[word_index + 1]))
        predicted = int(np.argmax(np.mean(output[sl], axis=0)))
        truth = int(np.argmax(np.mean(target[sl], axis=0)))
        confusion[truth, predicted] += 1
    diag = np.diag(confusion).astype(float)
    row_sum = confusion.sum(axis=1).astype(float)
    col_sum = confusion.sum(axis=0).astype(float)
    recall = np.divide(diag, row_sum, out=np.zeros_like(diag), where=row_sum > 0)
    precision = np.divide(diag, col_sum, out=np.zeros_like(diag), where=col_sum > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diag),
        where=precision + recall > 0,
    )
    return {
        'accuracy': float(diag.sum() / max(1, confusion.sum()) * 100.0),
        'recall': float(np.mean(recall) * 100.0),
        'precision': float(np.mean(precision) * 100.0),
        'f1': float(np.mean(f1) * 100.0),
        'confusion_matrix': confusion,
    }


def _train_eval(
    X_train,
    target_train,
    lengths_train,
    X_test,
    target_test,
    lengths_test,
    alpha=0.0,
):
    model = Ridge(alpha=float(alpha))
    model.fit(X_train, target_train)
    train_output = model.predict(X_train)
    test_output = model.predict(X_test)
    result = _classification_metrics(test_output, target_test, lengths_test)
    train_result = _classification_metrics(train_output, target_train, lengths_train)
    result.update({
        'train_accuracy': train_result['accuracy'],
        'train_recall': train_result['recall'],
        'train_precision': train_result['precision'],
        'train_f1': train_result['f1'],
        'train_confusion_matrix': train_result['confusion_matrix'],
        'selected_alpha': float(alpha),
        'n_train_rows': int(X_train.shape[0]),
        'n_features': int(X_train.shape[1]),
    })
    return result


def _frame_nrmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    scale = float(np.std(y_true))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return rmse if scale == 0 else rmse / scale


def _select_ridge_alpha(
    X,
    target,
    alpha_grid=RIDGE_CV_ALPHAS,
    n_folds=RIDGE_CV_FOLDS,
):
    """Blocked CV on source training frames only, matching the reviewer baseline."""
    indices = np.arange(len(X))
    validation_blocks = np.array_split(indices, max(2, min(int(n_folds), len(X))))
    rows = []
    for alpha in tuple((float(value) for value in alpha_grid)):
        scores = []
        for validation in validation_blocks:
            training = np.setdiff1d(indices, validation, assume_unique=True)
            try:
                model = Ridge(alpha=alpha)
                model.fit(X[training], target[training])
                scores.append(_frame_nrmse(
                    target[validation],
                    model.predict(X[validation]),
                ))
            except Exception:
                scores.append(np.inf)
        rows.append((alpha, float(np.mean(scores)), float(np.std(scores, ddof=1))))
    finite = [row for row in rows if np.isfinite(row[1])]
    if not finite:
        raise RuntimeError('All Ridge-CV alpha candidates failed.')
    return (min(finite, key=lambda row: row[1])[0], rows)


def _switched_from_source_features(source_features, lengths):
    """Compose TS from contiguous word blocks of already assembled source matrices."""
    output = np.empty_like(source_features[0])
    offsets = _offsets(lengths)
    for source_index, word_indices in enumerate(
        np.array_split(np.arange(len(lengths)), len(source_features)),
    ):
        row0 = int(offsets[word_indices[0]])
        row1 = int(offsets[word_indices[-1] + 1])
        output[row0:row1] = source_features[source_index][row0:row1]
    return output


def _output_ensemble(
    source_features,
    target_train,
    lengths_train,
    X_test,
    target_test,
    lengths_test,
):
    train_outputs, test_outputs, train_metrics = ([], [], [])
    for X_source in source_features:
        model = Ridge(alpha=0.0)
        model.fit(X_source, target_train)
        train_output = model.predict(X_source)
        train_outputs.append(train_output)
        test_outputs.append(model.predict(X_test))
        train_metrics.append(_classification_metrics(
            train_output,
            target_train,
            lengths_train,
        ))
    mean_train_output = np.mean(np.stack(train_outputs, axis=0), axis=0)
    mean_test_output = np.mean(np.stack(test_outputs, axis=0), axis=0)
    result = _classification_metrics(mean_test_output, target_test, lengths_test)
    train_mean = _classification_metrics(mean_train_output, target_train, lengths_train)
    result.update({
        'train_accuracy': float(np.mean([x['accuracy'] for x in train_metrics])),
        'train_recall': float(np.mean([x['recall'] for x in train_metrics])),
        'train_precision': float(np.mean([x['precision'] for x in train_metrics])),
        'train_f1': float(np.mean([x['f1'] for x in train_metrics])),
        'train_confusion_matrix': train_mean['confusion_matrix'],
        'selected_alpha': 0.0,
        'n_train_rows': int(len(source_features) * source_features[0].shape[0]),
        'n_features': int(X_test.shape[1]),
    })
    return result


def _write_dict_rows(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_confusion_matrix(confusion, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(4, 3.5))
    sns.set(font_scale=0.8)
    ax = sns.heatmap(
        np.asarray(confusion, dtype=int),
        annot=False,
        fmt='d',
        cmap='Blues',
        linewidths=0.5,
        linecolor='white',
        cbar=True,
    )
    labels = [str(i) for i in range(10)]
    ax.set_xlabel('Predicted label', fontsize=8)
    ax.set_ylabel('True label', fontsize=8)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels, rotation=0)
    plt.tight_layout()
    plt.savefig(path, format='svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


def Voice_baseline_compare(
    num_node=DEFAULT_NUM_NODE,
    down_sampling_start=DEFAULT_DOWNSAMPLING_START,
    state_bank=None,
    split_matrix=None,
    random_seed=0,
    rebuild_cache=False,
    ridge_cv_alphas=RIDGE_CV_ALPHAS,
    ridge_cv_folds=RIDGE_CV_FOLDS,
    suffix='',
    verbose=True,
):
    """Compare TS, output ensemble, state average and source-only Ridge-CV.

    The three source sets are TS_SETS[:, :3]; the target set is (0, 1, 2).
    Ensemble and state average use all three source sets. Ridge-CV uses only
    the first source set and never uses target labels for alpha selection.
    """
    if state_bank is None:
        state_bank = load_response_state_bank(
            num_node=num_node,
            down_sampling_start=down_sampling_start,
            rebuild=rebuild_cache,
            verbose=verbose,
        )
    if split_matrix is None:
        split_matrix = load_or_create_cv_split(random_seed=random_seed)
    source_sets = TS_SETS[:, :3]
    target_set = np.arange(3, dtype=int).reshape(1, -1)
    methods = ('TS', 'ensemble', 'state-average', 'ridge-CV')
    fold_rows = []
    test_confusions = {method: [] for method in methods}
    train_confusions = {method: [] for method in methods}
    cv_rows = []
    for fold in range(10):
        test_rows = np.arange(fold * 5, (fold + 1) * 5)
        train_set = np.delete(split_matrix, test_rows, axis=0)
        test_set = split_matrix[test_rows]
        train_words = 450
        VL_tr, Target_tr, FileInfo_tr = target_signal_gen(train_words, train_set, 10)
        source_features = [
            _assemble_features(
                state_bank,
                FileInfo_tr,
                VL_tr,
                np.repeat(
                    device_set.reshape(1, -1),
                    train_words,
                    axis=0,
                ),
                num_node,
            )
            for device_set in source_sets
        ]
        test_words = 50
        VL_ts, Target_ts, FileInfo_ts = target_signal_gen(test_words, test_set, 10)
        X_test = _assemble_features(
            state_bank,
            FileInfo_ts,
            VL_ts,
            np.repeat(target_set, test_words, axis=0),
            num_node,
        )
        X_ts = _switched_from_source_features(source_features, VL_tr)
        results = {
            'TS': _train_eval(
                X_ts,
                Target_tr,
                VL_tr,
                X_test,
                Target_ts,
                VL_ts,
                alpha=0.0,
            ),
            'ensemble': _output_ensemble(
                source_features,
                Target_tr,
                VL_tr,
                X_test,
                Target_ts,
                VL_ts,
            ),
        }
        X_average = np.zeros_like(source_features[0])
        for X_source in source_features:
            X_average += X_source
        X_average /= len(source_features)
        results['state-average'] = _train_eval(
            X_average,
            Target_tr,
            VL_tr,
            X_test,
            Target_ts,
            VL_ts,
            alpha=0.0,
        )
        selected_alpha, alpha_rows = _select_ridge_alpha(
            source_features[0],
            Target_tr,
            alpha_grid=ridge_cv_alphas,
            n_folds=ridge_cv_folds,
        )
        results['ridge-CV'] = _train_eval(
            source_features[0],
            Target_tr,
            VL_tr,
            X_test,
            Target_ts,
            VL_ts,
            alpha=selected_alpha,
        )
        for alpha, mean_score, std_score in alpha_rows:
            cv_rows.append({
                'fold': fold + 1,
                'alpha': alpha,
                'mean_cv_nrmse': mean_score,
                'std_cv_nrmse': std_score,
                'selected': int(alpha == selected_alpha),
            })
        for method, result in results.items():
            test_confusions[method].append(result['confusion_matrix'])
            train_confusions[method].append(result['train_confusion_matrix'])
            fold_rows.append({
                'fold': fold + 1,
                'method': method,
                'train_accuracy': result['train_accuracy'],
                'accuracy': result['accuracy'],
                'recall': result['recall'],
                'precision': result['precision'],
                'f1': result['f1'],
                'selected_alpha': result['selected_alpha'],
                'n_train_rows': result['n_train_rows'],
                'n_features': result['n_features'],
                'source_sets': str(
                    tuple(
                        (tuple((int(x) for x in row)) for row in source_sets),
                    )
                    if method != 'ridge-CV'
                    else (
                        tuple(int(x) for x in source_sets[0]),
                    ),
                ),
                'target_set': str(tuple((int(x) for x in target_set[0]))),
            })
        if verbose:
            scores = ', '.join(
                '{}={:.2f}%'.format(
                    method,
                    results[method]['accuracy'],
                )
                for method in methods
            )
            print('Baseline fold {}: {}'.format(fold + 1, scores))
    output_dir = _voice_root('outputs', 'baseline_comparison')
    figure_dir = os.path.join('.', 'Figure', 'Voice', DEVICE, 'baseline_comparison')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    suffix_text = str(suffix)
    _write_dict_rows(
        os.path.join(output_dir, 'digit_baseline_foldlevel{}.csv'.format(suffix_text)),
        fold_rows,
    )
    _write_dict_rows(
        os.path.join(output_dir, 'digit_ridge_cv_details{}.csv'.format(suffix_text)),
        cv_rows,
    )
    metric_names = ('train_accuracy', 'accuracy', 'recall', 'precision', 'f1')
    summary_rows = []
    for method in methods:
        rows = [row for row in fold_rows if row['method'] == method]
        summary = {'method': method, 'n_folds': len(rows)}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[metric + '_mean'] = float(np.mean(values))
            summary[metric + '_std'] = float(np.std(values, ddof=1))
            summary[metric + '_median'] = float(np.median(values))
        summary['selected_alpha_by_fold'] = ','.join(
            (str(row['selected_alpha']) for row in rows),
        )
        summary['source_sets'] = rows[0]['source_sets']
        summary['target_set'] = rows[0]['target_set']
        summary_rows.append(summary)
        method_tag = method.lower().replace('-', '_')
        train_path = os.path.join(
            output_dir,
            '{}_train_con_mat{}.csv'.format(method_tag, suffix_text),
        )
        test_path = os.path.join(
            output_dir,
            '{}_test_con_mat{}.csv'.format(method_tag, suffix_text),
        )
        _save_confusion_matrices(train_path, train_confusions[method])
        _save_confusion_matrices(test_path, test_confusions[method])
        aggregate = np.sum(test_confusions[method], axis=0).astype(int)
        with open(
            os.path.join(
                output_dir,
                '{}_aggregate_test_confusion{}.csv'.format(method_tag, suffix_text),
            ),
            'w',
            newline='',
            encoding='utf-8',
        )as file:
            csv.writer(file).writerows(aggregate)
        _plot_confusion_matrix(
            aggregate,
            os.path.join(
                figure_dir,
                '{}_aggregate_test_confusion{}.svg'.format(method_tag, suffix_text),
            ),
        )
    _write_dict_rows(
        os.path.join(output_dir, 'digit_baseline_summary{}.csv'.format(suffix_text)),
        summary_rows,
    )
    standard_output_dir = _voice_root('outputs')
    _save_confusion_matrices(
        os.path.join(standard_output_dir, 'TS_train_con_mat{}.csv'.format(suffix_text)),
        train_confusions['TS'],
    )
    _save_confusion_matrices(
        os.path.join(standard_output_dir, 'TS_test_con_mat{}.csv'.format(suffix_text)),
        test_confusions['TS'],
    )
    ts_rows = [row for row in fold_rows if row['method'] == 'TS']
    with open(
        os.path.join(standard_output_dir, 'TS_both_acc{}.csv'.format(suffix_text)),
        'w',
        newline='',
        encoding='utf-8',
    )as file:
        csv.writer(file).writerows([
            [row['train_accuracy'] for row in ts_rows],
            [row['accuracy'] for row in ts_rows],
        ])
    return summary_rows


def _save_confusion_matrices(path, matrices):
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for matrix in matrices:
            writer.writerows(matrix)
            writer.writerow(['---END_OF_MATRIX---'])


def _load_confusion_matrices(path, num_classification):
    matrices, current = ([], [])
    with open(path, newline='', encoding='utf-8') as file:
        for row in csv.reader(file):
            if row and row[0] == '---END_OF_MATRIX---':
                if current:
                    matrices.append(np.asarray(current, dtype=float))
                    current = []
            elif row:
                current.append(row[:num_classification])
    if current:
        matrices.append(np.asarray(current, dtype=float))
    return matrices


def con_mat_plot(
    filename,
    Device=DEVICE,
    num_classification=10,
):
    if Device != DEVICE:
        raise ValueError('This file is configured for {}.'.format(DEVICE))
    path = _voice_root('outputs', filename + '.csv')
    matrices = _load_confusion_matrices(path, num_classification)
    confusion_matrix = np.rint(np.sum(matrices, axis=0)).astype(int)
    plt.figure(figsize=(4, 3.5))
    sns.set(font_scale=0.8)
    ax = sns.heatmap(
        confusion_matrix,
        annot=False,
        fmt='d',
        cmap='Blues',
        linewidths=0.5,
        linecolor='white',
        cbar=True,
    )
    labels = [str(i) for i in range(num_classification)]
    ax.set_xlabel('Predicted label', fontsize=8)
    ax.set_ylabel('True label', fontsize=8)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels, rotation=0)
    plt.tight_layout()
    figure_dir = os.path.join('.', 'Figure', 'Voice', DEVICE)
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(
        os.path.join(figure_dir, filename + '.svg'),
        format='svg',
        dpi=300,
        bbox_inches='tight',
        transparent=True,
    )
    plt.show()
    plt.close()


if __name__ == '__main__':
    for directory in (
        os.path.join('.', 'Figure', 'Voice', DEVICE),
        _voice_root('inputs'),
        _voice_root('outputs'),
        _voice_root('response'),
        _voice_root('cache'),
    ):
        os.makedirs(directory, exist_ok=True)

    # # please disable the following function when the voice signal file is generated;
    # # this function is placed here to show how the data is generated
    # create_Voice_signal_file()

    num_classification = 10
    random_seed = 0
    state_bank = load_response_state_bank(
        num_node=40,
        down_sampling_start=15,
        rebuild=False,
        verbose=True,
    )
    split_matrix = load_or_create_cv_split(random_seed=random_seed)
    Voice_SRC_exp(
        parallel=True,
        channels=3,
        identical=True,
        num_classification=num_classification,
        state_bank=state_bank,
        split_matrix=split_matrix,
    )
    Voice_SRC_exp(
        parallel=True,
        channels=3,
        direct_transfer=True,
        num_classification=num_classification,
        state_bank=state_bank,
        split_matrix=split_matrix,
    )
    Voice_baseline_compare(
        state_bank=state_bank,
        split_matrix=split_matrix,
        num_node=40,
        down_sampling_start=15,
        verbose=True,
    )
    for name in (
        'DT_train_con_mat',
        'DT_test_con_mat',
        'ID_train_con_mat',
        'ID_test_con_mat',
        'TS_train_con_mat',
        'TS_test_con_mat',
    ):
        con_mat_plot(name)


import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
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
import numpy as np
from sklearn.linear_model import Ridge
TRIAL_DEVICE = 'TiOx'
# Integrated accelerated backend for the TiOx 1000-trial study
TRIAL_DEVICE_SET = ('15d', '16d', '4u', '11u')
TRIAL_PREFIX = ('15d_', '16d_', '4u_', '11u_')
TRIAL_VOICE_NUM_CLASS = 10
TRIAL_VOICE_NUM_FOLDS = 10
TRIAL_MAX_VOICE_STEPS = 25
TRIAL_DEFAULT_NUM_NODE = 40
TRIAL_DEFAULT_DOWNSAMPLING_START = 15
TRIAL_DEFAULT_ALPHA = 1e-16
TRIAL_RIDGE_CV_ALPHAS = (
    0.0,
    1e-16,
    1e-14,
    1e-12,
    1e-10,
    1e-08,
    1e-06,
    0.0001,
    0.01,
    1.0,
    100.0,
)
TRIAL_RIDGE_CV_FOLDS = 5


def trial_voice_root(
    *parts: str,
    project_root: Union[str, Path]='.',
) -> Path:
    return Path(project_root) / 'Data' / 'Voice' / TRIAL_DEVICE / Path(*parts)


def trial_response_file(
    device_index: int,
    digit: int,
    subject: int,
    set_number: int,
    project_root: Union[str, Path]='.',
) -> Path:
    code = TRIAL_DEVICE_SET[int(device_index)]
    name = f'{TRIAL_PREFIX[int(device_index,)]}0{int(digit)}f{int(subject)}set{int(set_number)}.csv'
    return trial_voice_root('response', code, name, project_root=project_root)


def trial_state_bank_path(
    project_root: Union[str, Path]='.',
    num_node: int=TRIAL_DEFAULT_NUM_NODE,
    down_sampling_start: int=TRIAL_DEFAULT_DOWNSAMPLING_START,
) -> Path:
    path = trial_voice_root('cache', project_root=project_root)
    path.mkdir(parents=True, exist_ok=True)
    return path / f'response_statebank_nodes{int(num_node)}_start{int(down_sampling_start,)}_float64_v3.npy'


def _trial_read_response_state(
    path: Union[str, Path],
    num_node: int,
    down_sampling_start: int,
) -> np.ndarray:
    """Read the 20,000 numeric current samples.

    The instrument header is located without decoding the whole file.
    """
    path = Path(path)
    lines = path.read_bytes().splitlines()
    header = next(
        (i for i, line in enumerate(lines) if b'MeasResult2_value' in line),
        None,
    )
    if header is None:
        start, current_col = (148, 3)
    else:
        fields = lines[header].replace(b'\x00', b'').split(b',')
        current_col = next(
            (i for i, field in enumerate(fields) if b'MeasResult2_value' in field),
        )
        start = header + 1
    values: List[float] = []
    for line in lines[start:]:
        fields = line.replace(b'\x00', b'').split(b',')
        if len(fields) <= current_col:
            continue
        token = fields[current_col].strip().strip(b'"')
        try:
            values.append(-float(token))
        except ValueError:
            continue
        if len(values) == 20000:
            break
    if len(values) != 20000:
        raise ValueError(
            f'{path} contains {len(values)} valid current samples; '
            'expected 20000'
        )
    current = np.asarray(values, dtype=np.float64)
    ratio = int(len(current) / TRIAL_MAX_VOICE_STEPS / int(num_node))
    required = TRIAL_MAX_VOICE_STEPS * int(num_node)
    sampled = current[int(down_sampling_start)::ratio][:required]
    if sampled.size != required:
        raise ValueError(f'{path} produced {sampled.size} samples; expected {required}')
    return sampled.reshape(TRIAL_MAX_VOICE_STEPS, int(num_node))


def trial_load_state_bank(
    project_root: Union[str, Path]='.',
    num_node: int=TRIAL_DEFAULT_NUM_NODE,
    down_sampling_start: int=TRIAL_DEFAULT_DOWNSAMPLING_START,
    rebuild: bool=False,
    verbose: bool=True,
) -> np.ndarray:
    path = trial_state_bank_path(project_root, num_node, down_sampling_start)
    shape = (len(TRIAL_DEVICE_SET), 5, 10, 10, TRIAL_MAX_VOICE_STEPS, int(num_node))
    if path.exists() and (not rebuild):
        bank = np.load(path, mmap_mode='r')
        if bank.shape != shape:
            raise ValueError(f'State-bank shape {bank.shape} != {shape}; rebuild it')
        return bank
    tmp = Path(str(path) + '.tmp.npy')
    tmp.unlink(missing_ok=True)
    bank = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float64, shape=shape)
    total = int(np.prod(shape[:4]))
    count = 0
    try:
        for device in range(len(TRIAL_DEVICE_SET)):
            for subject in range(1, 6):
                for digit in range(10):
                    for set_number in range(10):
                        bank[
                            device,
                            subject - 1,
                            digit,
                            set_number,
                        ] = _trial_read_response_state(
                            trial_response_file(
                                device,
                                digit,
                                subject,
                                set_number,
                                project_root,
                            ),
                            int(num_node),
                            int(down_sampling_start),
                        )
                        count += 1
                        if verbose and (count % 100 == 0 or count == total):
                            print(f'[Digit state bank] {count}/{total}')
        bank.flush()
    except Exception:
        del bank
        tmp.unlink(missing_ok=True)
        raise
    del bank
    os.replace(tmp, path)
    return np.load(path, mmap_mode='r')

@lru_cache(maxsize=8)


def _trial_steps_rec_cached(path_text: str) -> np.ndarray:
    return np.loadtxt(path_text, delimiter=',', usecols=0, dtype=np.int32)


def trial_load_or_create_cv_split(
    project_root: Union[str, Path]='.',
    seed: int=0,
    rebuild: bool=False,
) -> np.ndarray:
    out = trial_voice_root('inputs', project_root=project_root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'cv_split_seed{int(seed)}.npy'
    if path.exists() and (not rebuild):
        arr = np.load(path)
        if arr.shape != (50, 10, 3):
            raise ValueError(f'Unexpected CV split shape: {arr.shape}')
        return arr
    base = np.empty((50, 10, 3), dtype=np.int16)
    row = 0
    for subject in range(1, 6):
        for set_number in range(10):
            for digit in range(10):
                base[row, digit] = (digit, subject, set_number)
            row += 1
    rng = np.random.default_rng(int(seed))
    split = np.empty_like(base)
    for digit in range(10):
        split[:, digit] = base[rng.permutation(50), digit]
    np.save(path, split)
    return split


def trial_make_cv_folds(split: np.ndarray) -> List[Dict[
    str,
    Union[np.ndarray, int],
]]:
    folds = []
    for fold in range(TRIAL_VOICE_NUM_FOLDS):
        test_rows = np.arange(fold * 5, (fold + 1) * 5)
        folds.append({
            'fold': fold,
            'train_set': np.delete(split, test_rows, axis=0),
            'test_set': split[test_rows],
        })
    return folds


def trial_target_signal(
    file_matrix: np.ndarray,
    project_root: Union[str, Path]='.',
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    file_info = np.asarray(file_matrix, dtype=np.int16).reshape(-1, 3)
    steps = _trial_steps_rec_cached(
        str(trial_voice_root(
            'inputs',
            'steps_rec.csv',
            project_root=project_root,
        ).resolve()),
    )
    digit = file_info[:, 0].astype(int)
    subject = file_info[:, 1].astype(int)
    set_number = file_info[:, 2].astype(int)
    lengths = steps[set_number + digit * 10 + (subject - 1) * 100].astype(int)
    labels = np.tile(np.arange(TRIAL_VOICE_NUM_CLASS), file_matrix.shape[0])
    frame_labels = np.repeat(labels, lengths)
    target = np.zeros((int(lengths.sum()), TRIAL_VOICE_NUM_CLASS), dtype=np.float64)
    target[np.arange(len(frame_labels)), frame_labels] = 1.0
    return (lengths.tolist(), target, file_info)


def trial_offsets(lengths: Sequence[int]) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(np.asarray(lengths, dtype=np.int64))))


def trial_assemble_device_matrix(
    bank: np.ndarray,
    file_info: np.ndarray,
    lengths: Sequence[int],
    device: int,
    num_node: int=TRIAL_DEFAULT_NUM_NODE,
) -> np.ndarray:
    off = trial_offsets(lengths)
    X = np.empty((int(off[-1]), int(num_node)), dtype=np.float64)
    for word, (
        digit,
        subject,
        set_number,
    )in enumerate(np.asarray(file_info, dtype=int)):
        r0, r1 = (int(off[word]), int(off[word + 1]))
        X[r0:r1] = bank[
            int(device),
            subject - 1,
            digit,
            set_number,
            :r1 - r0,
            :int(num_node),
        ]
    return X


def trial_classification_metrics(
    output: np.ndarray,
    target: np.ndarray,
    lengths: Sequence[int],
) -> Dict:
    off = trial_offsets(lengths)
    cm = np.zeros((TRIAL_VOICE_NUM_CLASS, TRIAL_VOICE_NUM_CLASS), dtype=int)
    for word in range(len(lengths)):
        sl = slice(int(off[word]), int(off[word + 1]))
        pred = int(np.argmax(np.mean(output[sl], axis=0)))
        truth = int(np.argmax(np.mean(target[sl], axis=0)))
        cm[truth, pred] += 1
    diag = np.diag(cm).astype(float)
    rows = cm.sum(axis=1).astype(float)
    cols = cm.sum(axis=0).astype(float)
    recall = np.divide(diag, rows, out=np.zeros_like(diag), where=rows > 0)
    precision = np.divide(diag, cols, out=np.zeros_like(diag), where=cols > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diag),
        where=precision + recall > 0,
    )
    return {
        'accuracy': float(diag.sum() / max(1, cm.sum()) * 100),
        'recall': float(recall.mean() * 100),
        'precision': float(precision.mean() * 100),
        'f1': float(f1.mean() * 100),
        'confusion_matrix': cm,
    }


def trial_fit_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lengths_train: Sequence[int],
    X_test: np.ndarray,
    y_test: np.ndarray,
    lengths_test: Sequence[int],
    alpha: float=TRIAL_DEFAULT_ALPHA,
    return_model: bool=False,
) -> Dict:
    model = Ridge(alpha=float(alpha), solver='cholesky')
    model.fit(X_train, y_train)
    tr = trial_classification_metrics(model.predict(X_train), y_train, lengths_train)
    ts = trial_classification_metrics(model.predict(X_test), y_test, lengths_test)
    out = {
        **{f'train_{k}': v for k, v in tr.items() if k != 'confusion_matrix'},
        **{k: v for k, v in ts.items()},
        'train_confusion_matrix': tr['confusion_matrix'],
        'selected_alpha': float(alpha),
        'n_train_rows': int(X_train.shape[0]),
        'n_features': int(X_train.shape[1]),
    }
    if return_model:
        out['model'] = model
    return out


def _trial_frame_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scale = float(np.std(y_true))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return rmse if scale == 0 else rmse / scale


def trial_select_ridge_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alpha_grid: Sequence[float]=TRIAL_RIDGE_CV_ALPHAS,
    n_folds: int=TRIAL_RIDGE_CV_FOLDS,
) -> Tuple[float, List[Dict]]:
    idx = np.arange(len(X))
    blocks = np.array_split(idx, max(2, min(int(n_folds), len(X))))
    rows = []
    for alpha in map(float, alpha_grid):
        scores = []
        for va in blocks:
            tr = np.setdiff1d(idx, va, assume_unique=True)
            try:
                model = Ridge(alpha=alpha, solver='cholesky').fit(X[tr], y[tr])
                scores.append(_trial_frame_nrmse(y[va], model.predict(X[va])))
            except Exception:
                scores.append(np.inf)
        rows.append({
            'alpha': alpha,
            'mean_cv_nrmse': float(np.mean(scores)),
            'std_cv_nrmse': float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        })
    finite = [r for r in rows if np.isfinite(r['mean_cv_nrmse'])]
    if not finite:
        raise RuntimeError('All Ridge-CV candidates failed')
    return (float(min(finite, key=lambda r: r['mean_cv_nrmse'])['alpha']), rows)

@dataclass


class TiOx1000VoiceFoldCache:
    bank: np.ndarray
    train_set: np.ndarray
    test_set: np.ndarray
    project_root: Union[str, Path] = '.'
    num_node: int = TRIAL_DEFAULT_NUM_NODE
    lengths_train: List[int] = field(init=False)
    target_train: np.ndarray = field(init=False)
    file_info_train: np.ndarray = field(init=False)
    lengths_test: List[int] = field(init=False)
    target_test: np.ndarray = field(init=False)
    file_info_test: np.ndarray = field(init=False)
    train_device: Dict[int, np.ndarray] = field(init=False, default_factory=dict)
    test_device: Dict[int, np.ndarray] = field(init=False, default_factory=dict)
    fixed_models: Dict[Tuple[int, ...], Dict] = field(init=False, default_factory=dict)
    ridge_models: Dict[Tuple[int, ...], Dict] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        (
            self.lengths_train,
            self.target_train,
            self.file_info_train,
        ) = trial_target_signal(
            self.train_set,
            self.project_root,
        )
        self.lengths_test, self.target_test, self.file_info_test = trial_target_signal(
            self.test_set,
            self.project_root,
        )
        for d in range(len(TRIAL_DEVICE_SET)):
            self.train_device[d] = trial_assemble_device_matrix(
                self.bank,
                self.file_info_train,
                self.lengths_train,
                d,
                self.num_node,
            )
            self.test_device[d] = trial_assemble_device_matrix(
                self.bank,
                self.file_info_test,
                self.lengths_test,
                d,
                self.num_node,
            )

    def set_X(
        self,
        device_set: Sequence[int],
        train: bool=True,
    ) -> np.ndarray:
        store = self.train_device if train else self.test_device
        return np.concatenate([store[int(d)] for d in device_set], axis=1)

    def fixed_source_model(
        self,
        device_set: Sequence[int],
        alpha: float=TRIAL_DEFAULT_ALPHA,
    ) -> Dict:
        key = tuple(map(int, device_set))
        if key not in self.fixed_models:
            X = self.set_X(key, True)
            model = Ridge(
                alpha=float(alpha),
                solver='cholesky',
            ).fit(X, self.target_train)
            tr = trial_classification_metrics(
                model.predict(X),
                self.target_train,
                self.lengths_train,
            )
            self.fixed_models[key] = {
                'model': model,
                'train': tr,
                'selected_alpha': float(alpha),
            }
        return self.fixed_models[key]

    def ridge_source_model(
        self,
        device_set: Sequence[int],
        alpha_grid=TRIAL_RIDGE_CV_ALPHAS,
        n_folds: int=TRIAL_RIDGE_CV_FOLDS,
    ) -> Dict:
        key = tuple(map(int, device_set))
        if key not in self.ridge_models:
            X = self.set_X(key, True)
            alpha, cv = trial_select_ridge_alpha(
                X,
                self.target_train,
                alpha_grid,
                n_folds,
            )
            model = Ridge(alpha=alpha, solver='cholesky').fit(X, self.target_train)
            tr = trial_classification_metrics(
                model.predict(X),
                self.target_train,
                self.lengths_train,
            )
            self.ridge_models[key] = {
                'model': model,
                'train': tr,
                'selected_alpha': alpha,
                'cv': cv,
            }
        return self.ridge_models[key]

    def evaluate_model_on_target(
        self,
        item: Dict,
        target_set: Sequence[int],
    ) -> Dict:
        X_test = self.set_X(target_set, False)
        ts = trial_classification_metrics(
            item['model'].predict(X_test),
            self.target_test,
            self.lengths_test,
        )
        tr = item['train']
        return {
            **ts,
            **{f'train_{k}': v for k, v in tr.items() if k != 'confusion_matrix'},
            'train_confusion_matrix': tr['confusion_matrix'],
            'selected_alpha': float(item['selected_alpha']),
            'n_train_rows': int(self.target_train.shape[0]),
            'n_features': int(X_test.shape[1]),
        }


def trial_switched_X(
    source_X: Sequence[np.ndarray],
    lengths: Sequence[int],
) -> np.ndarray:
    out = np.empty_like(source_X[0])
    off = trial_offsets(lengths)
    for source_index, words in enumerate(
        np.array_split(np.arange(len(lengths)), len(source_X)),
    ):
        r0, r1 = (int(off[words[0]]), int(off[words[-1] + 1]))
        out[r0:r1] = source_X[source_index][r0:r1]
    return out


def trial_output_ensemble(
    fold: TiOx1000VoiceFoldCache,
    source_sets: Sequence[Sequence[int]],
    target_set: Sequence[int],
    alpha: float=TRIAL_DEFAULT_ALPHA,
) -> Dict:
    X_test = fold.set_X(target_set, False)
    train_outputs, test_outputs, train_metrics = ([], [], [])
    for source in source_sets:
        item = fold.fixed_source_model(source, alpha)
        X_train = fold.set_X(source, True)
        train_outputs.append(item['model'].predict(X_train))
        test_outputs.append(item['model'].predict(X_test))
        train_metrics.append(item['train'])
    tr_mean = trial_classification_metrics(
        np.mean(np.stack(train_outputs), axis=0),
        fold.target_train,
        fold.lengths_train,
    )
    ts = trial_classification_metrics(
        np.mean(np.stack(test_outputs), axis=0),
        fold.target_test,
        fold.lengths_test,
    )
    return {
        **ts,
        'train_accuracy': float(np.mean([m['accuracy'] for m in train_metrics])),
        'train_recall': float(np.mean([m['recall'] for m in train_metrics])),
        'train_precision': float(np.mean([m['precision'] for m in train_metrics])),
        'train_f1': float(np.mean([m['f1'] for m in train_metrics])),
        'train_confusion_matrix': tr_mean['confusion_matrix'],
        'selected_alpha': float(alpha),
        'n_train_rows': int(len(source_sets) * fold.target_train.shape[0]),
        'n_features': int(X_test.shape[1]),
    }


def trial_evaluate_family(
    fold: TiOx1000VoiceFoldCache,
    combos_by_n: Dict[int, Sequence[Sequence[int]]],
    target_set: Sequence[int],
    alpha: float=TRIAL_DEFAULT_ALPHA,
    ridge_grid=TRIAL_RIDGE_CV_ALPHAS,
    ridge_folds: int=TRIAL_RIDGE_CV_FOLDS,
) -> List[Dict]:
    target_set = tuple(map(int, target_set))
    X_test = fold.set_X(target_set, False)
    unique_sets = {tuple(map(int, s)) for combo in combos_by_n.values() for s in combo}
    X_source = {s: fold.set_X(s, True) for s in unique_sets}
    rows: List[Dict] = []
    for N in (2, 3, 4, 5):
        combo = tuple((tuple(map(int, s)) for s in combos_by_n[N]))
        source_X = [X_source[s] for s in combo]
        results = {
            'TS': trial_fit_eval(
                trial_switched_X(source_X, fold.lengths_train),
                fold.target_train,
                fold.lengths_train,
                X_test,
                fold.target_test,
                fold.lengths_test,
                alpha,
            ),
            'ensemble': trial_output_ensemble(fold, combo, target_set, alpha),
            'state-average': trial_fit_eval(
                np.mean(np.stack(source_X), axis=0),
                fold.target_train,
                fold.lengths_train,
                X_test,
                fold.target_test,
                fold.lengths_test,
                alpha,
            ),
        }
        for method, res in results.items():
            rows.append({
                'N_source_sets': N,
                'display_method': method,
                **res,
            })
    first = tuple(map(int, combos_by_n[3][0]))
    classical = fold.evaluate_model_on_target(
        fold.fixed_source_model(first, alpha),
        target_set,
    )
    ridge_item = fold.ridge_source_model(first, ridge_grid, ridge_folds)
    ridge = fold.evaluate_model_on_target(ridge_item, target_set)
    self_res = fold.evaluate_model_on_target(
        fold.fixed_source_model(target_set, alpha),
        target_set,
    )
    rows.extend([
        {'N_source_sets': 1, 'display_method': 'classical', **classical},
        {
            'N_source_sets': 1,
            'display_method': 'ridge CV',
            **ridge,
            'cv_table_json': json.dumps(ridge_item['cv']),
        },
        {'N_source_sets': 1, 'display_method': 'self-training', **self_res},
    ])
    return rows


def trial_first_words_slice(
    X: np.ndarray,
    target: np.ndarray,
    lengths: Sequence[int],
    n_words: int,
):
    n_words = int(n_words)
    off = trial_offsets(lengths)
    stop = int(off[n_words])
    return (X[:stop], target[:stop], list(lengths[:n_words]))


def trial_evaluate_fewshot(
    fold: TiOx1000VoiceFoldCache,
    source_combo: Sequence[Sequence[int]],
    target_set: Sequence[int],
    shots: Sequence[int],
    alpha: float=TRIAL_DEFAULT_ALPHA,
) -> List[Dict]:
    combo = tuple((tuple(map(int, s)) for s in source_combo))
    target_set = tuple(map(int, target_set))
    source_X = [fold.set_X(s, True) for s in combo]
    X_ts = trial_switched_X(source_X, fold.lengths_train)
    X_single = fold.set_X(combo[0], True)
    X_target_train = fold.set_X(target_set, True)
    X_test = fold.set_X(target_set, False)
    zero = trial_fit_eval(
        X_ts,
        fold.target_train,
        fold.lengths_train,
        X_test,
        fold.target_test,
        fold.lengths_test,
        alpha,
    )
    rows = [{'method': 'TS-zeroshot', 'fewshot': 0, **zero}]
    for k in map(int, shots):
        X_fs, T_fs, L_fs = trial_first_words_slice(
            X_target_train,
            fold.target_train,
            fold.lengths_train,
            k * TRIAL_VOICE_NUM_CLASS,
        )
        T_comb = np.vstack([fold.target_train, T_fs])
        L_comb = list(fold.lengths_train) + L_fs
        rows.append({
            'method': 'TS-fewshot',
            'fewshot': k,
            **trial_fit_eval(
                np.vstack([X_ts, X_fs]),
                T_comb,
                L_comb,
                X_test,
                fold.target_test,
                fold.lengths_test,
                alpha,
            ),
        })
        rows.append({
            'method': 'classical-fewshot',
            'fewshot': k,
            **trial_fit_eval(
                np.vstack([X_single, X_fs]),
                T_comb,
                L_comb,
                X_test,
                fold.target_test,
                fold.lengths_test,
                alpha,
            ),
        })
    return rows
