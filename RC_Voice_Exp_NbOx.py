"""
Train and evaluate reservoir computing for spoken-digit classification.

This version keeps the original ID / DT / TS definitions, but replaces repeated
raw-CSV parsing with a memory-mapped response-state bank.  The first run builds
one cache file; later runs read states directly from that cache.
"""

import csv
import os
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import Ridge
from Voice_Inputs import *


DEVICE = 'NbOx'
DEVICE_SET = ['a', 'b', 'c', 'd']
PREFIX = ['', '', '', '']
DT_SET = np.asarray([1, 2, 2, 0], dtype=int)
TS_SETS = np.asarray([[1, 2, 2, 0], [2, 1, 2, 1], [3, 1, 1, 0]], dtype=int)
MAX_VOICE_STEPS = 25
DEFAULT_NUM_NODE = 40
DEFAULT_DOWNSAMPLING_START = 15


def _voice_root(*parts):
    return os.path.join('.', 'Data', 'Voice', DEVICE, *parts)


def _response_file(device_index, digit, subject, set_number):
    code = DEVICE_SET[int(device_index)]
    name = '{}0{}f{}set{}.csv'.format(
        PREFIX[int(device_index)], int(digit), int(subject), int(set_number)
    )
    if DEVICE == 'NbOx':
        return _voice_root('response', 'Device {}'.format(code), name)
    return _voice_root('response', code, name)


def _state_bank_path(num_node=DEFAULT_NUM_NODE,
                     down_sampling_start=DEFAULT_DOWNSAMPLING_START):
    cache_dir = _voice_root('cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir,
        'response_statebank_nodes{}_start{}_float64_v2.npy'.format(
            int(num_node), int(down_sampling_start)
        ),
    )


def _read_response_state(file_path, num_node, down_sampling_start):
    """Read 20,000 numeric current samples after the instrument CSV header."""
    row_count = 20000

    with open(file_path, 'rb') as handle:
        lines = handle.read().splitlines()

    # Locate the actual table header instead of relying on physical line numbers.
    header_index = next(
        (i for i, line in enumerate(lines) if b'MeasResult2_value' in line),
        None,
    )
    if header_index is None:
        start_index, current_column = 148, 3
    else:
        header_fields = lines[header_index].replace(b'\x00', b'').split(b',')
        current_column = next(
            i for i, field in enumerate(header_fields)
            if b'MeasResult2_value' in field
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
            'Response {} contains {} valid current samples after the data header; expected {}.'.format(
                file_path, len(values), row_count
            )
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
                file_path, sampled.size, required
            )
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

    if os.path.exists(cache_path) and not force:
        bank = np.load(cache_path, mmap_mode='r')
        if bank.shape != expected_shape:
            raise ValueError(
                'Cache shape {} does not match expected {}. Rebuild with force=True.'.format(
                    bank.shape, expected_shape
                )
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
                            device_index, digit, subject, set_number
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
    if os.path.exists(path) and not rebuild:
        split = np.load(path)
        if split.shape != (50, 10, 3):
            raise ValueError('Unexpected split shape: {}'.format(split.shape))
        return split

    # Entries are (digit, subject, set_number).  Each digit has 5 x 10 samples.
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


def target_signal_gen(words, sample_set, num_classification):
    """Build variable lengths, one-hot frame targets and sample metadata."""
    sample_set = np.asarray(sample_set, dtype=np.int16)
    file_info = sample_set.reshape(-1, 3)
    if len(file_info) != int(words):
        raise ValueError('words={} but sample_set contains {} samples.'.format(words, len(file_info)))

    digit = file_info[:, 0].astype(int)
    subject = file_info[:, 1].astype(int)
    set_number = file_info[:, 2].astype(int)
    steps = _load_steps_rec()
    lengths = steps[set_number + digit * 10 + (subject - 1) * 100].astype(int)

    labels = np.tile(np.arange(int(num_classification)), sample_set.shape[0])
    frame_labels = np.repeat(labels, lengths)
    target = np.zeros((int(np.sum(lengths)), int(num_classification)), dtype=float)
    target[np.arange(len(frame_labels)), frame_labels] = 1.0
    return lengths.tolist(), target, file_info


def _offsets(lengths):
    return np.concatenate(([0], np.cumsum(np.asarray(lengths, dtype=np.int64))))


def _assemble_features(state_bank, file_info, lengths, pointers, num_node):
    """Assemble one fold directly from the memory-mapped response state bank."""
    pointers = np.asarray(pointers, dtype=int)
    if pointers.ndim != 2 or pointers.shape[0] != len(file_info):
        raise ValueError('pointers must have shape (n_words, channels).')

    offsets = _offsets(lengths)
    channels = pointers.shape[1]
    X = np.empty((int(offsets[-1]), int(num_node) * channels), dtype=np.float64)

    for word_index, (digit, subject, set_number) in enumerate(file_info.astype(int)):
        row0, row1 = int(offsets[word_index]), int(offsets[word_index + 1])
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


def conmat_acc(words, output, VL, num_classification):
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
    return confusion, correct


def _train_device_sets(parallel, channels, identical, direct_transfer):
    channels = int(channels) if parallel else 1
    if channels < 1 or channels > len(DEVICE_SET):
        raise ValueError('channels must be between 1 and {}.'.format(len(DEVICE_SET)))

    test_set = np.arange(channels, dtype=int).reshape(1, -1) if parallel else np.array([[0]])
    if identical:
        train_sets = test_set.copy()
        mode_name = 'ID'
    elif direct_transfer:
        train_sets = DT_SET[:channels].reshape(1, -1)
        mode_name = 'DT'
    else:
        train_sets = TS_SETS[:, :channels]
        mode_name = 'TS'
    return train_sets, test_set, channels, mode_name


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
        parallel, channels, identical, direct_transfer
    )
    if verbose:
        print({'ID': 'identical', 'DT': 'direct transfer', 'TS': 'temporal switch'}[mode_name])

    if state_bank is None:
        state_bank = load_response_state_bank(
            num_node=num_node,
            down_sampling_start=down_sampling_start,
            rebuild=rebuild_cache,
            verbose=verbose,
        )
    if split_matrix is None:
        split_matrix = load_or_create_cv_split(random_seed=random_seed)

    train_con_mat, test_con_mat = [], []
    train_acc, test_acc = [], []
    train_power, test_power = [], []

    for fold in range(10):
        test_rows = np.arange(fold * 5, (fold + 1) * 5)
        train_set = np.delete(split_matrix, test_rows, axis=0)
        test_set = split_matrix[test_rows]

        train_words = 45 * num_classification
        VL_tr, Target_tr, FileInfo_tr = target_signal_gen(
            train_words, train_set, num_classification
        )
        train_pointers = _word_pointers(train_sets, train_words)
        X_tr = _assemble_features(
            state_bank, FileInfo_tr, VL_tr, train_pointers, num_node
        )

        model = Ridge(alpha=0.0)
        model.fit(X_tr, Target_tr)
        Output_tr = model.predict(X_tr)
        cm_tr, correct_tr = conmat_acc(
            train_words, Output_tr, VL_tr, num_classification
        )
        train_con_mat.append(cm_tr)
        train_acc.append(round(correct_tr / train_words * 100, 2))

        test_words = 5 * num_classification
        VL_ts, Target_ts, FileInfo_ts = target_signal_gen(
            test_words, test_set, num_classification
        )
        test_pointers = _word_pointers(test_set_devices, test_words)
        X_ts = _assemble_features(
            state_bank, FileInfo_ts, VL_ts, test_pointers, num_node
        )
        Output_ts = model.predict(X_ts)
        cm_ts, correct_ts = conmat_acc(
            test_words, Output_ts, VL_ts, num_classification
        )
        test_con_mat.append(cm_ts)
        test_acc.append(round(correct_ts / test_words * 100, 2))

        if verbose:
            print(
                'Fold No.{}, training acc = {:.2f}%, testing acc = {:.2f}%'.format(
                    fold + 1,
                    correct_tr / train_words * 100,
                    correct_ts / test_words * 100,
                )
            )

    if OUTPUT:
        return train_acc, train_con_mat, train_power, test_acc, test_con_mat, test_power

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
            'w', newline='', encoding='utf-8') as file:
        csv.writer(file).writerows(np.asarray([train_acc, test_acc]))
    return None


def _save_confusion_matrices(path, matrices):
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for matrix in matrices:
            writer.writerows(matrix)
            writer.writerow(['---END_OF_MATRIX---'])


def _load_confusion_matrices(path, num_classification):
    matrices, current = [], []
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


def con_mat_plot(filename, Device=DEVICE, num_classification=10):
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
        format='svg', dpi=300, bbox_inches='tight', transparent=True,
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

    # First run: builds the cache. Later runs only memory-map the .npy file.
    state_bank = load_response_state_bank(
        num_node=40,
        down_sampling_start=15,
        rebuild=False,
        verbose=True,
    )
    split_matrix = load_or_create_cv_split(random_seed=random_seed)

    Voice_SRC_exp(
        identical=True, parallel=False, channels=1,
        num_classification=num_classification,
        state_bank=state_bank, split_matrix=split_matrix,
    )
    Voice_SRC_exp(
        direct_transfer=True, parallel=False, channels=1,
        num_classification=num_classification,
        state_bank=state_bank, split_matrix=split_matrix,
    )
    Voice_SRC_exp(
        direct_transfer=False, identical=False, parallel=False, channels=1,
        num_classification=num_classification,
        state_bank=state_bank, split_matrix=split_matrix,
    )

    con_mat_plot('ID_test_con_mat')
    con_mat_plot('TS_test_con_mat')
    con_mat_plot('DT_test_con_mat')
