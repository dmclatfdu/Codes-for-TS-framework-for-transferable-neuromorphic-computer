#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mackey-Glass reservoir-computing demos with TS framework.

The file contains the classical, temporal-switch, temporal-permutation,
state-average, output-ensemble and Ridge-CV experiments under ``if __name__ == '__main__'``.
It also contains supporting functions for the 1000 trials processing.

Zefeng Zhang, Research Institute of Intelligent Complex Systems, Fudan University
"""
from sim_RC_library import *
import matplotlib as mpl


def MG_SRC_sim(
    length=1440,
    shift=1,
    num_node=10,
    mask_abs=0.1,
    direct_transfer=0,
    no_pic=False,
    AVG=0,
    self=False,
    noise_level=1e-06,
    C2C_variation=1e-07,
    C2C_test_control=False,
    noise_test_control=False,
    num_res=3,
    Ts_k3=1.16e-05,
):
    MG_gen = MG_generator(0.2, 0.1, 10, 18, shift=shift)
    signal, target = MG_gen.iterate(1, length)
    mask = create_mask(num_node, abs_value=mask_abs)
    Input_tr, input_ts, Target_tr, target_ts = signal_process(signal, target, mask)
    if not direct_transfer:
        if AVG:
            input_tr = Input_tr[-len(Input_tr):]
            target_tr = Target_tr[-len(Target_tr):]
            target_tr = np.tile(target_tr, (num_res, 1))
        else:
            input_tr = Input_tr[-int(len(Input_tr) / num_res):]
            target_tr = Target_tr[-int(len(Target_tr) / num_res):]
            target_tr = np.tile(target_tr, (num_res, 1))
    else:
        input_tr = Input_tr
        target_tr = Target_tr
        num_res = 1
    SRC = TiOx_SRC()
    Tr_set_k3 = np.linspace(9.6e-06, 1.2e-05, num_res)
    if self is True:
        Tr_set_k3 = np.linspace(Ts_k3, 1.2e-05, 1)
    if not AVG:
        State_tr = np.zeros((int(len(Target_tr) / num_res) * num_res, num_node))
    else:
        State_tr = np.zeros((len(Target_tr) * num_res, num_node))
    for i in range(num_res):
        i_tr, g_tr, g0_tr = SRC.iterate_SRC(
            input_tr,
            2e-05,
            k3=Tr_set_k3[i],
            virtual_nodes=num_node,
            clear=True,
            C2C_strength=C2C_variation,
        )
        if not AVG:
            State_tr[
                i * int(len(Target_tr) / num_res) :
                (i + 1) * int(len(Target_tr) / num_res),
                :,
            ]= i_tr.reshape(int(len(Target_tr) / num_res), num_node)
        else:
            State_tr[
                i * len(Target_tr):(i + 1) * len(Target_tr),
                :,
            ]+= i_tr.reshape(len(Target_tr), num_node)
    if AVG:
        State_tr /= num_res
    State_tr += noise_level * np.random.randn(State_tr.shape[0], State_tr.shape[1])
    lin = Ridge(alpha=0)
    lin.fit(State_tr, target_tr)
    Output_tr = lin.predict(State_tr)
    if C2C_test_control:
        C2C_variation = 0
    if noise_test_control:
        noise_level = 1e-06
    i_ts, g_ts, g0_ts = SRC.iterate_SRC(
        input_ts,
        2e-05,
        k3=Ts_k3,
        virtual_nodes=num_node,
        clear=True,
        C2C_strength=C2C_variation,
    )
    State_ts = i_ts.reshape(len(target_ts), num_node)
    State_ts += noise_level * np.random.randn(State_ts.shape[0], State_ts.shape[1])
    Output_ts = lin.predict(State_ts)
    NRMSE_tr, NRMSE_ts = (nrmse(target_tr, Output_tr), nrmse(target_ts, Output_ts))
    if not no_pic:
        color4 = np.array([107, 158, 184]) / 255
        color3 = np.array([103, 149, 216]) / 255
        color2 = np.array([110, 167, 151]) / 255
        color1 = np.array([117, 185, 86]) / 255
        figure, ax = plt.subplots(2, 2, figsize=(2.4, 2), sharey='row', sharex='col')
        ax1, ax2, ax3, ax4 = (ax[1, 0], ax[1, 1], ax[0, 0], ax[0, 1])
        plt.rc('font', family='Arial', size=6)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['lines.linewidth'] = 1.2
        ylim_max = 0.035
        if direct_transfer:
            ax1.plot(
                (Output_tr[:, 0] - target_tr[:, 0]) ** 2,
                label='Training Error',
                color=color1,
            )
            ax3.plot(target_tr[:, 0], color=np.array([200, 200, 200]) / 255)
            ax3.plot(Output_tr[:, 0], color=color1)
        else:
            colors = [color1, color2, color3]
            for i in range(num_res):
                color = colors[i % num_res]
                if AVG:
                    color = np.array([138, 127, 214]) / 255
                ax1.axvline(240, ls='--', color=np.array([180, 180, 180]) / 255)
                ax1.axvline(480, ls='--', color=np.array([180, 180, 180]) / 255)
                ax3.axvline(240, ls='--', color=np.array([180, 180, 180]) / 255)
                ax3.axvline(480, ls='--', color=np.array([180, 180, 180]) / 255)
                ax1.plot(
                    np.arange(
                        i * int(len(Target_tr) / num_res),
                        (i + 1) * int(len(Target_tr) / num_res),
                    ),
                    (Output_tr[i * int(len(Target_tr) / num_res) :
                (i + 1) * int(len(Target_tr) / num_res), 0] - target_tr[
                        i * int(len(Target_tr) / num_res) :
                (i + 1) * int(len(Target_tr) / num_res),
                        0,
                    ]) ** 2,
                    label='Training Error',
                    color=color,
                )
                ax3.plot(
                    np.arange(
                        i * int(len(Target_tr) / num_res),
                        (i + 1) * int(len(Target_tr) / num_res),
                    ),
                    target_tr[
                        i * int(len(Target_tr) / num_res) :
                (i + 1) * int(len(Target_tr) / num_res),
                        0,
                    ],
                    color=np.array([200, 200, 200]) / 255,
                )
                ax3.plot(
                    np.arange(
                        i * int(len(Target_tr) / num_res),
                        (i + 1) * int(len(Target_tr) / num_res),
                    ),
                    Output_tr[
                        i * int(len(Target_tr) / num_res) :
                (i + 1) * int(len(Target_tr) / num_res),
                        0,
                    ],
                    color=color,
                )
        ax1.set_xlim(0, 720)
        ax1.set_ylim(0, ylim_max)
        ax3.set_ylim(0.2, 1.6)
        ax1.set_xlabel(
            'Time step',
            fontdict={'family': 'arial', 'size': 6},
            labelpad=1,
            x=0.8,
            ha='left',
        )
        ax1.set_ylabel('Squared error', fontdict={'family': 'arial', 'size': 6})
        ax3.set_ylabel('$x$', fontdict={'family': 'arial', 'size': 6})
        ax1.tick_params(axis='both', direction='in', labelsize=6)
        ax3.tick_params(axis='both', direction='in', labelsize=6)
        ax1.set_xticks([0, 240, 480, 720])
        ax1.set_yticks([0, 0.01, 0.02, 0.03])
        ax2.plot(
            np.arange(720, 1440),
            (Output_ts[:, 0] - target_ts[:, 0]) ** 2,
            label='Testing Error',
            color=color4,
        )
        ax4.plot(
            np.arange(720, 1440),
            target_ts[:, 0],
            color=np.array([200, 200, 200]) / 255,
        )
        ax4.plot(np.arange(720, 1440), Output_ts[:, 0], color=color4)
        ax2.set_xlim(720, 1440)
        ax2.set_ylim(0, ylim_max)
        ax2.set_xticks([960, 1200, 1440])
        ax2.tick_params(axis='both', direction='in', labelsize=6)
        ax4.tick_params(axis='both', direction='in', labelsize=6)
        figure.subplots_adjust(wspace=0, hspace=0.1)
        if AVG:
            suffix = '_AVG'
        else:
            suffix = ''
        if not direct_transfer:
            plt.savefig(
                './Figure/MG/Sim_Error_SRC_SC{}.svg'.format(suffix),
                dpi=300,
                format='svg',
                transparent=True,
                bbox_inches='tight',
            )
        else:
            plt.savefig(
                './Figure/MG/Sim_Error_DT_SC{}.svg'.format(suffix),
                dpi=300,
                format='svg',
                transparent=True,
                bbox_inches='tight',
            )
        plt.show()
        print('NRMSE tr is {}'.format(NRMSE_tr))
        print('NRMSE ts is {}'.format(NRMSE_ts))
    else:
        return (NRMSE_tr, NRMSE_ts)


def MG_Expr_read_in():
    device_code = ['4u', '6u', '7u', '8u', '9u', '11u', '14d', '15d', '16d']
    for i_mask in range(1, 4):
        path = './Data/MG readin/mask {}'.format(i_mask)
        os.makedirs(path, exist_ok=True)
        for i_device in range(len(device_code)):
            i_device_state = np.zeros((1600, 10))
            for i_segment in range(0, 20):
                mask_choice = '.\\Data\\MG\\Exp\\TiOx/5um_mask{}'.format(i_mask)
                file = mask_choice + '/results/5um_{}/5um_{}_Mask{}Seg{}.csv'.format(
                    device_code[i_device],
                    device_code[i_device],
                    i_mask,
                    i_segment,
                )
                df = pd.read_csv(file, header=None, sep='\n')
                df = df[0].str.split(',', expand=True)
                df_0 = df.iloc[148:20148, 1:4]
                df_0_numpy = df_0.to_numpy()
                data = df_0_numpy.astype(np.float64)
                data_RC_one_device = -data[:, 2]
                down_sampling_ratio = int(len(data_RC_one_device) / (80 + 20) / 10)
                data_resampled = data_RC_one_device[15::down_sampling_ratio]
                data_reshaped = np.reshape(data_resampled, (80 + 20, 10))
                data_response = data_reshaped[5:-20 + 5]
                i_device_state[
                    i_segment * 80:(i_segment + 1) * 80,
                    :,
                ]= data_response[:, :]
            np.savetxt(
                path + '/MG device {}.csv'.format(device_code[i_device]),
                i_device_state,
                delimiter=',',
            )
    point_number = 80 * 20 + 20
    a = 0.2
    b = 0.1
    c = 10
    tau = 18
    tr_warmup_overlap = 5
    warm_up = max(int(1 * point_number), 1000)
    pred_shift = 1
    time_total = pred_shift + point_number + warm_up
    x_initial_len = int(tau)
    running_len = pred_shift + point_number + warm_up
    x_total_len = x_initial_len + running_len
    t = np.array([np.arange(0, time_total, 1)]).T
    x_record = np.zeros((x_total_len, 1)) + 0.01
    for i in range(running_len):
        x_record[i + x_initial_len, 0] = mackey_glass_func(
            1,
            x_record[i + x_initial_len - 1],
            x_record[i],
            a,
            b,
            c,
        )
    x_target = x_record[x_initial_len + warm_up + pred_shift:]
    x_target = x_target[tr_warmup_overlap:-20 + tr_warmup_overlap]
    np.savetxt('./Data/MG readin/MG target.csv', x_target, delimiter=',')


def color_select(i, n_groups=8):
    c0 = np.array([117, 185, 86]) / 255
    c1 = np.array([103, 149, 216]) / 255
    t = i / n_groups if n_groups > 1 else 0.0
    return c0 + t * (c1 - c0)


def MG_SRC_Expr(
    train_combo=(8, 0, 5),
    test_index=7,
    no_pic=True,
    direct_transfer=False,
    mask_choice=3,
    ts_mode='partition',
    output_suffix=None,
):
    """
    Experimental single-channel MG demo.

    ts_mode controls the non-direct-transfer training acquisition:
        'partition' / 'temporal_partition' / 'different_input' : main TS mode.
            Different source devices receive different contiguous MG training blocks.
        'paired' / 'same_input' : paired-input control mode.
            Each source device receives the same final training block.

    The default in this v3 file is the main temporal-partition mode.
    """
    if direct_transfer:
        train_combo = (8,)
    ts_mode_norm = str(ts_mode).lower().replace('-', '_')
    if ts_mode_norm in (
        'main',
        'mode 1',
        'mode i',
        '1',
        'i',
        'partition',
        'temporal_partition',
        'different_input',
        'different',
    ):
        ts_mode_norm = 'partition'
    elif ts_mode_norm in (
        'paired',
        'mode 2',
        'mode ii',
        '2',
        'ii',
        'same_input',
        'same',
        'ts_equal',
        'pooled_equal',
    ):
        ts_mode_norm = 'paired'
    elif direct_transfer:
        ts_mode_norm = 'direct_transfer'
    else:
        raise ValueError('Unknown ts_mode: {}'.format(ts_mode))
    test_device = test_index
    combination = train_combo
    N = len(combination)
    rep_len = int(720 / N)
    device_code = ['4u', '6u', '7u', '8u', '9u', '11u', '14d', '15d', '16d']
    device_character_I = [153.2, 147.7, 154.1, 155.5, 154.9, 160.1, 154.0, 158.0, 130.8]
    color_sort_rank = [sorted(device_character_I).index(x) for x in device_character_I]
    file_target = './Data/MG readin/MG target.csv'
    df = pd.read_csv(file_target, header=None, sep='\n')
    full_target = df.to_numpy().astype(np.float64)
    RC_tr_storage = np.zeros((720, 10))
    target_tr = np.zeros((720, 1))
    for k_device in range(N):
        i_device = combination[k_device]
        file = './Data/MG readin/mask {}/MG device {}.csv'.format(
            mask_choice,
            device_code[i_device],
        )
        df = pd.read_csv(file, header=None, sep='\n')
        df = df[0].str.split(',', expand=True)
        df_n = df.to_numpy().astype(np.float64)
        if not direct_transfer and ts_mode_norm == 'partition':
            block_start = 800 - N * rep_len + k_device * rep_len
            block_stop = block_start + rep_len
        else:
            block_start = 800 - rep_len
            block_stop = 800
        select_slice = df_n[block_start:block_stop, :]
        RC_tr_storage[
            k_device * rep_len:(k_device + 1) * rep_len,
            :,
        ]= select_slice[:, :]
        target_tr[
            k_device * rep_len:(k_device + 1) * rep_len,
            :,
        ]= full_target[block_start:block_stop, :]
    file_test = './Data/MG readin/mask {}/MG device {}.csv'.format(
        mask_choice,
        device_code[test_device],
    )
    df = pd.read_csv(file_test, header=None, sep='\n')
    df = df[0].str.split(',', expand=True)
    df_n = df.to_numpy().astype(np.float64)
    RC_ts_storage = df_n[800:800 + 720, :]
    target_ts = full_target[800:800 + 720, :]
    ridge_alpha = 0
    lin = Ridge(alpha=ridge_alpha)
    lin.fit(RC_tr_storage, target_tr)
    output_tr = lin.predict(RC_tr_storage)
    output_ts = lin.predict(RC_ts_storage)
    NRMSE_tr, NRMSE_ts = (nrmse(target_tr, output_tr), nrmse(target_ts, output_ts))
    print('Current combination is {}'.format(combination))
    if not direct_transfer:
        print('TS acquisition mode is {}'.format(ts_mode_norm))
    print('Train NRMSE is {}'.format(NRMSE_tr))
    print('Test NRMSE is {}'.format(NRMSE_ts))
    if not no_pic:
        figure, ax = plt.subplots(2, 2, figsize=(2.4, 2), sharey='row', sharex='col')
        ax1, ax2, ax3, ax4 = (ax[1, 0], ax[1, 1], ax[0, 0], ax[0, 1])
        plt.rc('font', family='Arial', size=6)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['lines.linewidth'] = 1.2
        ylim_max = 0.035
        for i in range(N):
            color = color_select(color_sort_rank[combination[i]])
            if i != 0:
                ax1.axvline(rep_len * i, ls='--', color=np.array([180, 180, 180]) / 255)
                ax3.axvline(rep_len * i, ls='--', color=np.array([180, 180, 180]) / 255)
            ax1.plot(
                np.arange(rep_len * i, rep_len * (i + 1)),
                (output_tr[i * rep_len:(i + 1) * rep_len, 0] - target_tr[
                    i * rep_len:(i + 1) * rep_len,
                    0,
                ]) ** 2,
                label='Training Error',
                color=color,
            )
            ax3.plot(
                np.arange(i * rep_len, (i + 1) * rep_len),
                target_tr[i * rep_len:(i + 1) * rep_len, 0],
                color=np.array([200, 200, 200]) / 255,
            )
            ax3.plot(
                np.arange(i * rep_len, (i + 1) * rep_len),
                output_tr[i * rep_len:(i + 1) * rep_len, 0],
                color=color,
            )
        ax1.set_xlim(0, 720)
        ax1.set_ylim(0, ylim_max)
        ax3.set_ylim(0.2, 1.6)
        ax1.set_xlabel('Time step', fontdict={'family': 'arial', 'size': 6}, labelpad=1)
        ax1.set_ylabel('Squared error', fontdict={'family': 'arial', 'size': 6})
        ax3.set_ylabel('$x$', fontdict={'family': 'arial', 'size': 6})
        ax1.tick_params(axis='both', direction='in', labelsize=6)
        ax3.tick_params(axis='both', direction='in', labelsize=6)
        ax1.set_xticks([rep_len * j for j in range(N + 1)])
        ax1.set_yticks([0, 0.01, 0.02, 0.03])
        color_test = color_select(color_sort_rank[test_device])
        ax2.plot(
            np.arange(720, 1440),
            (output_ts[:, 0] - target_ts[:, 0]) ** 2,
            label='Testing Error',
            color=color_test,
        )
        ax4.plot(
            np.arange(720, 1440),
            target_ts[:, 0],
            color=np.array([200, 200, 200]) / 255,
        )
        ax4.plot(np.arange(720, 1440), output_ts[:, 0], color=color_test)
        ax2.set_xlim(720, 1440)
        ax2.set_ylim(0, ylim_max)
        ax2.set_xticks([rep_len * j for j in range(N + 1, 2 * N + 1)])
        ax2.tick_params(axis='both', direction='in', labelsize=6)
        ax4.tick_params(axis='both', direction='in', labelsize=6)
        figure.subplots_adjust(wspace=0, hspace=0.1)
        if output_suffix is None:
            if direct_transfer:
                output_suffix_eff = ''
            elif ts_mode_norm == 'partition':
                output_suffix_eff = '_TP_main'
            else:
                output_suffix_eff = '_paired_control'
        else:
            output_suffix_eff = str(output_suffix)
        if not direct_transfer:
            plt.savefig(
                './Figure/MG/Exp_Error_SRC_SC{}.svg'.format(output_suffix_eff),
                dpi=300,
                format='svg',
                transparent=True,
                bbox_inches='tight',
            )
        else:
            plt.savefig(
                './Figure/MG/Exp_Error_DT_SC{}.svg'.format(output_suffix_eff),
                dpi=300,
                format='svg',
                transparent=True,
                bbox_inches='tight',
            )
        plt.show()
    else:
        devices_in_set = combination
        device_character_list = []
        for i in devices_in_set:
            device_character_list.append(device_character_I[i])
        device_position = device_character_I[test_device]
        _max = max(device_character_list)
        _min = min(device_character_list)
        if device_position < _min or device_position > _max:
            q = 1
        else:
            q = 0
        return (NRMSE_tr, NRMSE_ts, q)


def _mg_mc_read_segment(
    device_code,
    mask_choice,
    segment_index,
    each_length=80,
    overlap=20,
    out_dim=10,
    down_sampling_start=15,
    tr_warmup_overlap=5,
):
    """Read one measured MG segment and return its 80 x 10 reservoir-state matrix."""
    candidate_roots = [
        os.path.join('.', 'Data', 'TiOx Exp', 'MG', '5um_mask{}'.format(mask_choice)),
        os.path.join(
            '.',
            'Data',
            'MG',
            'Exp',
            'TiOx',
            '5um_mask{}'.format(mask_choice),
        ),
        '.\\Data\\TiOx Exp\\MG/5um_mask{}'.format(mask_choice),
    ]
    root = next((p for p in candidate_roots if os.path.exists(p)), candidate_roots[0])
    file = os.path.join(
        root,
        'results',
        '5um_{}'.format(device_code),
        '5um_{}_Mask{}Seg{}.csv'.format(device_code, mask_choice, segment_index),
    )
    df = pd.read_csv(file, header=None, sep='\n')
    df = df[0].str.split(',', expand=True)
    data = df.iloc[148:20148, 1:4].to_numpy().astype(np.float64)
    current = -data[:, 2]
    down_sampling_ratio = int(len(current) / (each_length + overlap) / out_dim)
    data_resampled = current[down_sampling_start::down_sampling_ratio]
    data_reshaped = np.reshape(data_resampled, (each_length + overlap, out_dim))
    return data_reshaped[tr_warmup_overlap:-overlap + tr_warmup_overlap, :]


def _mg_mc_load_device_set(
    device_set,
    serial_order,
    each_length=80,
    out_dim=10,
    down_sampling_start=15,
    overlap=20,
    tr_warmup_overlap=5,
):
    """Assemble one three-channel device set over the requested measured segments."""
    device_set = tuple(device_set)
    serial_order = tuple((int(x) for x in serial_order))
    if len(device_set) != 3:
        raise ValueError(
            'A multi-channel MG device set must contain exactly three devices.'
        )
    state = np.zeros((len(serial_order) * each_length, len(device_set) * out_dim))
    for i_seg, segment_index in enumerate(serial_order):
        row0, row1 = (i_seg * each_length, (i_seg + 1) * each_length)
        for channel, device_code in enumerate(device_set):
            col0, col1 = (channel * out_dim, (channel + 1) * out_dim)
            state[row0:row1, col0:col1] = _mg_mc_read_segment(
                device_code=device_code,
                mask_choice=channel + 1,
                segment_index=segment_index,
                each_length=each_length,
                overlap=overlap,
                out_dim=out_dim,
                down_sampling_start=down_sampling_start,
                tr_warmup_overlap=tr_warmup_overlap,
            )
    return state


def _mg_mc_select_ridge_alpha(
    X,
    y,
    alpha_grid=(
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
    ),
    n_folds=5,
):
    """Select Ridge alpha using contiguous blocked CV on source training data only."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    indices = np.arange(len(X))
    val_blocks = [np.asarray(v, dtype=int) for v in np.array_split(
        indices,
        int(n_folds),
    ) if len(v)]
    rows = []
    for alpha in tuple((float(a) for a in alpha_grid)):
        scores = []
        for val_idx in val_blocks:
            train_idx = np.setdiff1d(indices, val_idx, assume_unique=True)
            if len(train_idx) == 0:
                continue
            try:
                model = Ridge(alpha=alpha)
                model.fit(X[train_idx], y[train_idx])
                scores.append(float(nrmse(y[val_idx], model.predict(X[val_idx]))))
            except Exception:
                scores.append(np.inf)
        rows.append({
            'alpha': alpha,
            'mean_cv_nrmse': float(np.mean(scores)) if scores else np.inf,
            'std_cv_nrmse': float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        })
    finite = [row for row in rows if np.isfinite(row['mean_cv_nrmse'])]
    if not finite:
        raise RuntimeError('All Ridge-CV alpha candidates failed.')
    best = min(finite, key=lambda row: row['mean_cv_nrmse'])
    return (float(best['alpha']), pd.DataFrame(rows))


def MG_SRC_Expr_MultiChannel(
    direct_transfer=False,
    AVG=False,
    spatial_reorder=False,
    temporal_reorder=False,
    ts_mode='partition',
    output_suffix=None,
    baseline_compare=False,
    Ens=False,
    RCV=False,
    ENS=None,
    ridge_cv_alphas=(
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
    ),
    ridge_cv_folds=5,
):
    """
    Experimental three-channel MG demo.

    Modes
    -----
    direct_transfer=True
        Classical single-source transfer. The first SRC source set is used.

    direct_transfer=False, AVG=False, Ens=False, RCV=False
        TS training. ``mode I`` is temporal partition and ``mode II`` is the
        paired-input control.

    AVG=True
        Source-state average. The same three multi-channel source sets used by
        SRC are evaluated on the same complete 720-step training trajectory;
        their 30-dimensional states are averaged before fitting one readout.

    Ens=True (or ENS=True)
        Output ensemble. Three independent alpha=0 readouts are trained using
        the same three multi-channel source sets and complete training
        trajectory used by the AVG control. Their target-test outputs are
        averaged.

    RCV=True
        Single-source Ridge-CV. Only the first SRC source set is used.
        Alpha is selected by contiguous blocked CV using source training data
        only, and the selected readout is then evaluated on the target set.
    """
    if ENS is not None:
        if Ens and bool(ENS) != bool(Ens):
            raise ValueError('Ens and ENS specify conflicting values.')
        Ens = bool(ENS)
    baseline_flags = int(bool(AVG)) + int(bool(Ens)) + int(bool(RCV))
    if baseline_flags > 1:
        raise ValueError('AVG, Ens/ENS and RCV are mutually exclusive.')
    if direct_transfer and baseline_flags:
        raise ValueError('direct_transfer cannot be combined with AVG, Ens/ENS or RCV.')
    ts_mode_norm = str(ts_mode).lower().replace('-', '_').strip()
    if ts_mode_norm in (
        'main',
        'mode i',
        'mode_i',
        'i',
        '1',
        'partition',
        'temporal_partition',
        'different_input',
        'different',
    ):
        ts_mode_norm = 'partition'
    elif ts_mode_norm in (
        'paired',
        'mode ii',
        'mode_ii',
        'ii',
        '2',
        'same_input',
        'same',
        'ts_equal',
        'pooled_equal',
    ):
        ts_mode_norm = 'paired'
    elif direct_transfer or AVG or Ens or RCV:
        ts_mode_norm = 'direct_transfer' if direct_transfer else 'baseline'
    else:
        raise ValueError('Unknown ts_mode: {}'.format(ts_mode))
    each_length = 80
    out_dim = 10
    num_parallel = 3
    down_sampling_start = 15
    tr_warmup_overlap = 5
    overlap = 20
    a, b, c, tau, dt = (0.2, 0.1, 10, 18, 1)
    point_number = each_length * 20 + overlap
    warm_up = max(int(point_number), 1000)
    pred_shift = 1
    running_len = pred_shift + point_number + warm_up
    x_initial_len = int(tau / dt)
    x_record = np.zeros((x_initial_len + running_len, 1)) + 0.01
    for i in range(running_len):
        x_record[i + x_initial_len, 0] = mackey_glass_func(
            dt,
            x_record[i + x_initial_len - 1],
            x_record[i],
            a,
            b,
            c,
        )
    x_target = x_record[x_initial_len + warm_up + pred_shift:]
    x_target = np.asarray(
        x_target[tr_warmup_overlap:-overlap + tr_warmup_overlap],
        dtype=float,
    ).reshape(-1, 1)
    x_target_tr = x_target[:int(len(x_target) * 0.5)]
    x_target_ts = x_target[int(len(x_target) * 0.5):]
    target_tr_full = x_target_tr[-720:]
    target_ts = x_target_ts[:720]
    if baseline_compare:
        source_sets = (('14d', '15d', '6u'), ('15d', '15d', '6u'), ('6u', '16d', '14d'))
        target_set = ('7u', '16d', '15d')
    else:
        source_sets = (('14d', '8u', '14d'), ('16d', '7u', '4u'), ('9u', '15d', '11u'))
        target_set = ('16d', '11u', '11u')
    train_serial_full = tuple(range(1, 10))
    test_serial_order = tuple(range(10, 19))
    if spatial_reorder and direct_transfer:
        source_sets = (('11u', '4u', '16d'),) + source_sets[1:]
        target_set = ('16d', '4u', '11u')
    RC_ts_storage = _mg_mc_load_device_set(
        target_set,
        test_serial_order,
        each_length=each_length,
        out_dim=out_dim,
        down_sampling_start=down_sampling_start,
        overlap=overlap,
        tr_warmup_overlap=tr_warmup_overlap,
    )
    selected_alpha = 0.0
    cv_table = None
    ensemble_components_tr = None
    ensemble_components_ts = None
    if direct_transfer:
        method_name = 'Classical'
        RC_tr_storage = _mg_mc_load_device_set(
            source_sets[0],
            train_serial_full,
            each_length,
            out_dim,
            down_sampling_start,
            overlap,
            tr_warmup_overlap,
        )
        target_tr = target_tr_full
        model = Ridge(alpha=0.0)
        model.fit(RC_tr_storage, target_tr)
        x_bar_tr = np.asarray(model.predict(RC_tr_storage), dtype=float).reshape(-1, 1)
        x_bar_ts = np.asarray(model.predict(RC_ts_storage), dtype=float).reshape(-1, 1)
        train_nrmse_value = float(nrmse(target_tr, x_bar_tr))
    elif AVG:
        method_name = 'State average'
        source_states = [
            _mg_mc_load_device_set(
                device_set,
                train_serial_full,
                each_length,
                out_dim,
                down_sampling_start,
                overlap,
                tr_warmup_overlap,
            )
            for device_set in source_sets
        ]
        RC_tr_storage = np.mean(np.stack(source_states, axis=0), axis=0)
        target_tr = target_tr_full
        model = Ridge(alpha=0.0)
        model.fit(RC_tr_storage, target_tr)
        x_bar_tr = np.asarray(model.predict(RC_tr_storage), dtype=float).reshape(-1, 1)
        x_bar_ts = np.asarray(model.predict(RC_ts_storage), dtype=float).reshape(-1, 1)
        train_nrmse_value = float(nrmse(target_tr, x_bar_tr))
    elif Ens:
        method_name = 'Output ensemble'
        source_states = [
            _mg_mc_load_device_set(
                device_set,
                train_serial_full,
                each_length,
                out_dim,
                down_sampling_start,
                overlap,
                tr_warmup_overlap,
            )
            for device_set in source_sets
        ]
        target_tr = target_tr_full
        train_outputs = []
        test_outputs = []
        train_scores = []
        for X_source in source_states:
            model = Ridge(alpha=0.0)
            model.fit(X_source, target_tr)
            pred_tr = np.asarray(model.predict(X_source), dtype=float).reshape(-1, 1)
            pred_ts = np.asarray(
                model.predict(RC_ts_storage),
                dtype=float,
            ).reshape(-1, 1)
            train_outputs.append(pred_tr)
            test_outputs.append(pred_ts)
            train_scores.append(float(nrmse(target_tr, pred_tr)))
        ensemble_components_tr = np.stack(train_outputs, axis=0)
        ensemble_components_ts = np.stack(test_outputs, axis=0)
        x_bar_tr = np.mean(ensemble_components_tr, axis=0)
        x_bar_ts = np.mean(ensemble_components_ts, axis=0)
        RC_tr_storage = source_states[0]
        train_nrmse_value = float(np.mean(train_scores))
    elif RCV:
        method_name = 'Ridge CV'
        RC_tr_storage = _mg_mc_load_device_set(
            source_sets[0],
            train_serial_full,
            each_length,
            out_dim,
            down_sampling_start,
            overlap,
            tr_warmup_overlap,
        )
        target_tr = target_tr_full
        selected_alpha, cv_table = _mg_mc_select_ridge_alpha(
            RC_tr_storage,
            target_tr,
            alpha_grid=ridge_cv_alphas,
            n_folds=ridge_cv_folds,
        )
        model = Ridge(alpha=selected_alpha)
        model.fit(RC_tr_storage, target_tr)
        x_bar_tr = np.asarray(model.predict(RC_tr_storage), dtype=float).reshape(-1, 1)
        x_bar_ts = np.asarray(model.predict(RC_ts_storage), dtype=float).reshape(-1, 1)
        train_nrmse_value = float(nrmse(target_tr, x_bar_tr))
        cv_dir = './Data/MG/Exp/TiOx'
        os.makedirs(cv_dir, exist_ok=True)
        cv_table.to_csv(
            os.path.join(cv_dir, 'MG_MC_RidgeCV_alpha_selection.csv'),
            index=False,
        )
    else:
        method_name = 'TS {}'.format(ts_mode_norm)
        ordered_sources = source_sets[::-1] if temporal_reorder else source_sets
        RC_tr_storage = np.zeros((720, num_parallel * out_dim))
        if ts_mode_norm == 'partition':
            target_tr = target_tr_full
            for source_id, device_set in enumerate(ordered_sources):
                serial_order = tuple(range(1 + 3 * source_id, 4 + 3 * source_id))
                X_block = _mg_mc_load_device_set(
                    device_set,
                    serial_order,
                    each_length,
                    out_dim,
                    down_sampling_start,
                    overlap,
                    tr_warmup_overlap,
                )
                row0, row1 = (source_id * 240, (source_id + 1) * 240)
                RC_tr_storage[row0:row1, :] = X_block
        else:
            target_tr = np.tile(x_target_tr[-240:], (3, 1))
            serial_order = (7, 8, 9)
            for source_id, device_set in enumerate(ordered_sources):
                X_block = _mg_mc_load_device_set(
                    device_set,
                    serial_order,
                    each_length,
                    out_dim,
                    down_sampling_start,
                    overlap,
                    tr_warmup_overlap,
                )
                row0, row1 = (source_id * 240, (source_id + 1) * 240)
                RC_tr_storage[row0:row1, :] = X_block
        model = Ridge(alpha=0.0)
        model.fit(RC_tr_storage, target_tr)
        x_bar_tr = np.asarray(model.predict(RC_tr_storage), dtype=float).reshape(-1, 1)
        x_bar_ts = np.asarray(model.predict(RC_ts_storage), dtype=float).reshape(-1, 1)
        train_nrmse_value = float(nrmse(target_tr, x_bar_tr))
    test_nrmse_value = float(nrmse(target_ts, x_bar_ts))
    print('Method is {}'.format(method_name))
    print('Train NRMSE is {}'.format(train_nrmse_value))
    print('Test NRMSE is {}'.format(test_nrmse_value))
    if RCV:
        print('Ridge-CV selected alpha is {}'.format(selected_alpha))
    Target_tr = target_tr
    Output_tr = x_bar_tr
    Output_ts = x_bar_ts
    figure, ax = plt.subplots(2, 2, figsize=(2.4, 2), sharey='row', sharex='col')
    ax1, ax2, ax3, ax4 = (ax[1, 0], ax[1, 1], ax[0, 0], ax[0, 1])
    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1.2
    color4 = np.array([107, 158, 184]) / 255
    color3 = np.array([103, 149, 216]) / 255
    color2 = np.array([110, 167, 151]) / 255
    color1 = np.array([117, 185, 86]) / 255
    color_avg = np.array([138, 127, 214]) / 255
    color_ens = np.array([218, 69, 131]) / 255
    color_rcv = np.array([247, 183, 5]) / 255
    ylim_max = 0.1
    if direct_transfer:
        train_color = color1
        ax1.plot((Output_tr[:, 0] - Target_tr[:, 0]) ** 2, color=train_color)
        ax3.plot(Target_tr[:, 0], color=np.array([200, 200, 200]) / 255)
        ax3.plot(Output_tr[:, 0], color=train_color)
    elif AVG:
        for i in range(3):
            if i:
                ax1.axvline(240 * i, ls='--', color=np.array([180, 180, 180]) / 255)
                ax3.axvline(240 * i, ls='--', color=np.array([180, 180, 180]) / 255)
            sl = slice(i * 240, (i + 1) * 240)
            x = np.arange(i * 240, (i + 1) * 240)
            ax1.plot(x, (Output_tr[sl, 0] - Target_tr[sl, 0]) ** 2, color=color_avg)
            ax3.plot(x, Target_tr[sl, 0], color=np.array([200, 200, 200]) / 255)
            ax3.plot(x, Output_tr[sl, 0], color=color_avg)
    elif Ens or RCV:
        train_color = color_ens if Ens else color_rcv
        ax1.plot((Output_tr[:, 0] - Target_tr[:, 0]) ** 2, color=train_color)
        ax3.plot(Target_tr[:, 0], color=np.array([200, 200, 200]) / 255)
        ax3.plot(Output_tr[:, 0], color=train_color)
    else:
        colors = [color1, color2, color3]
        if temporal_reorder:
            colors = [color3, color2, color1]
        for i in range(3):
            if i:
                ax1.axvline(240 * i, ls='--', color=np.array([180, 180, 180]) / 255)
                ax3.axvline(240 * i, ls='--', color=np.array([180, 180, 180]) / 255)
            sl = slice(i * 240, (i + 1) * 240)
            x = np.arange(i * 240, (i + 1) * 240)
            ax1.plot(x, (Output_tr[sl, 0] - Target_tr[sl, 0]) ** 2, color=colors[i])
            ax3.plot(x, Target_tr[sl, 0], color=np.array([200, 200, 200]) / 255)
            ax3.plot(x, Output_tr[sl, 0], color=colors[i])
    ax1.set_xlim(0, 720)
    ax1.set_ylim(0, ylim_max)
    ax3.set_ylim(0.2, 1.6)
    ax1.set_ylabel('Squared error', fontdict={'family': 'arial', 'size': 6})
    ax3.set_ylabel('$x$', fontdict={'family': 'arial', 'size': 6})
    ax1.tick_params(axis='both', direction='in', labelsize=6)
    ax3.tick_params(axis='both', direction='in', labelsize=6)
    ax1.set_xticks([0, 240, 480, 720])
    ax1.set_yticks([0, 0.03, 0.06, 0.09])
    ax2.plot(
        np.arange(720, 1440),
        (Output_ts[:, 0] - target_ts[:, 0]) ** 2,
        color=color4,
    )
    ax4.plot(
        np.arange(720, 1440),
        target_ts[:, 0],
        color=np.array([200, 200, 200]) / 255,
    )
    ax4.plot(np.arange(720, 1440), Output_ts[:, 0], color=color4)
    ax2.set_xlim(720, 1440)
    ax2.set_ylim(0, ylim_max)
    ax2.set_xticks([960, 1200, 1440])
    ax2.tick_params(axis='both', direction='in', labelsize=6)
    ax4.tick_params(axis='both', direction='in', labelsize=6)
    figure.subplots_adjust(wspace=0, hspace=0.1, bottom=0.18)
    figure.supxlabel('Time step', x=0.5, y=0.04, fontname='Arial', fontsize=6)
    if temporal_reorder:
        suffix = '_TimePermut'
    elif spatial_reorder:
        suffix = '_SpacePermut'
    elif AVG:
        suffix = '_AVG'
    elif Ens:
        suffix = '_Ens'
    elif RCV:
        suffix = '_RCV'
    else:
        suffix = ''
    if output_suffix is None and (not direct_transfer) and (baseline_flags == 0):
        suffix += '_TP_main' if ts_mode_norm == 'partition' else '_paired_control'
    elif output_suffix is not None:
        suffix += str(output_suffix)
    if not direct_transfer:
        save_path = './Figure/MG/Exp_Error_SRC_MC{}.svg'.format(suffix)
    else:
        save_path = './Figure/MG/Exp_Error_DT_MC{}.svg'.format(suffix)
    plt.savefig(save_path, dpi=300, format='svg', transparent=True, bbox_inches='tight')
    plt.show()
    return {
        'method': method_name,
        'train_nrmse': train_nrmse_value,
        'test_nrmse': test_nrmse_value,
        'selected_alpha': float(selected_alpha),
        'source_sets': source_sets,
        'target_set': target_set,
        'n_train_rows': int(len(Target_tr)),
        'n_features': int(RC_ts_storage.shape[1]),
        'figure_path': save_path,
        'cv_table': cv_table,
        'ensemble_train_components': ensemble_components_tr,
        'ensemble_test_components': ensemble_components_ts,
    }


if __name__ == '__main__':
    mpl.rcParams.update({
        'font.family': 'Arial',
        'font.size': 6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.titlesize': 6,
        'axes.labelsize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    modes = ['Sim', 'Exp']
    for mode in modes:
        data_dir = './Data/MG/{}/TiOx'.format(mode)
        if not os.path.exists(data_dir):
            print('Creating new data file directory...')
            os.makedirs(data_dir)
    fig_dir = './Figure/MG'
    if not os.path.exists(fig_dir):
        print('Creating new figure file directory...')
        os.makedirs(fig_dir)
    # Read in the data to create state data to accelerate RC processing
    MG_Expr_read_in()
    # # Single channel, experiment
    # # Classical framework
    MG_SRC_Expr(direct_transfer=True, no_pic=False)
    # # TS framework
    MG_SRC_Expr(direct_transfer=False, no_pic=False, ts_mode='mode I')
    MG_SRC_Expr(direct_transfer=False, no_pic=False, ts_mode='mode II')
    #
    # # Single channel, simulation
    # # Classical framework
    MG_SRC_sim(direct_transfer=True)
    # # TS framework
    MG_SRC_sim(direct_transfer=False)
    #
    # # Multichannel, experiment
    # # Classical framework
    MG_SRC_Expr_MultiChannel(direct_transfer=True)
    # # TS framework
    MG_SRC_Expr_MultiChannel(direct_transfer=False, ts_mode='mode I')
    MG_SRC_Expr_MultiChannel(direct_transfer=False, ts_mode='mode II')
    #
    # # Other demos
    # Temporal re-ordering for mode I
    MG_SRC_Expr_MultiChannel(
        direct_transfer=False,
        temporal_reorder=True,
        ts_mode='mode I',
    )
    # State-average
    MG_SRC_Expr_MultiChannel(direct_transfer=False, AVG=True, baseline_compare=True)
    # Ensemble method
    MG_SRC_Expr_MultiChannel(direct_transfer=False, Ens=True, baseline_compare=True)
    # Ridge CV
    MG_SRC_Expr_MultiChannel(direct_transfer=False, RCV=True, baseline_compare=True)
    # Comparison with TS framework
    MG_SRC_Expr_MultiChannel(direct_transfer=False, baseline_compare=True)



# # Below are the functions for the 1000 trials processing
import json
import os
from dataclasses import dataclass, field
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
from sklearn.linear_model import Ridge
MG_DEVICE_CODE = (
    '4u',
    '6u',
    '7u',
    '8u',
    '9u',
    '11u',
    '14d',
    '15d',
    '16d',
)
MG_DEVICE_IDS = tuple(range(9))
MG_DEFAULT_TRAIN_START = 80
MG_DEFAULT_TRAIN_END = 800
MG_DEFAULT_TEST_START = 800
MG_DEFAULT_TEST_LEN = 720
MG_DEFAULT_TOTAL_TRAIN_LEN = 720
MG_FEWSHOT_STOP = 159
DEFAULT_ALPHA = 1e-16
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


def _tiox1000_mg_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    scale = float(np.std(y_true))
    return rmse if scale == 0 else rmse / scale


def cache_paths(data_root: Union[str, Path]='./Data/MG readin') -> Tuple[
    Path,
    Path,
]:
    root = Path(data_root)
    cache = root / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    return (
        cache / 'mg_statebank_masks123_float64_v1.npy',
        cache / 'mg_target_float64_v1.npy',
    )


def load_state_bank(
    data_root: Union[str, Path]='./Data/MG readin',
    rebuild: bool=False,
    verbose: bool=True,
) -> Tuple[np.ndarray, np.ndarray]:
    root = Path(data_root)
    bank_path, target_path = cache_paths(root)
    expected = (3, len(MG_DEVICE_CODE), 1600, 10)
    if bank_path.exists() and target_path.exists() and (not rebuild):
        bank = np.load(bank_path, mmap_mode='r')
        target = np.load(target_path, mmap_mode='r')
        if bank.shape != expected:
            raise ValueError(
                f'MG state-bank shape {bank.shape} != {expected}; rebuild it'
            )
        return (bank, target)
    target_csv = root / 'MG target.csv'
    if not target_csv.exists():
        raise FileNotFoundError(
            f'Cannot find {target_csv}; generate MG read-in data first'
        )
    target = pd.read_csv(target_csv, header=None).to_numpy(dtype=np.float64)
    if target.ndim == 1:
        target = target[:, None]
    np.save(target_path, target)
    tmp = Path(str(bank_path) + '.tmp.npy')
    tmp.unlink(missing_ok=True)
    bank = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float64, shape=expected)
    count = 0
    total = expected[0] * expected[1]
    try:
        for mask_index, mask in enumerate((1, 2, 3)):
            for device, code in enumerate(MG_DEVICE_CODE):
                path = root / f'mask {mask}' / f'MG device {code}.csv'
                if not path.exists():
                    raise FileNotFoundError(
                        f'Cannot find {path}; run MG_Expr_read_in first'
                    )
                X = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
                if X.shape != expected[2:]:
                    raise ValueError(f'{path} shape {X.shape} != {expected[2:]}')
                bank[mask_index, device] = X
                count += 1
                if verbose:
                    print(f'[MG state bank] {count}/{total}')
        bank.flush()
    except Exception:
        del bank
        tmp.unlink(missing_ok=True)
        raise
    del bank
    os.replace(tmp, bank_path)
    return (np.load(bank_path, mmap_mode='r'), np.load(target_path, mmap_mode='r'))


def set_states(
    bank: np.ndarray,
    device_set: Sequence[int],
    channel_masks: Sequence[int]=(1, 2, 3),
) -> np.ndarray:
    if len(device_set) != len(channel_masks):
        raise ValueError('device_set and channel_masks must have equal length')
    channel_states = [
        bank[
            int(mask) - 1,
            int(device),
        ]
        for device, mask in zip(device_set, channel_masks)
    ]
    return np.concatenate(channel_states, axis=1)


def fit_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float=DEFAULT_ALPHA,
    return_model: bool=False,
) -> Dict:
    model = Ridge(alpha=float(alpha), solver='cholesky').fit(X_train, y_train)
    out = {
        'train_nrmse': _tiox1000_mg_nrmse(y_train, model.predict(X_train)),
        'test_nrmse': _tiox1000_mg_nrmse(y_test, model.predict(X_test)),
        'selected_alpha': float(alpha),
        'n_train_rows': int(X_train.shape[0]),
        'n_features': int(X_train.shape[1]),
    }
    if return_model:
        out['model'] = model
    return out


def select_ridge_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alpha_grid: Sequence[float]=RIDGE_CV_ALPHAS,
    n_folds: int=RIDGE_CV_FOLDS,
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
                scores.append(_tiox1000_mg_nrmse(y[va], model.predict(X[va])))
            except Exception:
                scores.append(np.inf)
        rows.append({
            'alpha': alpha,
            'mean_cv_nrmse': float(np.mean(scores)),
            'std_cv_nrmse': float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        })
    finite = [r for r in rows if np.isfinite(r['mean_cv_nrmse'])]
    if not finite:
        raise RuntimeError('All MG Ridge-CV candidates failed')
    return (float(min(finite, key=lambda r: r['mean_cv_nrmse'])['alpha']), rows)

@dataclass


class MGEvaluator:
    bank: np.ndarray
    target: np.ndarray
    channel_masks: Tuple[int, int, int] = (1, 2, 3)
    alpha: float = DEFAULT_ALPHA
    set_cache: Dict[Tuple[int, ...], np.ndarray] = field(default_factory=dict)
    fixed_models: Dict[Tuple[int, ...], Dict] = field(default_factory=dict)
    ridge_models: Dict[Tuple[int, ...], Dict] = field(default_factory=dict)

    def X(self, device_set: Sequence[int]) -> np.ndarray:
        key = tuple(map(int, device_set))
        if key not in self.set_cache:
            self.set_cache[key] = set_states(self.bank, key, self.channel_masks)
        return self.set_cache[key]

    @property
    def y_train(self) -> np.ndarray:
        return self.target[MG_DEFAULT_TRAIN_START:MG_DEFAULT_TRAIN_END]

    @property
    def y_test(self) -> np.ndarray:
        return self.target[
            MG_DEFAULT_TEST_START :
            MG_DEFAULT_TEST_START + MG_DEFAULT_TEST_LEN
        ]

    def fixed_source_model(self, device_set: Sequence[int]) -> Dict:
        key = tuple(map(int, device_set))
        if key not in self.fixed_models:
            X = self.X(key)[MG_DEFAULT_TRAIN_START:MG_DEFAULT_TRAIN_END]
            model = Ridge(
                alpha=float(self.alpha),
                solver='cholesky',
            ).fit(X, self.y_train)
            self.fixed_models[key] = {
                'model': model,
                'train_nrmse': _tiox1000_mg_nrmse(self.y_train, model.predict(X)),
                'selected_alpha': float(self.alpha),
            }
        return self.fixed_models[key]

    def ridge_source_model(
        self,
        device_set: Sequence[int],
        alpha_grid=RIDGE_CV_ALPHAS,
        n_folds: int=RIDGE_CV_FOLDS,
    ) -> Dict:
        key = tuple(map(int, device_set))
        if key not in self.ridge_models:
            X = self.X(key)[MG_DEFAULT_TRAIN_START:MG_DEFAULT_TRAIN_END]
            alpha, cv = select_ridge_alpha(X, self.y_train, alpha_grid, n_folds)
            model = Ridge(alpha=alpha, solver='cholesky').fit(X, self.y_train)
            self.ridge_models[key] = {
                'model': model,
                'train_nrmse': _tiox1000_mg_nrmse(self.y_train, model.predict(X)),
                'selected_alpha': alpha,
                'cv': cv,
            }
        return self.ridge_models[key]

    def model_on_target(
        self,
        item: Dict,
        target_set: Sequence[int],
    ) -> Dict:
        X_test = self.X(
            target_set,
        )[MG_DEFAULT_TEST_START:MG_DEFAULT_TEST_START + MG_DEFAULT_TEST_LEN]
        return {
            'train_nrmse': float(item['train_nrmse']),
            'test_nrmse': _tiox1000_mg_nrmse(
                self.y_test,
                item['model'].predict(X_test),
            ),
            'selected_alpha': float(item['selected_alpha']),
            'n_train_rows': MG_DEFAULT_TOTAL_TRAIN_LEN,
            'n_features': int(X_test.shape[1]),
        }

    def ts(
        self,
        combo: Sequence[Sequence[int]],
        target_set: Sequence[int],
    ) -> Dict:
        combo = tuple((tuple(map(int, s)) for s in combo))
        N = len(combo)
        base, rem = divmod(MG_DEFAULT_TOTAL_TRAIN_LEN, N)
        cursor = MG_DEFAULT_TRAIN_END - MG_DEFAULT_TOTAL_TRAIN_LEN
        X_blocks = []
        for i, source in enumerate(combo):
            size = base + (1 if i < rem else 0)
            X_blocks.append(self.X(source)[cursor:cursor + size])
            cursor += size
        X_train = np.vstack(X_blocks)
        X_test = self.X(
            target_set,
        )[MG_DEFAULT_TEST_START:MG_DEFAULT_TEST_START + MG_DEFAULT_TEST_LEN]
        return fit_eval(X_train, self.y_train, X_test, self.y_test, self.alpha)

    def ensemble(
        self,
        combo: Sequence[Sequence[int]],
        target_set: Sequence[int],
    ) -> Dict:
        combo = tuple((tuple(map(int, s)) for s in combo))
        X_test = self.X(
            target_set,
        )[MG_DEFAULT_TEST_START:MG_DEFAULT_TEST_START + MG_DEFAULT_TEST_LEN]
        preds, train_scores = ([], [])
        for source in combo:
            item = self.fixed_source_model(source)
            preds.append(item['model'].predict(X_test))
            train_scores.append(item['train_nrmse'])
        return {
            'train_nrmse': float(np.mean(train_scores)),
            'test_nrmse': _tiox1000_mg_nrmse(
                self.y_test,
                np.mean(np.stack(preds), axis=0),
            ),
            'selected_alpha': float(self.alpha),
            'n_train_rows': int(len(combo) * MG_DEFAULT_TOTAL_TRAIN_LEN),
            'n_features': int(X_test.shape[1]),
        }

    def state_average(
        self,
        combo: Sequence[Sequence[int]],
        target_set: Sequence[int],
    ) -> Dict:
        combo = tuple((tuple(map(int, s)) for s in combo))
        X_train = np.mean(
            np.stack([
                self.X(source_set)[
                    MG_DEFAULT_TRAIN_START : MG_DEFAULT_TRAIN_END
                ]
                for source_set in combo
            ]),
            axis=0,
        )
        X_test = self.X(
            target_set,
        )[MG_DEFAULT_TEST_START:MG_DEFAULT_TEST_START + MG_DEFAULT_TEST_LEN]
        return fit_eval(X_train, self.y_train, X_test, self.y_test, self.alpha)

    def evaluate_family(
        self,
        combos_by_n: Dict[int, Sequence[Sequence[int]]],
        target_set: Sequence[int],
        ridge_grid=RIDGE_CV_ALPHAS,
        ridge_folds: int=RIDGE_CV_FOLDS,
    ) -> List[Dict]:
        rows: List[Dict] = []
        for N in (2, 3, 4, 5):
            combo = combos_by_n[N]
            for method, result in (
                ('TS', self.ts(combo, target_set)),
                ('ensemble', self.ensemble(combo, target_set)),
                ('state-average', self.state_average(combo, target_set)),
            ):
                rows.append({
                    'n_source_sets': N,
                    'method': method,
                    **result,
                })
        first = tuple(map(int, combos_by_n[3][0]))
        target_set = tuple(map(int, target_set))
        rows.append({
            'n_source_sets': 1,
            'reference_N_source_sets': 3,
            'method': 'classical',
            **self.model_on_target(self.fixed_source_model(first), target_set),
        })
        ridge_item = self.ridge_source_model(first, ridge_grid, ridge_folds)
        rows.append({
            'n_source_sets': 1,
            'reference_N_source_sets': 3,
            'method': 'ridge-CV',
            **self.model_on_target(ridge_item, target_set),
            'cv_table_json': json.dumps(ridge_item['cv']),
        })
        rows.append({
            'n_source_sets': 1,
            'reference_N_source_sets': 3,
            'method': 'self-training',
            **self.model_on_target(self.fixed_source_model(target_set), target_set),
        })
        return rows

    def fewshot(
        self,
        source_combo: Sequence[Sequence[int]],
        target_set: Sequence[int],
        shots: Sequence[int],
        stop: int=MG_FEWSHOT_STOP,
    ) -> List[Dict]:
        combo = tuple((tuple(map(int, s)) for s in source_combo))
        target_set = tuple(map(int, target_set))
        rows = [{
            'method': 'TS-zeroshot',
            'fewshot': 0,
            **self.ts(combo, target_set),
            'fewshot_start': np.nan,
            'fewshot_stop': int(stop),
        }]
        N = len(combo)
        base, rem = divmod(MG_DEFAULT_TOTAL_TRAIN_LEN, N)
        cursor = MG_DEFAULT_TRAIN_END - MG_DEFAULT_TOTAL_TRAIN_LEN
        ts_blocks = []
        for i, source in enumerate(combo):
            size = base + (1 if i < rem else 0)
            ts_blocks.append(self.X(source)[cursor:cursor + size])
            cursor += size
        X_ts = np.vstack(ts_blocks)
        X_single = self.X(combo[0])[MG_DEFAULT_TRAIN_START:MG_DEFAULT_TRAIN_END]
        X_target = self.X(target_set)
        X_test = X_target[
            MG_DEFAULT_TEST_START :
            MG_DEFAULT_TEST_START + MG_DEFAULT_TEST_LEN
        ]
        for k in map(int, shots):
            start = int(stop) - k
            X_fs = X_target[start:int(stop)]
            y_fs = self.target[start:int(stop)]
            y_comb = np.vstack([self.y_train, y_fs])
            rows.append({
                'method': 'TS-fewshot',
                'fewshot': k,
                'fewshot_start': start,
                'fewshot_stop': int(stop),
                **fit_eval(
                    np.vstack([X_ts, X_fs]),
                    y_comb,
                    X_test,
                    self.y_test,
                    self.alpha,
                ),
            })
            rows.append({
                'method': 'classical-fewshot',
                'fewshot': k,
                'fewshot_start': start,
                'fewshot_stop': int(stop),
                **fit_eval(
                    np.vstack([X_single, X_fs]),
                    y_comb,
                    X_test,
                    self.y_test,
                    self.alpha,
                ),
            })
        return rows
