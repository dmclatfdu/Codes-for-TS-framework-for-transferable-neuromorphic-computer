
"""
Train and evaluate reservoir computing in classical framework and in temporal-switch (TS) framework.
This is the file for all task demo based on Mackey-Glass prediction task.

Including:
    1. TS framework demonstration and comparison with classical method (Figs. 3, S11 & S12).
    2. Performance robustness investigation (Figs. 4, S15 & S16)
    3. Other demos (temporal permutation:      Fig. S23;
                    comparison with averaging: Fig. S24;
                    spatial permutation:       Fig. S25)

Zefeng Zhang, Research Institute of Intelligent Complex Systems, Fudan University

Last Updated: 2026/3/15
"""

from sim_RC_library import *
import matplotlib as mpl
from matplotlib.patches import Ellipse
from scipy.stats import spearmanr


def MG_SRC_sim(
        length=1440, shift=1,  # Default MG series settings
        num_node=10, mask_abs=0.1,  # Default RC settings
        direct_transfer=0,
        no_pic=False,
        AVG=0,
        self=False,
        noise_level=1e-6,
        C2C_variation=0.01e-5,
        C2C_test_control=False,
        noise_test_control=False,
        num_res=3,  # Number of different reservoirs when direct_transfer is False
        Ts_k3=1.16e-5,
):
    # In the training phase, the signal for different memristor reservoir is the same
    MG_gen = MG_generator(0.2, 0.1, 10, 18, shift=shift)
    signal, target = MG_gen.iterate(1, length)
    mask = create_mask(num_node, abs_value=mask_abs)

    Input_tr, input_ts, Target_tr, target_ts = \
        signal_process(signal, target, mask)

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

    # Create the SRC module
    SRC = TiOx_SRC()

    # Training
    Tr_set_k3 = np.linspace(0.96e-5, 1.2e-5, num_res)

    # # Temporary term
    if self is True:
        Tr_set_k3 = np.linspace(Ts_k3, 1.2e-5, 1)

    if not AVG:
        State_tr = np.zeros((int(len(Target_tr) / num_res) * num_res, num_node))
    else:
        State_tr = np.zeros((len(Target_tr) * num_res, num_node))
    for i in range(num_res):
        i_tr, g_tr, g0_tr = SRC.iterate_SRC(input_tr, 20e-6, k3=Tr_set_k3[i], virtual_nodes=num_node, clear=True,
                                            C2C_strength=C2C_variation)
        if not AVG:
            State_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), :] = \
                i_tr.reshape(int(len(Target_tr) / num_res), num_node)
        else:
            State_tr[i * len(Target_tr):(i + 1) * len(Target_tr), :] += \
                i_tr.reshape(len(Target_tr), num_node)

    if AVG:
        State_tr /= num_res

    # Add noise
    State_tr += noise_level * np.random.randn(State_tr.shape[0], State_tr.shape[1])

    # Linear regression
    lin = Ridge(alpha=0)
    lin.fit(State_tr, target_tr)
    Output_tr = lin.predict(State_tr)

    # Testing
    if C2C_test_control:
        C2C_variation = 0
    if noise_test_control:
        noise_level = 1e-6
    i_ts, g_ts, g0_ts = SRC.iterate_SRC(input_ts, 20e-6, k3=Ts_k3, virtual_nodes=num_node, clear=True,
                                        C2C_strength=C2C_variation)
    State_ts = i_ts.reshape(len(target_ts), num_node)
    State_ts += noise_level * np.random.randn(State_ts.shape[0], State_ts.shape[1])


    Output_ts = lin.predict(State_ts)

    NRMSE_tr, NRMSE_ts = nrmse(target_tr, Output_tr), nrmse(target_ts, Output_ts)

    if not no_pic:

        color4 = np.array([107, 158, 184]) / 255
        color3 = np.array([103, 149, 216]) / 255
        color2 = np.array([110, 167, 151]) / 255
        color1 = np.array([117, 185, 86]) / 255

        figure, ax = plt.subplots(2, 2, figsize=(2.4, 2), sharey='row', sharex='col')

        ax1, ax2, ax3, ax4 = ax[1, 0], ax[1, 1], ax[0, 0], ax[0, 1]
        plt.rc('font', family='Arial', size=6)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['lines.linewidth'] = 1.2

        ylim_max = 0.035

        # Subplot1

        if direct_transfer:
            ax1.plot((Output_tr[:, 0] - target_tr[:, 0]) ** 2, label='Training Error',
                     color=color1)
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
                ax1.plot(np.arange(i * int(len(Target_tr) / num_res), (i + 1) * int(len(Target_tr) / num_res)),
                         (Output_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0] -
                          target_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0]) ** 2,
                         label='Training Error',
                         color=color)
                ax3.plot(
                    np.arange(i * int(len(Target_tr) / num_res), (i + 1) * int(len(Target_tr) / num_res)),
                    target_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0],
                    color=np.array([200, 200, 200]) / 255
                )
                ax3.plot(
                    np.arange(i * int(len(Target_tr) / num_res), (i + 1) * int(len(Target_tr) / num_res)),
                    Output_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0],
                    color=color
                )

        ax1.set_xlim(0, 720)
        ax1.set_ylim(0, ylim_max)
        ax3.set_ylim(0.2, 1.6)

        ax1.set_xlabel('Time step', fontdict={'family': 'arial', 'size': 6}, labelpad=1, x=0.8, ha='left')
        ax1.set_ylabel('Squared error', fontdict={'family': 'arial', 'size': 6})
        ax3.set_ylabel(r'$x$', fontdict={'family': 'arial', 'size': 6})
        ax1.tick_params(axis='both', direction='in', labelsize=6)
        ax3.tick_params(axis='both', direction='in', labelsize=6)
        ax1.set_xticks([0, 240, 480, 720])
        ax1.set_yticks([0, 0.01, 0.02, 0.03])

        # Subplot2

        ax2.plot(np.arange(720, 1440), (Output_ts[:, 0] - target_ts[:, 0]) ** 2, label='Testing Error',
                 color=color4)
        ax4.plot(np.arange(720, 1440), target_ts[:, 0], color=np.array([200, 200, 200]) / 255)
        ax4.plot(np.arange(720, 1440), Output_ts[:, 0], color=color4)
        ax2.set_xlim(720, 1440)
        ax2.set_ylim(0, ylim_max)

        # ax2.set_xlabel('Time Step', fontdict={'family': 'arial', 'size': 6})
        ax2.set_xticks([960, 1200, 1440])
        ax2.tick_params(axis='both', direction='in', labelsize=6)
        ax4.tick_params(axis='both', direction='in', labelsize=6)
        figure.subplots_adjust(wspace=0, hspace=0.1)

        if AVG:
            suffix='_AVG'
        else:
            suffix=''

        if not direct_transfer:
            plt.savefig('./Figure/MG/Sim_Error_SRC_SC{}.svg'.format(suffix), dpi=300,
                        format='svg',
                        transparent=True, bbox_inches='tight')
        else:
            plt.savefig('./Figure/MG/Sim_Error_DT_SC{}.svg'.format(suffix), dpi=300,
                        format='svg',
                        transparent=True, bbox_inches='tight')
        plt.show()

        print('NRMSE tr is {}'.format(NRMSE_tr))
        print('NRMSE ts is {}'.format(NRMSE_ts))
    else:
        return NRMSE_tr, NRMSE_ts


def MG_Expr_read_in():
    device_code = ['4u', '6u', '7u', '8u', '9u', '11u', '14d', '15d', '16d']
    for i_mask in range(1, 4):
        path = './Data/MG readin/mask {}'.format(i_mask)
        os.makedirs(path, exist_ok=True)
        for i_device in range(len(device_code)):
            i_device_state = np.zeros((1600, 10))
            for i_segment in range(0, 20):
                mask_choice = '.\Data\MG\Exp\TiOx/5um_mask{}'.format(
                    i_mask)
                file = mask_choice + '/results/5um_{}/5um_{}_Mask{}Seg{}.csv'.format(
                    device_code[i_device], device_code[i_device],
                    i_mask, i_segment
                )

                df = pd.read_csv(file, header=None, sep='\n')
                df = df[0].str.split(',', expand=True)
                df_0 = df.iloc[148:20148, 1:4]
                df_0_numpy = df_0.to_numpy()
                data = df_0_numpy.astype(np.float64)
                data_RC_one_device = - data[:, 2]  # electric current
                # voltage_RC_one_device = data[:, 1]

                # Take the sampling points
                down_sampling_ratio = int(len(data_RC_one_device) / (80 + 20) / 10)
                data_resampled = data_RC_one_device[15::down_sampling_ratio]
                # Make sure that down_sampling_start should be smaller than down_sampling_ratio
                data_reshaped = np.reshape(data_resampled, (80 + 20, 10))
                data_response = data_reshaped[5:-20 + 5]
                i_device_state[i_segment * 80:(i_segment + 1) * 80, :] = data_response[:, :]

            np.savetxt(path + '/MG device {}.csv'.format(device_code[i_device]), i_device_state, delimiter=',')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% The Generation of Target Series %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Read the file and train and test based on the result from the physical experiment

    point_number = 80 * 20 + 20

    # Default Settings
    a = 0.2
    b = 0.1
    c = 10
    tau = 18
    tr_warmup_overlap = 5
    warm_up = max(int(1 * point_number), 1000)
    pred_shift = 1
    time_total = (pred_shift + point_number + warm_up)

    x_initial_len = int(tau)
    running_len = (pred_shift + point_number + warm_up)
    x_total_len = x_initial_len + running_len

    # Initialization for MG DDE
    t = np.array([np.arange(0, time_total, 1)]).T
    x_record = np.zeros((x_total_len, 1)) + 0.01

    for i in range(running_len):
        x_record[i + x_initial_len, 0] = mackey_glass_func(1, x_record[i + x_initial_len - 1], x_record[i], a, b, c)

    x_target = x_record[x_initial_len + warm_up + pred_shift:]
    x_target = x_target[tr_warmup_overlap:-20 + tr_warmup_overlap]

    np.savetxt('./Data/MG readin/MG target.csv', x_target, delimiter=',')


def color_select(i, n_groups=8):
    c0 = np.array([117, 185, 86]) / 255   # 起始色
    c1 = np.array([103, 149, 216]) / 255  # 结束色
    t = i / n_groups if n_groups > 1 else 0.0
    return c0 + t * (c1 - c0)

def MG_SRC_Expr(
        train_combo=(8, 0, 5),
        test_index=7,
        no_pic=True,
        direct_transfer=False,
        mask_choice=3
):
    if direct_transfer:
        train_combo = (8,)

    test_device = test_index
    combination = train_combo
    N = len(combination)
    rep_len = int(720 / N)
    device_code = ['4u', '6u', '7u', '8u', '9u', '11u', '14d', '15d', '16d']
    # The calculated averages of delta I1
    device_character_I = [153.2, 147.7, 154.1, 155.5, 154.9, 160.1, 154.0, 158.0, 130.8]
    color_sort_rank = [sorted(device_character_I).index(x) for x in device_character_I]

    # full target
    file_target = './Data/MG readin/MG target.csv'
    df = pd.read_csv(file_target, header=None, sep='\n')
    full_target = df.to_numpy().astype(np.float64)

    # Training
    RC_tr_storage = np.zeros((720, 10))
    target_tr = np.zeros((720, 1))

    # In K_train steps
    for k_device in range(N):
        i_device = combination[k_device]
        file = './Data/MG readin/mask {}/MG device {}.csv'.format(mask_choice, device_code[i_device])
        df = pd.read_csv(file, header=None, sep='\n')
        df = df[0].str.split(',', expand=True)
        df_n = df.to_numpy().astype(np.float64)
        select_slice = df_n[800-rep_len:800, :]
        RC_tr_storage[k_device * rep_len: (k_device+1) * rep_len, :] = select_slice[:, :]
        target_tr[k_device * rep_len: (k_device+1) * rep_len, :] = full_target[800-rep_len:800, :]

    # Data collection for testing
    file_test = './Data/MG readin/mask {}/MG device {}.csv'.format(mask_choice, device_code[test_device])
    df = pd.read_csv(file_test, header=None, sep='\n')
    df = df[0].str.split(',', expand=True)
    df_n = df.to_numpy().astype(np.float64)
    RC_ts_storage = df_n[800:800+720, :]
    target_ts = full_target[800:800+720, :]

    ridge_alpha = 0
    lin = Ridge(alpha=ridge_alpha)

    lin.fit(RC_tr_storage, target_tr)

    output_tr = lin.predict(RC_tr_storage)
    output_ts = lin.predict(RC_ts_storage)

    NRMSE_tr, NRMSE_ts = nrmse(target_tr, output_tr), nrmse(target_ts, output_ts)
    print('Current combination is {}'.format(combination))
    print('Train NRMSE is {}'.format(NRMSE_tr))
    print('Test NRMSE is {}'.format(NRMSE_ts))

    if not no_pic:

        figure, ax = plt.subplots(2, 2, figsize=(2.4, 2), sharey='row', sharex='col')

        ax1, ax2, ax3, ax4 = ax[1, 0], ax[1, 1], ax[0, 0], ax[0, 1]
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

            ax1.plot(np.arange(rep_len * i, rep_len * (i+1)),
                     (output_tr[i * rep_len:(i + 1) * rep_len, 0] -
                      target_tr[i * rep_len:(i + 1) * rep_len, 0]) ** 2,
                     label='Training Error',
                     color=color)
            ax3.plot(
                np.arange(i * rep_len, (i + 1) * rep_len),
                target_tr[i * rep_len:(i + 1) * rep_len, 0],
                color=np.array([200, 200, 200]) / 255
            )
            ax3.plot(
                np.arange(i * rep_len, (i + 1) * rep_len),
                output_tr[i * rep_len:(i + 1) * rep_len, 0],
                color=color
            )

        ax1.set_xlim(0, 720)
        ax1.set_ylim(0, ylim_max)
        ax3.set_ylim(0.2, 1.6)

        ax1.set_xlabel('Time step', fontdict={'family': 'arial', 'size': 6}, labelpad=1)
        ax1.set_ylabel('Squared error', fontdict={'family': 'arial', 'size': 6})
        ax3.set_ylabel(r'$x$', fontdict={'family': 'arial', 'size': 6})
        ax1.tick_params(axis='both', direction='in', labelsize=6)
        ax3.tick_params(axis='both', direction='in', labelsize=6)
        ax1.set_xticks([rep_len * j for j in range(N+1)])
        ax1.set_yticks([0, 0.01, 0.02, 0.03])

        # Subplot2
        color_test = color_select(color_sort_rank[test_device])
        ax2.plot(np.arange(720, 1440), (output_ts[:, 0] - target_ts[:, 0]) ** 2, label='Testing Error',
                 color=color_test)
        ax4.plot(np.arange(720, 1440), target_ts[:, 0], color=np.array([200, 200, 200]) / 255)
        ax4.plot(np.arange(720, 1440), output_ts[:, 0], color=color_test)
        ax2.set_xlim(720, 1440)
        ax2.set_ylim(0, ylim_max)

        # ax2.set_xlabel('Time Step', fontdict={'family': 'arial', 'size': 6})
        ax2.set_xticks([rep_len * j for j in range(N+1, 2*N+1)])
        ax2.tick_params(axis='both', direction='in', labelsize=6)
        ax4.tick_params(axis='both', direction='in', labelsize=6)
        figure.subplots_adjust(wspace=0, hspace=0.1)

        if not direct_transfer:
            plt.savefig('./Figure/MG/Exp_Error_SRC_SC.svg', dpi=300,
                        format='svg',
                        transparent=True, bbox_inches='tight')
        else:
            plt.savefig('./Figure/MG/Exp_Error_DT_SC.svg', dpi=300,
                        format='svg',
                        transparent=True, bbox_inches='tight')

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
        return NRMSE_tr, NRMSE_ts, q


def NRMSE_sim(**kwargs):
    levels = 9
    k3_list = np.linspace(1, 1.25, levels)
    repeat = kwargs.get('repeat', 20)
    dict_nrmse_dt = {}
    dict_nrmse_src = {}

    Storage_nrmse_dt = np.zeros((levels, repeat))
    Storage_nrmse_src = np.zeros((levels, repeat))

    for i in range(levels):
        for j in range(repeat):
            _, Storage_nrmse_dt[i, j] = MG_SRC_sim(
                direct_transfer=True, Ts_k3=0.96e-5 * k3_list[i], no_pic=True
            )
        dict_nrmse_dt['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_dt[i, :]

    for i in range(levels):
        for j in range(repeat):
            _, Storage_nrmse_src[i, j] = MG_SRC_sim(
                direct_transfer=False, Ts_k3=0.96e-5 * k3_list[i], no_pic=True
            )
        dict_nrmse_src['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_src[i, :]

    with open('./Data/MG/Sim/TiOx/nrmse_classical.csv', mode='w',
              encoding='UTF-8', newline='') as f_tr:
        writer = csv.writer(f_tr)
        writer.writerows(Storage_nrmse_dt)

    with open('./Data/MG/Sim/TiOx/nrmse_TS.csv', mode='w',
              encoding='UTF-8', newline='') as f_ts:
        writer = csv.writer(f_ts)
        writer.writerows(Storage_nrmse_src)

    pass


def NRMSE_sim_plot():
    levels = 9
    k3_list = np.linspace(1, 1.25, levels)
    Storage_nrmse_dt = pd.read_csv('./Data/MG/Sim/TiOx/nrmse_classical.csv', header=None).values
    Storage_nrmse_src = pd.read_csv('./Data/MG/Sim/TiOx/nrmse_TS.csv', header=None).values

    dict_nrmse_dt = {}
    dict_nrmse_src = {}

    for i in range(levels):
        dict_nrmse_dt['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_dt[i, :]
        dict_nrmse_src['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_src[i, :]

    color4 = np.array([107, 158, 184]) / 255
    color3 = np.array([103, 149, 216]) / 255
    color2 = np.array([110, 167, 151]) / 255
    color1 = np.array([117, 185, 86]) / 255

    plt.figure(figsize=(2.4, 1.6))
    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    sns.boxplot(data=pd.DataFrame(dict_nrmse_dt).iloc[:, :], color=color1, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth': 0.5}, whiskerprops={'linewidth': 0.5}, medianprops={'linewidth': 0.5},
                capprops={'linewidth': 0.5}, flierprops={'marker': 'o'})
    sns.pointplot(data=pd.DataFrame(dict_nrmse_dt).iloc[:, :], scale=0.75, color=color1,
                  label=r'Classical framework $({\bf W}^{(9)}_{\rm out})$')
    sns.boxplot(data=pd.DataFrame(dict_nrmse_src).iloc[:, :], color=color4, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth': 0.5}, whiskerprops={'linewidth': 0.5}, medianprops={'linewidth': 0.5},
                capprops={'linewidth': 0.5}, flierprops={'marker': 'o'})
    sns.pointplot(data=pd.DataFrame(dict_nrmse_src).iloc[:, :], scale=0.75, color=color4,
                  label=r'TS training $({\bf W}^{(1,6,9)}_{\rm out})$')
    plt.ylim([0, 0.6])
    plt.legend(frameon=False, loc=2)
    plt.ylabel('NRMSE')
    plt.xlabel('Percentage of difference in k3 (%)')

    plt.savefig('./Figure/MG/Sim_NRMSE_compare_ordered.svg', dpi=300,
                format='svg',
                transparent=True, bbox_inches='tight')
    plt.show()

def MG_SRC_Expr_MultiChannel(
        direct_transfer=False, AVG=False, spatial_reorder=False, temporal_reorder=False
):

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% The Generation of Target Series %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Read the file and train and test based on the result from the physical experiment

    # verify_pair = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Default Settings
    down_sampling_start = 15
    tr_warmup_overlap = None

    each_length = 80
    out_dim = 10
    a = 0.2
    b = 0.1
    c = 10
    tau = 18
    dt = 1
    overlap=20
    Split = 0.5

    # Default Setting about the warmup-overlap in actual training-inference of the RC based on physical measurement
    tr_warmup_overlap = 5
    # tr_warmup_overlap should not be bigger than overlap

    # Points in simulation
    # points for merely warmup, default to be identical length of the needed length
    point_number = 80*20+20
    warm_up = max(int(1 * point_number), 1000)
    pred_shift = 1  # pred_shift = 1  # Points for direct prediction shift

    time_total = (pred_shift + point_number + warm_up) * dt

    x_initial_len = int(tau / dt)
    running_len = (pred_shift + point_number + warm_up)
    x_total_len = x_initial_len + running_len

    # Initialization for MG DDE
    t = np.array([np.arange(0, time_total, dt)]).T
    x_record = np.zeros((x_total_len, 1)) + 0.01

    for i in range(running_len):
        x_record[i + x_initial_len, 0] = mackey_glass_func(dt, x_record[i + x_initial_len - 1], x_record[i], a, b, c)

    x_target = x_record[x_initial_len + warm_up + pred_shift:]

    # Generate the true target
    x_target = x_target[tr_warmup_overlap:-overlap + tr_warmup_overlap]

    x_target_tr = x_target[:int(len(x_target) * Split)]
    x_target_ts = x_target[int(len(x_target) * Split):]

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End of the Generation of the Target Series %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Collecting RC Response from the Devices %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    num_parallel = 3

    RC_tr_storage = np.zeros((720, num_parallel * 10))
    RC_ts_storage = np.zeros((720, num_parallel * 10))
    ts_segments = 9

    if not direct_transfer:
        if AVG:
            tr_segments = 27
            train_device_order = [['14d', '8u', '14d']] * 9 + [['16d', '7u', '4u']] * 9 + [['9u', '15d', '11u']] * 9
            target_tr = x_target_tr[-720:]
            target_ts = x_target_ts[:720]
            train_serial_order = [1, 2, 3, 4, 5 ,6, 7, 8, 9,
                                  1, 2, 3, 4, 5 ,6, 7, 8, 9,
                                  1, 2, 3, 4, 5 ,6, 7, 8, 9]
            test_serial_order = [i+10 for i in range(9)]
        else:
            tr_segments = 9
            train_device_order = [['14d', '8u', '14d']] * 3 + [['16d', '7u', '4u']] * 3 + [['9u', '15d', '11u']] * 3
            if temporal_reorder:
                train_device_order = [['9u', '15d', '11u']] * 3 + [['16d', '7u', '4u']] * 3 + [['14d', '8u', '14d']] * 3
            target_tr = np.tile(x_target_tr[-240:], (3, 1))
            target_ts = x_target_ts[:720]
            train_serial_order = [7, 8, 9, 7, 8, 9, 7, 8, 9]
            test_serial_order = [i+10 for i in range(9)]
    else:
        tr_segments = 9
        train_device_order = [['14d', '8u', '14d']] * 9
        target_tr = x_target_tr[-720:]
        target_ts = x_target_ts[:720]
        train_serial_order = [1+i for i in range(9)]
        test_serial_order = [i+10 for i in range(9)]
    test_device_order = [['16d', '11u', '11u']] * 9

    if spatial_reorder and direct_transfer:
        train_device_order = [['11u', '4u', '16d']] * 9
        test_device_order = [['16d', '4u', '11u']] * 9

    for i_tr in range(tr_segments):
        for choice in range(num_parallel):
            mask_choice = '.\Data\MG\Exp\TiOx/5um_mask{}'.format(choice+1)
            file = mask_choice + '/results/5um_{}/5um_{}_Mask{}Seg{}.csv'.format(
                train_device_order[i_tr][choice], train_device_order[i_tr][choice],
                choice+1, train_serial_order[i_tr]
            )

            df = pd.read_csv(file, header=None, sep='\n')
            df = df[0].str.split(',', expand=True)
            df_0 = df.iloc[148:20148, 1:4]
            df_0_numpy = df_0.to_numpy()
            data = df_0_numpy.astype(np.float64)
            data_RC_one_device = - data[:, 2]

            # Take the sampling points
            down_sampling_ratio = int(len(data_RC_one_device) / (each_length + overlap) / out_dim)
            data_resampled = data_RC_one_device[down_sampling_start::down_sampling_ratio]
            # Make sure that down_sampling_start should be smaller than down_sampling_ratio
            data_reshaped = np.reshape(data_resampled, (each_length + overlap, out_dim))
            data_response = data_reshaped[tr_warmup_overlap:-overlap + tr_warmup_overlap]

            RC_tr_storage[(i_tr % 9) * each_length:(i_tr % 9 + 1) * each_length, choice * out_dim:(choice+1) * out_dim] += \
                data_response[:, :]

    if AVG:
        RC_tr_storage = RC_tr_storage / 3

    for i_ts in range(ts_segments):
        for choice in range(num_parallel):
            mask_choice = '.\Data\MG\Exp\TiOx/5um_mask{}'.format(choice+1)
            file = mask_choice + '/results/5um_{}/5um_{}_Mask{}Seg{}.csv'.format(
                test_device_order[i_ts][choice], test_device_order[i_ts][choice],
                choice+1, test_serial_order[i_ts]
            )

            df = pd.read_csv(file, header=None, sep='\n')
            df = df[0].str.split(',', expand=True)
            df_0 = df.iloc[148:20148, 1:4]
            df_0_numpy = df_0.to_numpy()
            data = df_0_numpy.astype(np.float64)
            data_RC_one_device = - data[:, 2]

            # Take the sampling points
            down_sampling_ratio = int(len(data_RC_one_device) / (each_length + overlap) / out_dim)
            data_resampled = data_RC_one_device[down_sampling_start::down_sampling_ratio]
            # Make sure that down_sampling_start should be smaller than down_sampling_ratio
            data_reshaped = np.reshape(data_resampled, (each_length + overlap, out_dim))
            data_response = data_reshaped[tr_warmup_overlap:-overlap + tr_warmup_overlap]

            RC_ts_storage[i_ts * each_length:(i_ts + 1) * each_length, choice * out_dim:(choice+1) * out_dim] = \
                data_response[:, :]

    # RC training
    ridge_alpha = 0
    lin = Ridge(alpha=ridge_alpha)

    lin.fit(RC_tr_storage, target_tr)

    x_bar_tr = lin.predict(RC_tr_storage)
    x_bar_ts = lin.predict(RC_ts_storage)

    print('Train NRMSE is {}'.format(nrmse(target_tr, x_bar_tr)))
    print('Test NRMSE is {}'.format(nrmse(target_ts, x_bar_ts)))

    Target_tr = target_tr
    Output_tr = x_bar_tr
    Output_ts = x_bar_ts

    figure, ax = plt.subplots(2, 2, figsize=(4, 2.4), sharey='row', sharex='col')

    ax1, ax2, ax3, ax4 = ax[1, 0], ax[1, 1], ax[0, 0], ax[0, 1]
    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1.2

    ylim_max = 0.1

    # Subplot1
    color4 = np.array([107, 158, 184]) / 255
    color3 = np.array([103, 149, 216]) / 255
    color2 = np.array([110, 167, 151]) / 255
    color1 = np.array([117, 185, 86]) / 255

    if direct_transfer:
        ax1.plot((Output_tr[:, 0] - target_tr[:, 0]) ** 2, label='Training Error',
                 color=color1)
        ax3.plot(target_tr[:, 0], color=np.array([200, 200, 200]) / 255)
        ax3.plot(Output_tr[:, 0], color=color1)

    else:

        colors = [color1, color2, color3]
        if temporal_reorder:
            colors = [color3, color2, color1]
        num_res = 3
        for i in range(num_res):
            color = colors[i % num_res]
            if AVG:
                color = np.array([138, 127, 214]) / 255
            ax1.axvline(240, ls='--', color=np.array([180, 180, 180]) / 255)
            ax1.axvline(480, ls='--', color=np.array([180, 180, 180]) / 255)
            ax3.axvline(240, ls='--', color=np.array([180, 180, 180]) / 255)
            ax3.axvline(480, ls='--', color=np.array([180, 180, 180]) / 255)
            ax1.plot(np.arange(i * int(len(Target_tr) / num_res), (i + 1) * int(len(Target_tr) / num_res)),
                     (Output_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0] -
                      target_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0]) ** 2,
                     label='Training Error',
                     color=color)
            ax3.plot(
                np.arange(i * int(len(Target_tr) / num_res), (i + 1) * int(len(Target_tr) / num_res)),
                target_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0],
                color=np.array([200, 200, 200]) / 255
            )
            ax3.plot(
                np.arange(i * int(len(Target_tr) / num_res), (i + 1) * int(len(Target_tr) / num_res)),
                Output_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0],
                color=color
            )

    ax1.set_xlim(0, 720)
    ax1.set_ylim(0, ylim_max)
    ax3.set_ylim(0.2, 1.6)

    # ax1.set_xlabel('Time step', fontdict={'family': 'arial', 'size': 6}, labelpad=1)
    ax1.set_ylabel('Squared error', fontdict={'family': 'arial', 'size': 6})
    ax3.set_ylabel(r'$x$', fontdict={'family': 'arial', 'size': 6})
    ax1.tick_params(axis='both', direction='in', labelsize=6)
    ax3.tick_params(axis='both', direction='in', labelsize=6)
    ax1.set_xticks([0, 240, 480, 720])
    ax1.set_yticks([0, 0.03, 0.06, 0.09])

    # Subplot2

    ax2.plot(np.arange(720, 1440), (Output_ts[:, 0] - target_ts[:, 0]) ** 2, label='Testing Error',
             color=color4)
    ax4.plot(np.arange(720, 1440), target_ts[:, 0], color=np.array([200, 200, 200]) / 255)
    ax4.plot(np.arange(720, 1440), Output_ts[:, 0], color=color4)
    ax2.set_xlim(720, 1440)
    ax2.set_ylim(0, ylim_max)

    ax2.set_xticks([960, 1200, 1440])
    ax2.tick_params(axis='both', direction='in', labelsize=6)
    ax4.tick_params(axis='both', direction='in', labelsize=6)
    # figure.subplots_adjust(wspace=0, hspace=0.1)
    figure.subplots_adjust(wspace=0, hspace=0.1, bottom=0.18)  # 给底部留点空间
    figure.supxlabel('Time step', x=0.5, y=0.04, fontname='Arial', fontsize=6)

    if temporal_reorder:
        suffix='_TimePermut'
    elif spatial_reorder:
        suffix='_SpacePermut'
    elif AVG:
        suffix='_AVG'
    else:
        suffix=''
        
    if not direct_transfer:
        plt.savefig('./Figure/MG/Exp_Error_SRC_MC{}.svg'.format(suffix), dpi=300,
                    format='svg',
                    transparent=True, bbox_inches='tight')
    else:
        plt.savefig('./Figure/MG/Exp_Error_DT_MC{}.svg'.format(suffix), dpi=300,
                    format='svg',
                    transparent=True, bbox_inches='tight')

    plt.show()


def NRMSE_expr():

    dict_nrmse_dt = {}
    dict_nrmse_src = {}

    nrmse_record_dt= np.zeros(9)
    nrmse_record_src = np.zeros(9)

    for i in range(9):
        _, nrmse_record_dt[i], _ = MG_SRC_Expr(
            no_pic=True, direct_transfer=True, test_index=i
        )

        dict_nrmse_dt['{}'.format(i+1)] = nrmse_record_dt[i]

    for i in range(9):
        _, nrmse_record_src[i], _ = MG_SRC_Expr(
            no_pic=True, direct_transfer=False, test_index=i
        )

        dict_nrmse_src['{}'.format(i+1)] = nrmse_record_src[i]

    # Original device order
    plt.figure(figsize=(2.4, 1.6))
    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    device_serial = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    color4 = np.array([107, 158, 184]) / 255
    color3 = np.array([103, 149, 216]) / 255
    color2 = np.array([110, 167, 151]) / 255
    color1 = np.array([117, 185, 86]) / 255

    sns.lineplot(x=device_serial, y=nrmse_record_dt, color=color1)
    sns.lineplot(x=device_serial, y=nrmse_record_src, color=color4)
    plt.scatter(x=device_serial, s=10, y=nrmse_record_dt, color=color1, label=r'Classical framework $({\bf W}^{(9)}_{\rm out})$')
    plt.scatter(x=device_serial, s=10, y=nrmse_record_src, color=color4, label=r'TS training $({\bf W}^{(1,6,9)}_{\rm out})$')
    plt.ylim([0, 0.6])
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.legend(frameon=False, loc=2)
    plt.ylabel('NRMSE')
    plt.xlabel('Device serial')
    plt.savefig('./Figure/MG/Exp_NRMSE_compare.svg', dpi=300, format='svg', transparent=True, bbox_inches='tight')
    plt.show()

    # Rearranged I1 order
    plt.figure(figsize=(2.4, 1.6))
    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1

    average_record = Average_deltaI1()
    train_base = 8
    average_record = np.abs(average_record - average_record[train_base]) / average_record[train_base]

    sns.lineplot(x=average_record*100, y=nrmse_record_dt, color=color1)
    sns.lineplot(x=average_record*100, y=nrmse_record_src, color=color4)
    plt.scatter(x=average_record*100, y=nrmse_record_dt, s=10, color=color1, label=r'Classical framework $({\bf W}^{(9)}_{\rm out})$')
    plt.scatter(x=average_record*100, y=nrmse_record_src, s=10, color=color4, label=r'TS training $({\bf W}^{(1,6,9)}_{\rm out})$')
    plt.ylim([0, 0.6])
    plt.legend(frameon=False, loc=2)
    plt.ylabel('NRMSE')
    plt.xlabel('Percentage of difference in $\Delta I(t_1)$ (%)')
    plt.savefig('./Figure/MG/Exp_NRMSE_compare_ordered.svg', dpi=300, format='svg', transparent=True, bbox_inches='tight')
    plt.show()


def Average_deltaI1():
    voltages = [3]
    rounds = 50

    device_serial = ['4u', '6u', '7u', '8u', '9u', '11u', '14d', '15d', '16d']  # For 5um devices

    Average_record = np.zeros(9)
    # for i in range(devices):
    for i in range(len(device_serial)):
        for k in range(len(voltages)):
            voltage = voltages[k]
            I_0 = np.zeros(rounds)
            time_0 = np.zeros(rounds)
            I_1 = np.zeros(rounds)
            time_1 = np.zeros(rounds)
            I_2 = np.zeros(rounds)
            time_2 = np.zeros(rounds)

            df = pd.read_csv('./Data/Characterization/TiOx/Pulse/' + '5um_{}_Pulse_W3R0.5_50.csv'.format(device_serial[i]),
                             header=None, sep='\n')
            df = df[0].str.split(',', expand=True)
            for j in range(rounds):
                df_0 = df.iloc[158 + 796 * j:159 + 796 * j, 1:4]
                df_0_numpy = df_0.to_numpy()

                time_0[j] = float(df_0_numpy[0, 0])
                I_0[j] = float(df_0_numpy[0, 2])

                df_1 = df.iloc[178 + 796 * j:179 + 796 * j, 1:4]
                df_1_numpy = df_1.to_numpy()
                time_1[j] = float(df_1_numpy[0, 0])
                I_1[j] = float(df_1_numpy[0, 2])

                df_2 = df.iloc[228 + 796 * j:229 + 796 * j, 1:4]
                df_2_numpy = df_2.to_numpy()
                time_2[j] = float(df_2_numpy[0, 0])
                I_2[j] = float(df_2_numpy[0, 2])

            delta_I_1 = I_0 - I_1

            Average_record[i] = np.average(delta_I_1*1e6)

    return Average_record


def Extended_Data_TS_advantage_result():

    # Experiment
    dict_nrmse_dt = {}
    dict_nrmse_src = {}

    nrmse_record_dt = np.zeros(9)
    nrmse_record_src = np.zeros(9)

    for i in range(9):
        _, nrmse_record_dt[i], _ = MG_SRC_Expr(
            no_pic=True, direct_transfer=True, test_index=i
        )

        dict_nrmse_dt['{}'.format(i+1)] = nrmse_record_dt[i]

    for i in range(9):
        _, nrmse_record_src[i], _ = MG_SRC_Expr(
            no_pic=True, direct_transfer=False, test_index=i
        )

        dict_nrmse_src['{}'.format(i+1)] = nrmse_record_src[i]

    # Simulated
    levels = 9
    k3_list = np.linspace(1, 1.25, levels)
    Storage_nrmse_dt = pd.read_csv('./Data/MG/Sim/TiOx/nrmse_classical.csv', header=None).values
    Storage_nrmse_src = pd.read_csv('./Data/MG/Sim/TiOx/nrmse_TS.csv', header=None).values

    Diff_dict_nrmse_dt = {}

    for i in range(levels):
        Diff_dict_nrmse_dt['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_dt[i, :] - Storage_nrmse_src[i, :]

    # Rearranged I1 order
    fig = plt.figure(figsize=(4, 3))
    plt.rc('font', family='Arial', size=10)
    ax = AA.Subplot(fig, 111)
    fig.add_axes(ax)

    ax.axis['left'].set_axisline_style('-|>', size=1.5)
    ax.axis['left'].line.set_color('black')
    ax.axis['left'].label.set_text(r'$\Delta$ E')
    ax.axis['top', 'right', 'bottom'].set_visible(False)
    ax.axis['x'] = ax.new_floating_axis(nth_coord=0, value=0)
    ax.axis['x'].set_axisline_style('-|>', size=1.5)
    ax.axis['x'].label.set_text('\n Normalized D2D variation strength')
    ax.axis['x'].line.set_color('black')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(-0.04, 0.12)
    ax.set_yticks(np.linspace(-0.03, 0.09, 5))
    ax.set_xticks(np.linspace(0, 1, 3))
    ax.set_xticklabels(['','0.5', '1'])

    average_record = Average_deltaI1()
    train_base = 8
    average_record = np.abs(average_record - average_record[train_base]) / average_record[train_base]
    relative_record = average_record/np.max(average_record)

    color4 = np.array([107, 158, 184]) / 255
    color3 = np.array([103, 149, 216]) / 255
    color2 = np.array([110, 167, 151]) / 255
    color1 = np.array([117, 185, 86]) / 255

    sns.lineplot(x=relative_record*1, y=nrmse_record_dt**2-nrmse_record_src**2, color=color4)
    ax.scatter(x=relative_record*1, y=nrmse_record_dt**2-nrmse_record_src**2, s=10, color=color4, label=r'Experiment')
    sns.lineplot(x=(k3_list-1)*4, y=np.average(Storage_nrmse_dt**2-Storage_nrmse_src**2, axis=1), color=color1)
    ax.scatter(x=(k3_list-1)*4, y=np.average(Storage_nrmse_dt**2-Storage_nrmse_src**2, axis=1), s=10, color=color1, label=r'Simulation (Average)')

    plt.legend(frameon=False, loc=2)
    plt.savefig('./Figure/MG/TS_advantage_result.svg', dpi=300, format='svg', transparent=True, bbox_inches='tight')
    plt.show()


def Extended_Data_TS_advantage_schematic():
    # Only a schematic figure, the function value does not have practical significance

    x = np.linspace(0, 0.25, 26)
    y1 = 0.2/0.25*x - 0.05  # Linear approx. of the quadratic error difference
    y2 = 0.15*x**2 + y1  # The quadratic term 0.15 is far smaller than the linear term 0.8

    fig = plt.figure(figsize=(4, 3))
    plt.rc('font', family='Arial', size=10)
    ax = AA.Subplot(fig, 111)
    fig.add_axes(ax)

    ytick_position = [-0.05, 0, 0.05, 0.15]
    ytick_label = ['-E', '0', 'E', '>E']
    xtick_position = [1/4*25, 1/2*25, 1*25]
    xtick_label = [r'$p_{0}$', '0.5', '1']
    ax.axis['left'].set_axisline_style('-|>', size=1.5)
    ax.axis['left'].line.set_color('black')
    ax.axis['left'].label.set_text(r'$\Delta$ E')
    ax.axis['top', 'right', 'bottom'].set_visible(False)
    ax.axis['x'] = ax.new_floating_axis(nth_coord=0, value=0)
    ax.axis['x'].set_axisline_style('-|>', size=1.5)
    ax.axis['x'].label.set_text('\n Normalized D2D variation strength')
    ax.axis['x'].line.set_color('black')
    ax.set_xlim(0, 25*1.1)
    ax.set_ylim(-0.07, 0.22)
    ax.set_xticks(xtick_position)
    ax.set_xticklabels(xtick_label)
    ax.set_yticks(ytick_position)
    ax.set_yticklabels(ytick_label)

    plt.axvline(x=25, ymin=0.07/(0.07+0.22), ymax=(0.15+0.07)/(0.22+0.07), color='grey', linestyle='--')
    plt.axhline(y=0.15, xmin=0, xmax=1/1.1 , color='grey', linestyle='--')

    color4 = np.array([107, 158, 184]) / 255
    color3 = np.array([103, 149, 216]) / 255
    color2 = np.array([110, 167, 151]) / 255
    color1 = np.array([117, 185, 86]) / 255

    ax.plot(100*x, y2, color=color1, linestyle='--', label='Quadratic error')
    ax.plot(100*x, y1, color=color1, label='Linear approximation')
    plt.legend(frameon=False, loc=2)
    plt.savefig('./Figure/MG/TS_advantage_schematic.svg', dpi=300, format='svg', transparent=True, bbox_inches='tight')
    plt.show()


def SRC_Num_TS_device(readin=False):

    S = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 7 is left for testing
    order = [2, 3, 4, 5, 6]
    vals_by_N = {}
    if not readin:
        for N in range(2, 7):
            NRMSE_at_N = []

            print('-------------------------------------Current N is {}-------------------------------------'.format(N))
            for i in range(9):
                S_i = S[:i] + S[i+1:]
                Combos = [list(s) for s in combinations(S_i, N)]
                NumCombo_at_N = len(Combos)
                for j in range(NumCombo_at_N):
                    _, nrmse_ts, _ = MG_SRC_Expr(Combos[j], no_pic=True, test_index=i)
                    NRMSE_at_N.append(nrmse_ts)
            vals_by_N[str(N)] = np.array(NRMSE_at_N)
            print('------------------------Total number of combination at N={} is {}------------------------'.format(N, NumCombo_at_N*9))

        np.savez_compressed('./Data/MG readin/num_TS_device_effect.npz', **vals_by_N)
    else:
        vals_by_N = np.load('./Data/MG readin/num_TS_device_effect.npz', allow_pickle=False)

    df = pd.DataFrame(
        [(N, v) for N, arr in vals_by_N.items() for v in arr],
        columns=["N", "value"]
    )

    palette = [color_select(i, n_groups=5) for i in range(5)]

    df = df.copy()
    df["N"] = df["N"].astype(str).str.replace("N=", "", regex=False).astype(int)

    # statistics
    means = df.groupby("N")["value"].mean().reindex(order)
    medians = df.groupby("N")["value"].median().reindex(order)
    mins = df.groupby("N")["value"].min().reindex(order)

    q1 = df.groupby("N")["value"].quantile(0.25).reindex(order)
    q3 = df.groupby("N")["value"].quantile(0.75).reindex(order)
    iqr = (q3 - q1)

    p90 = df.groupby("N")["value"].quantile(0.90).reindex(order)
    p95 = df.groupby("N")["value"].quantile(0.95).reindex(order)

    xpos = np.arange(len(order))

    print('ALL means {}'.format(means.to_numpy().tolist()))
    print('ALL medians {}'.format(medians.to_numpy().tolist()))
    print('ALL IQRs {}'.format(iqr.to_numpy().tolist()))
    print('ALL p90 {}'.format(p90.to_numpy().tolist()))
    print('ALL p95 {}'.format(p95.to_numpy().tolist()))

    # =========================
    # plot
    # =========================
    fig1, ax1 = plt.subplots(figsize=(2.9, 2))
    sns.boxplot(
        data=df, x="N", y="value",
        order=order,
        palette=palette,
        ax=ax1,
        fliersize=1.5,
        saturation=1, width=0.7,
        boxprops={'linewidth': 0.5},
        whiskerprops={'linewidth': 0.5},
        medianprops={'linewidth': 0.5},
        capprops={'linewidth': 0.5},
        flierprops={'marker': 'o'}
    )
    ax1.plot(xpos, means.to_numpy(), marker="o", linewidth=1.2, markersize=4, label="Mean", zorder=3, color=np.array([160, 160, 160]) / 255)
    ax1.plot(xpos, medians.to_numpy(), marker="o", linewidth=1.2, markersize=4, label="Median", zorder=3, color=np.array([100, 100, 100]) / 255)
    ax1.set_yticks([0.15, 0.2, 0.25, 0.3, 0.35])
    ax1.set_xlabel("N (number of TS training devices)")
    ax1.set_ylabel("NRMSE")
    ax1.legend(frameon=False)
    fig1.tight_layout()
    fig1.savefig('./Figure/MG/MG_num_device_effect.svg', dpi=300, format='svg',
                 transparent=True, bbox_inches='tight')
    fig1.show()


def TS_robustness_boundary(readin=False):

    S = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    Ns = [2, 3, 4, 5, 6]
    target_test = 8

    vals_by_N = {}

    save_path = './Data/MG readin/num_TS_device_effect_q_with_testid.npz'

    # =========================
    # Generate and readin data
    # =========================
    if not readin:
        for N in range(2, 7):
            NRMSE_at_N = []
            q_at_N = []
            test_id_at_N = []

            print('-------------------------------------Current N is {}-------------------------------------'.format(N))

            for i in range(9):  # test_device
                S_i = S[:i] + S[i+1:]
                Combos = [list(s) for s in combinations(S_i, N)]
                NumCombo_at_N = len(Combos)

                for j in range(NumCombo_at_N):
                    _, nrmse_ts, q = MG_SRC_Expr(Combos[j], no_pic=True, test_index=i)
                    NRMSE_at_N.append(nrmse_ts)
                    q_at_N.append(q)
                    test_id_at_N.append(i)

            vals_by_N[str(N)] = np.array([NRMSE_at_N, q_at_N, test_id_at_N], dtype=object)
            print('------------------------Total number of combination at N={} is {}------------------------'.format(
                N, NumCombo_at_N * 9))

        np.savez_compressed(save_path, **vals_by_N)
        print(f"\nSaved to: {save_path}")
    else:
        vals_by_N = np.load(save_path, allow_pickle=True)
        print(f"\nLoaded from: {save_path}")

    # =========================
    # Three categories: in-range, out-of-range(all) and out-of-range(dev9)
    # =========================
    data_in = []          # q=0, all tests
    data_out_all = []     # q=1, all tests
    data_out_t8 = []      # q=1, test==8 (index 9)

    pvals_AB = []         # in vs out_all
    pvals_BC = []         # out_all vs out_t
    pvals_AC = []         # in vs out_t8

    print("\n==================== Summary per N (3 groups) ====================")

    for N in Ns:
        arr = vals_by_N[str(N)]  # shape (3, K)
        z = np.asarray(arr[0], float).ravel()
        q = np.asarray(arr[1], int).ravel()
        t = np.asarray(arr[2], int).ravel()

        z_in = z[q == 0]
        z_out = z[q == 1]
        z_out_8 = z[(q == 1) & (t == target_test)]

        data_in.append(z_in)
        data_out_all.append(z_out)
        data_out_t8.append(z_out_8)

        s_in = stats_summary_1d(z_in)
        s_out = stats_summary_1d(z_out)
        s_out8 = stats_summary_1d(z_out_8)

        print(f"\n[N={N}] n_in={s_in['n']}, n_out_all={s_out['n']}, n_out_test8={s_out8['n']}")
        print(f"  in-range      : median={s_in['median']:.6g}, IQR={s_in['iqr']:.6g}, P90={s_in['p90']:.6g}, P95={s_in['p95']:.6g}")
        print(f"  out-range(all): median={s_out['median']:.6g}, IQR={s_out['iqr']:.6g}, P90={s_out['p90']:.6g}, P95={s_out['p95']:.6g}")
        print(f"  out-range(t=8): median={s_out8['median']:.6g}, IQR={s_out8['iqr']:.6g}, P90={s_out8['p90']:.6g}, P95={s_out8['p95']:.6g}")

        # ---- Significance test, two-sided MWU test ----
        def safe_mwu(a, b):
            if (len(a) >= 2) and (len(b) >= 2):
                _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                return p
            return np.nan

        p_AB = safe_mwu(z_in, z_out)
        p_BC = safe_mwu(z_out, z_out_8)
        p_AC = safe_mwu(z_in, z_out_8)

        pvals_AB.append(p_AB)
        pvals_BC.append(p_BC)
        pvals_AC.append(p_AC)

        print(f"  MWU p(in vs out_all)   = {p_AB:.3g}" if np.isfinite(p_AB) else "  MWU p(in vs out_all)   = NA")
        print(f"  MWU p(out_all vs out8) = {p_BC:.3g}" if np.isfinite(p_BC) else "  MWU p(out_all vs out8) = NA")
        print(f"  MWU p(in vs out8)      = {p_AC:.3g}" if np.isfinite(p_AC) else "  MWU p(in vs out8)      = NA")

    # =========================
    # plot
    # =========================
    plt.figure(figsize=(2.9, 2.05))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1.0

    ax = plt.gca()

    centers = np.arange(len(Ns))
    offset = 0.24

    pos_out_all = centers - offset
    pos_in = centers
    pos_out_t8 = centers + offset

    c_in = np.array([117, 185, 86]) / 255      # green-ish
    c_out_all = np.array([103, 149, 216]) / 255  # blue-ish
    c_out_t8 = np.array([107, 158, 184]) / 255   # orange

    common_props = dict(
        patch_artist=True,
        showfliers=True,
        widths=0.22,
        boxprops={'linewidth': 0.5},
        whiskerprops={'linewidth': 0.5},
        medianprops={'linewidth': 0.5, 'color': 'k'},
        capprops={'linewidth': 0.5},
        flierprops={'marker': 'o', 'markersize': 2.0, 'markerfacecolor': 'k',
                    'markeredgecolor': 'k', 'alpha': 0.6},
    )

    # box-plot
    bp_out = ax.boxplot(data_out_all, positions=pos_out_all, **common_props)
    bp_in = ax.boxplot(data_in, positions=pos_in, **common_props)
    bp_out8 = ax.boxplot(data_out_t8, positions=pos_out_t8, **common_props)

    for patch in bp_in["boxes"]:
        patch.set_facecolor(c_in); patch.set_alpha(0.70)
    for patch in bp_out["boxes"]:
        patch.set_facecolor(c_out_all); patch.set_alpha(0.70)
    for patch in bp_out8["boxes"]:
        patch.set_facecolor(c_out_t8); patch.set_alpha(0.70)

    ax.set_xticks(centers)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_ylim([0.145, 0.405])
    ax.set_xlabel("N (number of TS training devices)")
    ax.set_ylabel("NRMSE")

    handles = [
        Patch(facecolor=c_out_all, edgecolor="k", alpha=0.70, label="Out-of-range (all tests)"),
        Patch(facecolor=c_in, edgecolor="k", alpha=0.70, label="In-range (all tests)"),
        Patch(facecolor=c_out_t8, edgecolor="k", alpha=0.70, label="Out-of-range (device 9)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="best")

    # =========================
    # significance symbol
    # =========================
    all_z = np.concatenate([np.concatenate([d for d in data_in if len(d)]),
                            np.concatenate([d for d in data_out_all if len(d)]),
                            np.concatenate([d for d in data_out_t8 if len(d)])])
    ymin, ymax = np.min(all_z), np.max(all_z)
    yr = ymax - ymin if ymax > ymin else 1.0
    h = 0.03 * yr
    pad = 0.02 * yr

    for i, N in enumerate(Ns):
        # in vs out_all
        zA = data_in[i]
        zB = data_out_all[i]
        if len(zA) and len(zB):
            y_top = np.max(np.concatenate([zA, zB]))
            y = y_top + pad
            stars = p_to_stars(pvals_AB[i], show_ns=False)
            add_sig_bracket(ax, pos_in[i], pos_out_all[i], y=y, h=h, text=stars, lw=0.9, fs=8)

        # in vs out_t8
        zC = data_out_t8[i]
        if len(zA) and len(zC):
            y_top2 = np.max(np.concatenate([zA, zC]))
            y2 = y_top2 + pad + h * 1.4  # Avoid overlapping of the brackets
            stars2 = p_to_stars(pvals_AC[i], show_ns=False)
            add_sig_bracket(ax, pos_in[i], pos_out_t8[i], y=y2, h=h, text=stars2, lw=0.9, fs=8)

    plt.tight_layout()
    plt.savefig("./Figure/MG/NRMSE_in_or_out_sig.svg", format="svg", dpi=300, transparent=True, bbox_inches="tight")
    plt.show()


def MG_SRC_sim_full_test(
        width = 0.12e-5,  # center 1.08e-5
        num_res = 3

):
    length = 1440
    shift = 1
    # Default MG series settings
    num_node = 10
    mask_abs = 0.1  # Default RC settings
    direct_transfer = 0

    noise_level =  1e-7
    C2C_variation = 0e-5
    C2C_test_control = False
    noise_test_control = False

    # In the training phase, the signal for different memristor reservoir is the same
    MG_gen = MG_generator(0.2, 0.1, 10, 18, shift=shift)
    signal, target = MG_gen.iterate(1, length)
    mask = create_mask(num_node, abs_value=mask_abs)

    Input_tr, input_ts, Target_tr, target_ts = \
        signal_process(signal, target, mask)

    if not direct_transfer:
        input_tr = Input_tr[-int(len(Input_tr) / num_res):]
        target_tr = Target_tr[-int(len(Target_tr) / num_res):]
        target_tr = np.tile(target_tr, (num_res, 1))
    else:
        input_tr = Input_tr
        target_tr = Target_tr
        num_res = 1

    # Create the SRC module
    SRC = TiOx_SRC()

    # Training
    Tr_set_k3 = np.linspace(1.08e-5 - width, 1.08e-5 + width, num_res)
    State_tr = np.zeros((int(len(Target_tr) / num_res) * num_res, num_node))
    for i in range(num_res):
        i_tr, g_tr, g0_tr = SRC.iterate_SRC(input_tr, 20e-6, k3=Tr_set_k3[i], virtual_nodes=num_node, clear=True,
                                            C2C_strength=C2C_variation)
        State_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), :] = \
            i_tr.reshape(int(len(Target_tr) / num_res), num_node)

    # Add noise
    State_tr += noise_level * np.random.randn(State_tr.shape[0], State_tr.shape[1])
    # Linear regression
    lin = Ridge(alpha=0)
    lin.fit(State_tr, target_tr)
    Output_tr = lin.predict(State_tr)
    NRMSE_tr = nrmse(target_tr, Output_tr)


    # Testing
    if C2C_test_control:
        C2C_variation = 0
    if noise_test_control:
        noise_level = 1e-6

    Ts_k3_list = np.linspace(1.08e-5 - 0.24e-5, 1.08e-5 + 0.24e-5, 25)
    NRMSE_ts_list = []
    for i_test in range(25):
        Ts_k3 = Ts_k3_list[i_test]
        i_ts, g_ts, g0_ts = SRC.iterate_SRC(input_ts, 20e-6, k3=Ts_k3, virtual_nodes=num_node, clear=True,
                                            C2C_strength=C2C_variation)
        State_ts = i_ts.reshape(len(target_ts), num_node)
        State_ts += noise_level * np.random.randn(State_ts.shape[0], State_ts.shape[1])
        Output_ts = lin.predict(State_ts)
        NRMSE_ts = nrmse(target_ts, Output_ts)
        NRMSE_ts_list.append(NRMSE_ts)

    return NRMSE_ts_list


def save_dict_nrmse_npz(dict_nrmse, Ts_k3_list, out_path="./Data/MG/dict_nrmse.npz"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = {"Ts_k3_list": np.asarray(Ts_k3_list, dtype=float)}
    for i_para, d in dict_nrmse.items():
        mat = np.vstack([d[float(t)] for t in Ts_k3_list])  # (25, repeat)
        payload[f"storage_i{i_para}"] = mat

    np.savez_compressed(out_path, **payload)
    return out_path

def load_npz_to_storage_and_dict(npz_path):
    z = np.load(npz_path)

    Ts_k3_list = z["Ts_k3_list"].astype(float)
    Ts_ref = 1.08e-5            # reference point
    k3_ratio = Ts_k3_list / Ts_ref
    labels = [f"{100*(r-1):.1f}" for r in k3_ratio]

    storages = {}   # storages[i_para] = (25, repeat)
    dicts = {}      # dicts[i_para][label] = (repeat,) array

    for key in z.files:
        if key.startswith("storage_i"):
            i_para = int(key.split("storage_i")[1])
            storage = np.asarray(z[key], dtype=float)   # (25, repeat)
            storages[i_para] = storage
            dicts[i_para] = {labels[k]: storage[k, :] for k in range(len(labels))}

    return Ts_k3_list, Ts_ref, labels, storages, dicts


def plot_box_no_dodge_with_mean_linesW(
    dicts,
    labels,
    Ts_k3_list=None,
    center=None,
    widths=None,
    bg_blue="#dbeeff",
    bg_red="#ffe0e0",
    bg_alpha=0.35,
    show=False
):
    """
    dicts[i_para][label] = (repeat,) array
    Ts_k3_list: shape (levels,), 与 label 顺序一一对应（同你生成 dict 的 enumerate 顺序）
    center: 例如 1.08e-5
    widths: 例如 [0.03e-5, 0.06e-5, 0.09e-5, 0.12e-5, 0.15e-5]
    """

    rows = []
    for i_para, d in dicts.items():
        for label, arr in d.items():
            for v in arr:
                rows.append((label, float(v), int(i_para)))
    df = pd.DataFrame(rows, columns=["k3_delta_pct", "NRMSE", "i_para"])

    df["k3_delta_pct_val"] = pd.to_numeric(df["k3_delta_pct"], errors="coerce")
    df["k3_abs"] = round(df["k3_delta_pct_val"].abs(), 2)

    # x 轴按绝对值升序排序（用数值排序）
    order_abs = sorted(df["k3_abs"].dropna().unique())

    # --------- font ----------
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    # --------- palette ----------
    hue_order = sorted(df["i_para"].unique())
    palette_list = sns.color_palette(n_colors=len(hue_order))
    palette = {h: palette_list[i] for i, h in enumerate(hue_order)}

    fig, ax = plt.subplots(figsize=(3, 2), dpi=300)

    # --------- boxplot：no dodging + thin frame ----------
    lw = 0.6
    sns.boxplot(
        data=df,
        x="k3_abs",
        y="NRMSE",
        hue="i_para",
        order=order_abs,
        hue_order=hue_order,
        dodge=False,
        width=0.55,
        palette=palette,
        fliersize=1.2,
        linewidth=lw,
        boxprops={"linewidth": lw},
        whiskerprops={"linewidth": lw},
        capprops={"linewidth": lw},
        medianprops={"linewidth": lw},
        ax=ax
    )

    # Transparency in case of overlapping
    for patch in ax.artists:
        patch.set_alpha(0.35)

    # --------- lineplot with mean value dots ----------
    mean_df = (df.groupby(["k3_abs", "i_para"], sort=False)["NRMSE"]
               .mean()
               .reset_index())

    x_pos = {lab: i for i, lab in enumerate(order_abs)}
    for h in hue_order:
        sub = mean_df[mean_df["i_para"] == h].copy()
        sub["x"] = sub["k3_abs"].map(x_pos)
        sub = sub.sort_values("x")

        ax.plot(
            sub["x"].to_numpy(),
            sub["NRMSE"].to_numpy(),
            marker="o",
            markersize=3.0,
            linewidth=0.8,
            color=palette[h],
            zorder=5,
        )

    # --------- More details ----------
    ax.set_xlabel("Test device D2D |Δ| (%)")
    ax.set_xticklabels([f"{x:.1f}" for x in order_abs], rotation=0)
    ax.set_ylabel("NRMSE")
    ax.set_yticks([0.15, 0.2, 0.25, 0.3, 0.35])
    ax.set_ylim([0.148, 0.352])

    # ---- legend: title + relabel ----
    width_labels = {0:"0 (classical)", 1: "3.7", 2: "7.4", 3: "11.1", 4: "14.8", 5: "18.5"}

    leg = ax.legend(
        title="Training width (% D2D)",
        frameon=False,
        loc="upper left",
        # bbox_to_anchor=(0.5, 1.02),
        ncol=1,
    )

    for t in leg.texts:
        s = t.get_text()
        try:
            k = int(float(s))
            if k in width_labels:
                t.set_text(width_labels[k])
        except Exception:
            pass

    plt.axhline(y=0.2, color="grey", linestyle="--")

    plt.tight_layout()
    plt.savefig('./Figure/MG/Sim_Robustness_boundary_training_width.svg', format="svg", bbox_inches="tight", dpi=300, transparent=True)
    if show:
        plt.show()
    plt.close(fig)


def NRMSE_sim_width(
        repeat=20
):


    widths = [0, 0.04e-5, 0.08e-5, 0.12e-5, 0.16e-5, 0.20e-5]
    Ts_k3_list = np.linspace(1.08e-5 - 0.24e-5, 1.08e-5 + 0.24e-5, 25)

    dict_nrmse = {}  # dict_nrmse[i_para][Ts_k3] = (repeat,) array

    for i_para, width in enumerate(widths):
        storage = np.column_stack([MG_SRC_sim_full_test(width=width) for _ in range(repeat)])
        dict_nrmse[i_para] = {float(t): storage[k, :] for k, t in enumerate(Ts_k3_list)}

    _ = save_dict_nrmse_npz(dict_nrmse, Ts_k3_list, out_path="./Data/MG/dict_nrmse_W.npz")


def NRMSE_sim_W_plot():
    npz_path = "./Data/MG/dict_nrmse_W.npz"
    Ts_k3_list, Ts_ref, labels, storages, dicts = load_npz_to_storage_and_dict(npz_path)

    plot_box_no_dodge_with_mean_linesW(
        dicts,
        Ts_k3_list=Ts_k3_list,
        labels=labels,
        center=1.08e-5,
        widths=[0, 0.04e-5, 0.08e-5, 0.012e-5, 0.16e-5, 0.20e-5],
    )


def add_pca_envelope_ellipse(
    ax, xg, yg,
    margin_major=1.05, margin_minor=1.20,   # Scaling factors along the major and minor axes, respectively (≥1 to ensure full coverage)
    dx=0.0, dy=0.0,                          # Allow slight translation for visual balance
    facecolor="#A9D6F5", alpha=0.22, zorder=0.2
):
    xg = np.asarray(xg); yg = np.asarray(yg)
    P = np.column_stack([xg, yg])

    # 1) PCA directions (from covariance eigenvectors)
    mu = P.mean(axis=0)
    C = np.cov(P.T)
    vals, vecs = np.linalg.eigh(C)
    order = vals.argsort()[::-1]
    vecs = vecs[:, order]  # vecs[:,0] direction of main axis

    # 2) PCA projection
    Z = (P - mu) @ vecs
    zmin = Z.min(axis=0)
    zmax = Z.max(axis=0)

    center_z = 0.5 * (zmin + zmax)
    r_major = 0.5 * (zmax[0] - zmin[0]) * margin_major
    r_minor = 0.5 * (zmax[1] - zmin[1]) * margin_minor

    center_xy = mu + vecs @ center_z
    center_xy = center_xy + np.array([dx, dy])

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    ell = Ellipse(
        xy=center_xy,
        width=2 * r_major,
        height=2 * r_minor,
        angle=angle,
        facecolor=facecolor,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder
    )
    ax.add_patch(ell)
    return ell

def _reshape_27_to_3x9(y, interleaved=False):
    y = np.asarray(y, dtype=float).ravel()
    assert y.size == 27, "Expect length-27 vector (3 trials × 9 indices)."
    if not interleaved:
        # [trial1, trial2, trial3] for all 9 devices
        return y.reshape(3, 9)
    else:
        return y.reshape(9, 3).T

def _center_and_band(Y, center="median", band="minmax"):
    # Y: (3,9)
    if center == "median":
        C = np.median(Y, axis=0)
    elif center == "mean":
        C = np.mean(Y, axis=0)
    else:
        raise ValueError("center must be 'median' or 'mean'")

    if band == "minmax":
        L = np.min(Y, axis=0)
        U = np.max(Y, axis=0)
    elif band == "std":
        s = np.std(Y, axis=0, ddof=1)
        L, U = C - s, C + s
    else:
        raise ValueError("band must be 'minmax' or 'std'")

    return C, L, U


def plot_gain_gap_one_figure(
    gain_vec_27, gap_vec_27, *,
    interleaved=False,
    center="median",
    band="minmax",
    epsilon=None,                   # e.g., 0.012
    show_zero_line=True,

    gain_line_color="C0",           
    gain_fill_alpha=0.25,
    gap_line_color=np.array([117, 185, 86]) / 255,       
    gap_fill_color=np.array([117, 185, 86]) / 255,       
    gap_fill_alpha=0.6
):
    GY = _reshape_27_to_3x9(gain_vec_27, interleaved=interleaved)
    DY = _reshape_27_to_3x9(gap_vec_27,  interleaved=interleaved)

    Gc, Gl, Gu = _center_and_band(GY, center=center, band=band)
    Dc, Dl, Du = _center_and_band(DY, center=center, band=band)

    x = np.arange(1, 10)

    plt.figure(figsize=(2.9, 2))

    # ===== gain =====
    plt.fill_between(
        x, Gl, Gu,
        color=gain_line_color, alpha=gain_fill_alpha,
        linewidth=0 
    )
    plt.plot(x, Gc, marker='o', markersize=5, linewidth=2.2,
             color=gain_line_color, label=r'Transfer gain by TS ($G_j$)')

    # ===== gap =====
    plt.fill_between(
        x, Dl, Du,
        color=gap_fill_color, alpha=gap_fill_alpha,
        linewidth=0  
    )
    plt.plot(x, Dc, marker='o', markersize=5, linewidth=2.2,
             color=gap_line_color, label=r'Gap to optimality ($\Delta_j$)')

    # reference line
    if show_zero_line:
        plt.axhline(0.0, linestyle="--", linewidth=1.0, color="k", alpha=0.7)
    if epsilon is not None:
        plt.axhline(epsilon, linestyle="--", linewidth=1.0, color="k", alpha=0.7, label='Tolerance margin')

    plt.ylim([-0.055, 0.155])
    plt.yticks([-0.05, 0, 0.05, 0.1, 0.15])
    plt.xticks(x, [str(i) for i in x])
    plt.xlabel("Device serial No.")
    plt.ylabel("NRMSE")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("./Figure/MG/Exp_all_device_gain_gap.svg", format="svg", dpi=300, transparent=True, bbox_inches="tight")
    plt.show()


def All_device_gain_gap():
    DT_record = np.zeros((9, 27))
    TS_record = np.zeros((1, 27))
    OP_record = np.zeros((1, 27))
    for i_mask in range(0, 3):

        for i_train in range(9):
            for i_test in range(9):
                _, DT_record[i_train, i_test + i_mask * 9], _ = MG_SRC_Expr(train_combo=(i_train,), test_index=i_test, mask_choice=i_mask+1)


        # Part II: run the TS framework on all test devices, with the identical train set

        for i_test in range(9):
            _, TS_record[0, i_test + i_mask * 9], _ = MG_SRC_Expr(train_combo=(0, 5, 8), test_index=i_test, mask_choice=i_mask+1)

        # Part III: run the optimal train on each test devices themselves

        for i_test in range(9):
            _, OP_record[0, i_test + i_mask * 9], _ = MG_SRC_Expr(train_combo=(i_test,), test_index=i_test, mask_choice=i_mask+1)

    ALL_RECORD = np.concatenate((DT_record, TS_record, OP_record), axis=0)

    np.savetxt("./Data/MG/Exp/all_device_gain_gap.csv", ALL_RECORD, delimiter=",", fmt="%.18e")  # 浮点推荐

    # =================================================================================================================
    # 读：loadtxt 默认就是 float
    ALL_RECORD = np.loadtxt("./Data/MG/Exp/all_device_gain_gap.csv", delimiter=",", dtype=float)

    DT_record = ALL_RECORD[:9, :]
    mean_DT = DT_record.mean(axis=0)
    var_DT = DT_record.var(axis=0)
    TS_record = ALL_RECORD[9:10, :]
    OP_record = ALL_RECORD[10:11, :]

    print(mean_DT)
    print(var_DT)
    print(TS_record)
    print(DT_record[-1, :] - TS_record)
    print(TS_record - OP_record)

    # =================================================================================================================
    direct_transfer_9 = DT_record[-1, :]
    temporal_switch = TS_record[0, :]
    optimals = OP_record[0, :]

    print('******', np.ptp(optimals.reshape((3, 9)), axis=0))
    print('Fluctuation of optimal is {}'.format(np.ptp(optimals.reshape((3, 9)), axis=0)))

    direct_transfer_9 = direct_transfer_9.reshape(3, 9)
    temporal_switch = temporal_switch.reshape(3, 9)
    optimals = optimals.reshape(3, 9)
    x = np.arange(1, 10)

    # Main plot, Fig. 4E
    plot_gain_gap_one_figure(direct_transfer_9 - temporal_switch, temporal_switch - optimals,
                             center='mean', epsilon=0.012)


    # Extra, optimality fluctuation
    optimals = OP_record[0, :9]
    num_list = [1, 2, 3, 4, 5, 6, 7, 8 ,9]

    plt.figure(figsize=(4, 3))
    plt.scatter(num_list, optimals, c='#2A9D8F', label='Per-device optimality')

    plt.axhline(optimals.max(), linestyle='dashed', color='k')
    plt.axhline(optimals.min(), linestyle='dashed', color='grey')
    plt.text(1.2, 0.16, 'Optimality fluctuation is {:.4f}'.format(optimals.max()-optimals.min()))

    plt.ylim([0.145, 0.168])
    plt.plot(num_list, optimals, c='#2A9D8F')
    plt.xlabel('Device serial No.')
    plt.ylabel('NRSME (MG prediction)')
    plt.legend(frameon=False, loc='best')
    plt.show()


    # Trade-off, Fig. 4F
    direct_transfer_9 = DT_record[-1, :]
    temporal_switch = TS_record[0, :]
    optimals = OP_record[0, :]

    x_scatter, y_scatter = direct_transfer_9 - temporal_switch, temporal_switch - optimals

    fig, ax = plt.subplots(figsize=(3, 2))

    idx = np.argsort(x_scatter)
    left_idx = idx[:4]
    right_idx = idx[4:]

    # --- 左边：更胖更短，且尽量别往右伸 ---
    add_pca_envelope_ellipse(
        ax, x_scatter[left_idx], y_scatter[left_idx],
        margin_major=1.2, margin_minor=5,  # 关键：主轴少一点、副轴多一点
        dx=0.001, dy=0.002,  # 轻微往左挪（按需要调成 -0.002~-0.008）
        facecolor=np.array([117, 185, 86]) / 255, alpha=0.25, zorder=0.2
    )

    # # --- 右边：更紧一点，并轻微右上平移 ---
    add_pca_envelope_ellipse(
        ax, x_scatter[right_idx], y_scatter[right_idx],
        margin_major=1.2, margin_minor=1.40,  # 更紧（接近1）
        dx=-0.002, dy=+0.002,  # 轻微右上移（按需要调成 0~0.008）
        facecolor=np.array([103, 149, 216]) / 255, alpha=0.22, zorder=0.2
    )

    rho_spm, p_spm = spearmanr(x_scatter[right_idx], y_scatter[right_idx])

    print('rho_spm is {}, p_spm is {}'.format(rho_spm, p_spm))

    # 散点
    ax.scatter(x_scatter, y_scatter, zorder=2)
    # 左边四个点：加深粉色（再画一遍覆盖）
    ax.scatter(
        x_scatter[left_idx], y_scatter[left_idx],
        s=55,  # 可调：点大小
        color=np.array([117, 185, 86]) / 255,  # 较重的粉色（可换）
        # edgecolors="white", linewidths=0.6,  # 可选：白色描边更醒目
        zorder=4
    )

    # ===== 只拟合右边点，并把趋势线向右上延长 =====
    xr = np.asarray(x_scatter)[right_idx]
    yr = np.asarray(y_scatter)[right_idx]

    m, b = np.polyfit(xr, yr, 1)

    x_span = xr.max() - xr.min()
    extend_ratio = 0.35  # 想更长就调大：0.2~0.8 都行
    xline = np.linspace(xr.min(), xr.max() + extend_ratio * x_span, 200)
    yline = m * xline + b

    ax.plot(
        xline, yline, linestyle="--", linewidth=2,
        label=rf"Linear fit (for $G_j>0$): y={m:.3g}x{b:.3g}", zorder=3

    )
    ax.set_ylabel(r'Gap to optimality ($\Delta_j$)')
    ax.set_xlabel(r'Transfer gain by TS ($G_j$)')
    ax.set_ylim([-0.012, 0.07])
    ax.set_yticks([0, 0.02, 0.04, 0.06])

    ax.legend(frameon=False)
    plt.savefig("./Figure/MG/Exp_gain_gap_trade_off.svg", format="svg", dpi=300, transparent=True, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':

    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Check whether the folder for storing figures is created
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

    # =================================== Part I: Basic demo of the TS framework ===================================
    # Only run the following for once, it saves all calculated data in file for the following
    MG_Expr_read_in()

    # FIGURE 3
    # For classical framework in experiment, Fig.3C
    MG_SRC_Expr(direct_transfer=True, no_pic=False)
    # For TS training framework in experiment, Fig.3E
    MG_SRC_Expr(direct_transfer=False, no_pic=False)

    # FIGURE S11
    # For classical framework in simulation, Fig.S11A
    MG_SRC_sim(direct_transfer=True)
    # For TS training framework in simulation, Fig.S11B
    MG_SRC_sim(direct_transfer=False)

    # FIGURE S12
    # For Fig.S12
    MG_SRC_Expr_MultiChannel(direct_transfer=True)  # Multichannel + classical framework, Fig. S12B
    MG_SRC_Expr_MultiChannel(direct_transfer=False)  # Multichannel + TS framework, Fig. S12D

    # =============================================== End of Part I ================================================
    #
    #
    #
    # ====================================== Part II: Performance robustness =======================================

    # FIGURE 4
    # For NRMSE-D2D relationship with experimental data, Fig. 4A
    NRMSE_expr()

    # NRMSE_sim() would take some time, please wait patiently (if you wish to speed up and just examine the function,
    # please set repeat as smaller values, such as 1)
    NRMSE_sim(repeat=20)  # Default repeat=20
    NRMSE_sim_plot()  # Fig. 4B

    # The following two set as readin=False for the first time;
    # Once the data file is generated, you may set as readin=True for faster plotting
    SRC_Num_TS_device(readin=False)  # Fig. 4C
    TS_robustness_boundary(readin=False)  # Fig. 4D

    All_device_gain_gap()  # Figs. 4E and F

    # NRMSE_sim_width() would take some time, please wait patiently (if you wish to speed up and just examine the function,
    # please set repeat as smaller values, such as 1)
    NRMSE_sim_width(repeat=20) # Default repeat=20
    NRMSE_sim_W_plot()  # Fig. S15

    # # For Fig. S16
    Extended_Data_TS_advantage_result()
    Extended_Data_TS_advantage_schematic()

    # =============================================== End of Part II =================================================
    #
    #
    #
    # ============================================ Part III: Other demos =============================================

    # Temporal permutation, Fig. S23
    MG_SRC_Expr_MultiChannel(direct_transfer=False, temporal_reorder=True)

    # Average comparison, Fig. S24
    MG_SRC_sim(direct_transfer=False, AVG=True)                     # Fig. S24A
    MG_SRC_sim(direct_transfer=False)                               # Fig. S24B
    MG_SRC_Expr_MultiChannel(direct_transfer=False, AVG=True)       # Fig. S24C
    MG_SRC_Expr_MultiChannel(direct_transfer=False)                 # Fig. S24D
    
    # Spatial permutation, Fig. S25
    MG_SRC_Expr_MultiChannel(direct_transfer=True, spatial_reorder=True)

    # =============================================== End of Part III ================================================

