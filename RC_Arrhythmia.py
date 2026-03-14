
"""
Train and evaluate reservoir computing in classical framework and in temporal-switch (TS) framework.
This is the file for all task demo of the arrhythmia detection. Corresponding to Figs. 5 & S17.

Zefeng Zhang, Research Institute of Intelligent Complex Systems, Fudan University

Last Updated: 2026/3/15
"""

from sim_RC_library import *


def ECG_SRC_sim(
        num_node=500, mask_abs=0.1,  # Default RC settings
        direct_transfer=False,
        noise_level=1e-6,
        no_pic=False,
        ridge_alpha=3e-9,
        mode='high',
        fig_suffix='',
        C2C_tr=0.01e-5, C2C_ts=0.01e-5,
        Ts_k3=1.16e-5
):
    data = io.loadmat('./Data/Arrhythmia/ECGdataset.mat')['dataset'][:1000, :, :]
    # DATA PREPROCESSING
    inputs = data[:, :, 0] / np.max(np.abs(data[:, :, 0]), axis=1).reshape((-1, 1))
    labels = data[:, :, 1:]
    signal, target = inputs.reshape((1, -1, 1))[0, :, :], labels.reshape((1, -1, 1))[0, :, :]
    mask = create_mask(num_node, abs_value=mask_abs)
    Input_tr, input_ts, Target_tr, target_ts = \
        signal_process(signal, target, mask, split=0.6)

    if not direct_transfer:  # TS training framework
        num_res = 3
        Tr_set_k3 = np.linspace(0.96e-5, 1.2e-5, num_res)
    else:  # direct transfer in classical framework
        num_res = 1
        if mode == 'high':  # Train 1.2, test 1.16
            Tr_set_k3 = np.linspace(1.2e-5, 1.2e-5, num_res)
        else:  # Train 0.96, test 1.16
            Tr_set_k3 = np.linspace(0.96e-5, 0.96e-5, num_res)

    # Create the SRC module
    SRC = TiOx_SRC()

    # Training
    State_tr = np.zeros((int(len(Target_tr) / num_res) * num_res, num_node))
    for i in range(num_res):
        _input = Input_tr[
                 i * int(len(Target_tr) / num_res) * num_node:(i + 1) * int(len(Target_tr) / num_res) * num_node]
        i_tr, g_tr, g0_tr = SRC.iterate_SRC(_input, 20e-6, k3=Tr_set_k3[i], virtual_nodes=num_node,
                                            C2C_strength=C2C_tr, clear=True)
        State_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), :] = \
            i_tr.reshape(int(len(Target_tr) / num_res), num_node)

    # Add noise
    State_tr += noise_level * np.random.randn(State_tr.shape[0], State_tr.shape[1])

    # Linear regression
    lin = Ridge(alpha=ridge_alpha)
    lin.fit(State_tr, Target_tr)
    Output_tr = lin.predict(State_tr)

    # Testing
    i_ts, g_ts, g0_ts = SRC.iterate_SRC(input_ts, 20e-6, k3=Ts_k3, virtual_nodes=num_node,
                                        C2C_strength=C2C_ts, clear=True)
    State_ts = i_ts.reshape(len(target_ts), num_node)
    State_ts += noise_level * np.random.randn(State_ts.shape[0], State_ts.shape[1])
    Output_ts = lin.predict(State_ts)

    NRMSE_tr, NRMSE_ts = nrmse(Target_tr, Output_tr), nrmse(target_ts, Output_ts)

    LEN = 50
    ACC = np.zeros((60, 5))
    TH_list = np.zeros(2)
    TH_box = np.arange(0.21, 0.8, 0.01)
    THS_box = np.arange(1, 6)
    j = 0

    for TH in TH_box:
        k = 0
        for THS in THS_box:
            Fout = np.heaviside(Output_tr.T[0, :].reshape(-1, LEN) - TH, 1)
            Fout = np.heaviside(np.sum(Fout, axis=1) - THS, 1)
            Ftar = np.max(Target_tr.T[0, :].reshape(-1, LEN), axis=1)
            Fbox = Fout - Ftar
            ACC[j, k] = len(Fbox[Fbox == 0]) / len(Fbox)
            k = k + 1
        j = j + 1
    print('Training Acc is {}'.format(np.max(ACC)))

    TH_candidate_list, THS_candidate_list = np.where(ACC == np.max(ACC))

    Acc_test = np.zeros(len(TH_candidate_list))

    for k in range(len(TH_candidate_list)):
        TH, THS = TH_box[TH_candidate_list[k]], THS_box[THS_candidate_list[k]]
        Fout = np.heaviside(Output_ts.T[0, :].reshape(-1, LEN) - TH, 1)
        Fout = np.heaviside(np.sum(Fout, axis=1) - THS, 1)
        Ftar = np.max(target_ts.T[0, :].reshape(-1, LEN), axis=1)
        Fbox = Fout - Ftar
        Acc_test[k] = len(Fbox[Fbox == 0]) / len(Fbox)

    print('Test Acc is {}'.format(np.max(Acc_test)))
    Choice = np.where(Acc_test == np.max(Acc_test))[0][0]

    print('TH is {}, THS is {}'.format(TH_box[TH_candidate_list[Choice]], THS_box[THS_candidate_list[Choice]]))
    TH_choice = TH_box[TH_candidate_list[Choice]]
    print(np.max(Acc_test))

    print('NRMSE tr is {}'.format(NRMSE_tr))
    print('NRMSE ts is {}'.format(NRMSE_ts))

    if not no_pic:

        color4 = np.array([107, 158, 184]) / 255
        color3 = np.array([103, 149, 216]) / 255
        color2 = np.array([110, 167, 151]) / 255
        color1 = np.array([117, 185, 86]) / 255

        if direct_transfer:
            with h5py.File('./Data/Arrhythmia/storage_ECG_signal_classical_{}{}.h5'.format(mode, fig_suffix), 'w') as wfile:
                wfile.create_dataset('Target_tr', data=Target_tr)
                wfile.create_dataset('Output_tr', data=Output_tr)
                wfile.create_dataset('target_ts', data=target_ts)
                wfile.create_dataset('Output_ts', data=Output_ts)
        else:
            with h5py.File('./Data/Arrhythmia/storage_ECG_signal_TS{}.h5'.format(fig_suffix), 'w') as wfile:
                wfile.create_dataset('Target_tr', data=Target_tr)
                wfile.create_dataset('Output_tr', data=Output_tr)
                wfile.create_dataset('target_ts', data=target_ts)
                wfile.create_dataset('Output_ts', data=Output_ts)

        figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 0.6), sharey='row', sharex='col')

        plt.rc('font', family='Arial', size=6)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['lines.linewidth'] = 1.2

        # Subplot1
        if direct_transfer:
            ax1.axvline(300, ls='--', color=np.array([180, 180, 180]) / 255)
            ax1.axvline(600, ls='--', color=np.array([180, 180, 180]) / 255)
            ax1.plot(np.arange(0, 900), Target_tr[0:900, 0], color=np.array([200, 200, 200]) / 255)
            if mode == 'high':
                Color_tr = color3
            else:
                Color_tr = color1
            ax1.plot(np.arange(0, 900), Output_tr[0:900, 0], color=Color_tr)

        else:
            colors = [color1, color2, color3]
            ax1.axvline(300, ls='--', color=np.array([180, 180, 180]) / 255)
            ax1.axvline(600, ls='--', color=np.array([180, 180, 180]) / 255)
            ax1.plot(np.arange(600, 900), Target_tr[0:300, 0], color=np.array([80, 80, 80]) / 255)
            ax1.plot(np.arange(300, 600),
                     Target_tr[1 * int(len(Target_tr) / num_res):1 * int(len(Target_tr) / num_res) + 300, 0],
                     color=np.array([80, 80, 80]) / 255)
            ax1.plot(np.arange(0, 300),
                     Target_tr[2 * int(len(Target_tr) / num_res):2 * int(len(Target_tr) / num_res) + 300, 0],
                     color=np.array([80, 80, 80]) / 255)
            for i in range(num_res):
                color = colors[i % num_res]
                ax1.plot(np.arange((2- i) * 300, (3 - i) * 300),
                         Output_tr[i * int(len(Target_tr) / num_res):i * int(len(Target_tr) / num_res) + 300, 0],
                         color=color)
        ax1.set_xlim(0, 900)
        ax1.set_xlabel('Time step', fontdict={'family': 'arial', 'size': 6}, labelpad=1)
        ax1.set_ylabel('Output value', fontdict={'family': 'arial', 'size': 6})
        ax1.tick_params(axis='both', direction='in', labelsize=6)
        ax1.set_xticks([0, 300, 600, 900])
        ax1.axhline(TH_choice, color=np.array([218, 69, 131]) / 255, linestyle='--')

        # Subplot2
        ax2.plot(np.arange(900, 1800), target_ts[1800:2700, 0], color=np.array([200, 200, 200]) / 255)
        ax2.plot(np.arange(900, 1800), Output_ts[1800:2700, 0], color=color4)
        ax2.axhline(TH_choice, color=np.array([218, 69, 131]) / 255, linestyle='--')
        ax2.set_xlim(900, 1800)
        ax2.set_xticks([1200, 1500, 1800])
        ax2.tick_params(axis='both', direction='in', labelsize=6)
        figure.subplots_adjust(wspace=0, hspace=0.1)

        if not direct_transfer:
            plt.savefig('./Figure/Arrhythmia/Sim_ECG_SRC{}.svg'.format(fig_suffix), dpi=300,
                        format='svg', transparent=True, bbox_inches='tight')
        else:
            plt.savefig('./Figure/Arrhythmia/Sim_ECG_DT_{}{}.svg'.format(mode, fig_suffix), dpi=300,
                        format='svg', transparent=True, bbox_inches='tight')

        plt.show()

    else:
        return np.max(Acc_test), NRMSE_tr, NRMSE_ts


def compare_src_regularization_ECG(
        rounds=20, mode='high', num_node=400
):
    # Compare the effect of SRC and other normal regularization methods, given other parameters the same among
    # different trials

    L2_penalty_sessions = []
    noise_sessions = []

    Types = ['SRC', 'L2 penalty']
    if mode == 'high':
        L2_penalty_sessions = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
    else:
        L2_penalty_sessions = [5e-4, 1e-3, 2e-3, 4e-3]

    Full_Storage_tr = np.zeros((1 + len(L2_penalty_sessions) + len(noise_sessions), rounds))
    Full_Storage_ts = np.zeros((1 + len(L2_penalty_sessions) + len(noise_sessions), rounds))
    Full_Storage_acc = np.zeros((1 + len(L2_penalty_sessions) + len(noise_sessions), rounds))
    Dict_tr = {}
    Dict_ts = {}
    Dict_acc = {}

    for typename in Types:
        if typename == 'SRC':
            for i in range(rounds):
                Full_Storage_acc[0, i], Full_Storage_tr[0, i], Full_Storage_ts[0, i] = ECG_SRC_sim(
                    noise_level=1e-6, direct_transfer=False, num_node=num_node,
                    ridge_alpha=0,
                    no_pic=True
                )

                print('%%%%%%%%%%%%%%%%%%%%%%%%%  Finished SRC Round {}  %%%%%%%%%%%%%%%%%%%%'.format(i + 1))
            Dict_acc['SRC'] = Full_Storage_acc[0, :]
            Dict_tr['SRC'] = Full_Storage_tr[0, :]
            Dict_ts['SRC'] = Full_Storage_ts[0, :]

        elif typename == 'L2 penalty':
            for j in range(1, 1 + len(L2_penalty_sessions)):
                for i in range(rounds):
                    Full_Storage_acc[j, i], Full_Storage_tr[j, i], Full_Storage_ts[j, i] = ECG_SRC_sim(
                        noise_level=1e-6, direct_transfer=True, num_node=num_node,
                        no_pic=True, mode=mode,
                        ridge_alpha=L2_penalty_sessions[j - 1]
                    )

                    print('%%%%%%%%%%%%%%%%%%%%%%  Finished L2 Round {}  %%%%%%%%%%%%%%%%%%%%'.format(
                        i + 1 + rounds * (j - 1)))
                Dict_tr[r'$\lambda$={}'.format(L2_penalty_sessions[j - 1])] = Full_Storage_tr[j, :]
                Dict_ts[r'$\lambda$={}'.format(L2_penalty_sessions[j - 1])] = Full_Storage_ts[j, :]
                Dict_acc[r'$\lambda$={}'.format(L2_penalty_sessions[j - 1])] = Full_Storage_acc[j, :]

        elif typename == 'noise':
            for j in range(1 + len(L2_penalty_sessions), 1 + len(noise_sessions)+ len(L2_penalty_sessions)):
                for i in range(rounds):
                    Full_Storage_acc[j, i], Full_Storage_tr[j, i], Full_Storage_ts[j, i] = ECG_SRC_sim(
                        noise_level=noise_sessions[j - 1 - len(L2_penalty_sessions)], direct_transfer=True, num_node=num_node,
                        no_pic=True, mode=mode,
                        ridge_alpha=0
                    )
                    print('noise level is {}'.format(noise_sessions[j-1 - len(L2_penalty_sessions)]))
                    print('%%%%%%%%%%%%%%%%%%%%%%  Finished Noise Round {}  %%%%%%%%%%%%%%%%%%%%'.format(
                        i + 1 + rounds * (j - 1 - len(L2_penalty_sessions))))

                Dict_tr[r'$\lambda$={}'.format(noise_sessions[j - 1 - len(L2_penalty_sessions)])] = Full_Storage_tr[j, :]
                Dict_ts[r'$\lambda$={}'.format(noise_sessions[j - 1 - len(L2_penalty_sessions)])] = Full_Storage_ts[j, :]
                Dict_acc[r'$\lambda$={}'.format(noise_sessions[j - 1 - len(L2_penalty_sessions)])] = Full_Storage_acc[j, :]

    with open('./Data/Arrhythmia/Compare_SRC_Regularize_TR_ECG_{}.csv'.format(mode), mode='w',
              encoding='UTF-8', newline='') as f_tr:
        writer = csv.writer(f_tr)
        writer.writerows(Full_Storage_tr)

    with open('./Data/Arrhythmia/Compare_SRC_Regularize_TS_ECG_{}.csv'.format(mode), mode='w',
              encoding='UTF-8', newline='') as f_ts:
        writer = csv.writer(f_ts)
        writer.writerows(Full_Storage_ts)

    with open('./Data/Arrhythmia/Compare_SRC_Regularize_ACC_ECG_{}.csv'.format(mode), mode='w',
              encoding='UTF-8', newline='') as f_acc:
        writer = csv.writer(f_acc)
        writer.writerows(Full_Storage_acc)

    pass


def compare_src_regularization_ECG_plot(
        rounds=20, mode='high'
):
    # Compare the effect of SRC and other normal regularization methods, given other parameters the same among
    # different trials

    Types = ['Temporal switch', 'L2 penalty']

    if mode == 'high':
        L2_penalty_sessions = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
    else:
        L2_penalty_sessions = [5e-4, 1e-3, 2e-3, 4e-3]


    Full_Storage_acc = pd.read_csv(
        './Data/Arrhythmia/Compare_SRC_Regularize_ACC_ECG_{}.csv'.format(mode), header=None).values
    Full_Storage_acc = Full_Storage_acc * 100
    Dict_acc = {}

    color3 = np.array([103, 149, 216]) / 255
    color2 = np.array([110, 167, 151]) / 255
    color1 = np.array([117, 185, 86]) / 255
    colors = []
    for i in range(len(L2_penalty_sessions)):
        if i <= (len(L2_penalty_sessions) / 2):
            color = color1 + (color2 - color1) / ((len(L2_penalty_sessions) - 1) / 2) * i
        else:
            color = color2 + (color3 - color2) / ((len(L2_penalty_sessions) - 1) / 2) * (i - ((len(L2_penalty_sessions) - 1) / 2))
        colors.append(color)

    for typename in Types:
        if typename == 'Temporal switch':
            Dict_acc['TS'] = Full_Storage_acc[0, :]

        elif typename == 'L2 penalty':
            for j in range(1, 1 + len(L2_penalty_sessions)):
                Dict_acc[r'$\lambda=${}'.format(L2_penalty_sessions[j-1])] = Full_Storage_acc[j, :]

    plt.figure(figsize=(4.8, 2))
    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    sns.boxplot(data=pd.DataFrame(Dict_acc).iloc[:, :], palette=colors, fliersize=1.5, saturation=1, width=0.7,
                boxprops={'linewidth': 0.5}, whiskerprops={'linewidth': 0.5}, medianprops={'linewidth': 0.5},
                capprops={'linewidth': 0.5}, flierprops={'marker': 'o'})
    plt.xticks(rotation=30)
    plt.ylabel('Accuracy (%)')
    plt.ylim(50, 100)
    if mode == 'high':
        plt.savefig('./Figure/Arrhythmia/TS_Penalty_Compare_ecg_{}.svg'.format(mode), dpi=300, format='svg',
                    transparent=True, bbox_inches='tight')
    else:
        plt.savefig('./Figure/Arrhythmia/TS_Penalty_Compare_ecg_{}.svg'.format(mode), dpi=300, format='svg',
                    transparent=True, bbox_inches='tight')
    # plt.show()
    plt.close()



if __name__ == '__main__':

    # Check whether the folder for storing figures is created
    fig_dir = './Figure/Arrhythmia'
    if not os.path.exists(fig_dir):
        print('Creating new figure file directory...')
        os.makedirs(fig_dir)

    data_dir = './Data/Arrhythmia'
    if not os.path.exists(data_dir):
        print('Creating new data file directory...')
        os.makedirs(data_dir)

    # The output signals, smaller num_node brings less run time, but results in the loss of output signal precision
    # You may change the region of testing phase, which would result in different presentation of the output signal

    ECG_SRC_sim(num_node=400, direct_transfer=True, mode='high')  # classical framework, train 1.2 test 1.16, Fig.5C up
    ECG_SRC_sim(num_node=400, direct_transfer=True, mode='low')  # classical framework, train 0.96 test 1.16, Fig.S17A up
    ECG_SRC_sim(num_node=400, direct_transfer=False)  # TS framework, train [0.96, 1.08, 1.2] test 1.16, Fig.5C & S9A down


    rounds = 20
    compare_src_regularization_ECG(mode='high',      rounds=rounds)
    compare_src_regularization_ECG(mode='low',       rounds=rounds)
    compare_src_regularization_ECG_plot(mode='high', rounds=rounds)
    compare_src_regularization_ECG_plot(mode='low',  rounds=rounds)


