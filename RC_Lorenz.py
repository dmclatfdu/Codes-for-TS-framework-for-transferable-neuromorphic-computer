
"""
Train and evaluate reservoir computing in classical framework and in temporal-switch (TS) framework.
This is the file for the task demo based on Lorenz63 prediction task. Corresponding to Fig. 13.

Zefeng Zhang, Research Institute of Intelligent Complex Systems, Fudan University

Last Updated: 2026/3/15
"""

from sim_RC_library import *


def Lorenz_SRC_k3(
        direct_transfer=False, warmup_before_pred=200, num_node=200, num_res=3, Ts_k3=1.16e-5, noise_level=0,
        C2C_training=0, fig_suffix=''
):
    total_len = 2400
    Lorenz_Gen = Lorenz_generator(length=total_len)
    Input, Target = Lorenz_Gen.series()
    mask = create_mask(num_node, abs_value=0.1, in_dim=3)
    voltage_range = np.array([2, 2.5])

    # Create the SRC module
    SRC = TiOx_SRC()

    if not direct_transfer:
        start = 800
        end = 1200
        warmup_end = end + warmup_before_pred
        input_episode_train = Input[start:end, :]
        input_warmup = Input[end:warmup_end, :]
        Target_tr = np.tile(Target[start:end, :], (num_res, 1))
        Target_ts = Target[end:, :]
    else:
        start = 0
        end = 1200
        warmup_end = end + warmup_before_pred
        input_episode_train = Input[start:end, :]
        input_warmup = Input[end:warmup_end, :]
        Target_tr = Target[start:end, :]
        Target_ts = Target[end:, :]
        num_res = 1

    Input_tr_masked = np.dot(input_episode_train, mask)
    Input_warmup_masked = np.dot(input_warmup, mask)
    Max, Min = np.max(Input_tr_masked), np.min(Input_tr_masked)
    Input_tr_masked_scaled = \
        (Input_tr_masked - (Max + Min) / 2) / (Max - Min) * (voltage_range[1] - voltage_range[0]) \
        + (voltage_range[0] + voltage_range[1]) / 2
    Input_warmup_masked_scaled = \
        (Input_warmup_masked - (Max + Min) / 2) / (Max - Min) * (voltage_range[1] - voltage_range[0]) \
        + (voltage_range[0] + voltage_range[1]) / 2
    Input_tr = Input_tr_masked_scaled.flatten()
    Input_warmup = Input_warmup_masked_scaled.flatten()

    # Training

    Tr_set_k3 = np.linspace(0.96e-5, 1.2e-5, num_res)
    State_tr = np.zeros((int(len(Target_tr) / num_res) * num_res, num_node))
    for i in range(num_res):
        i_tr, g_tr, g0_tr = SRC.iterate_SRC(Input_tr, 20e-6, k3=Tr_set_k3[i], virtual_nodes=num_node,
                                            C2C_strength=C2C_training,
                                            clear=True)
        State_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), :] = \
            i_tr.reshape(int(len(Target_tr) / num_res), num_node)

    # Training of the output weights
    lin = Ridge(alpha=1e-16)
    State_tr += noise_level * np.random.randn(State_tr.shape[0], State_tr.shape[1])
    lin.fit(State_tr, Target_tr)
    Output_tr = lin.predict(State_tr)

    # Testing/Continuous predicting
    test_len = total_len - warmup_end
    X_ts = np.zeros((test_len + warmup_before_pred, num_node))

    # Warm-up

    i_wm, _, _ = SRC.iterate_SRC(Input_warmup, 20e-6, k3=Ts_k3, virtual_nodes=num_node,
                                 C2C_strength=0,
                                 clear=True)
    X_ts[:warmup_before_pred, :] = i_wm.reshape(warmup_before_pred, num_node)
    X_ts += noise_level * np.random.randn(X_ts.shape[0], X_ts.shape[1])
    Transient_pred = lin.predict(np.array([X_ts[warmup_before_pred - 1, :]]))

    for j in range(warmup_before_pred, warmup_before_pred + test_len):
        # input and masking
        Input_ts_trans = np.dot(Transient_pred, mask)
        Input_ts_masked_scaled = \
            (Input_ts_trans - (Max + Min) / 2) / (Max - Min) * (voltage_range[1] - voltage_range[0]) \
            + (voltage_range[0] + voltage_range[1]) / 2
        Input_TiOx_ts = Input_ts_masked_scaled.flatten()

        # voltage range limiting, in case of extremely large input to incur infinity,
        # which is also not feasible in physical world, hard boundary [-5V, +5V]
        Input_TiOx_ts = np.minimum(Input_TiOx_ts, np.ones(Input_TiOx_ts.shape)*5)
        Input_TiOx_ts = np.maximum(Input_TiOx_ts, np.ones(Input_TiOx_ts.shape)*(-5))

        # Ts_k3_v = Ts_k3 + 0.01e-5 * np.random.randn()
        Ts_k3_v = Ts_k3
        i_ts, g_ts, g0_ts = SRC.iterate_one_step(Input_TiOx_ts, 20e-6, Ts_k3_v)

        X_ts[j, :] += i_ts

        Transient_pred = lin.predict(np.array([X_ts[j, :]]))

    Output_ts = lin.predict(X_ts)

    # Plotting
    # Draw Figure for paper
    channels = ['x', 'y', 'z']

    color4 = np.array([107, 158, 184]) / 255
    color3 = np.array([103, 149, 216]) / 255
    color2 = np.array([110, 167, 151]) / 255
    color1 = np.array([117, 185, 86]) / 255

    for channel in range(3):
        figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 1.8), sharey=True)

        plt.rc('font', family='Arial', size=6)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['lines.linewidth'] = 1

        # Subplot1
        if direct_transfer:
            ax1.plot(np.arange(0, len(Output_tr[:, channel])), Output_tr[:, channel], label='Training Error',
                     color=color1)
        else:
            slice_len = end - start
            colors = [color1, color2, color3]
            ax1.axvline(start - slice_len, ls='--', color=np.array([180, 180, 180]) / 255)
            ax1.axvline(start, ls='--', color=np.array([180, 180, 180]) / 255)
            for i in range(num_res):
                color = colors[i % num_res]
                ax1.plot(np.arange(i * slice_len, (i + 1) * slice_len),
                         Output_tr[i * slice_len:(i + 1) * slice_len, channel],
                         label='Training ',
                         color=color)

        ax1.set_xlim(0, len(Output_tr[:, channel]))
        ax1.set_ylim(min(min(Output_ts[200:, channel]), min(Output_tr[:, channel])) - 2,
                     max(max(Output_ts[200:, channel]), max(Output_tr[:, channel])) + 2)

        if channel == 0:  # x channel for demonstration
            ax1.set_ylim(-22, 22)
            ax1.set_yticks([-20, -10, 0, 10, 20])
        else:
            pass
        # else:  # z channel
        #     ax1.set_ylim(min(Output_ts[200:, channel])-2, max(Output_ts[200:, channel])+2)
        #     # ax1.set_yticks([10, 20, 30, 40, 50])

        ax1.set_xticks([400 * i for i in range(1 + int(end / 400))])
        ax1.set_xlabel('Time step', fontdict={'family': 'arial', 'size': 6})
        ax1.set_ylabel('{} value'.format(channels[channel]), fontdict={'family': 'arial', 'size': 6})
        ax1.tick_params(axis='both', direction='in', labelsize=6)

        # Subplot2
        ax2.plot(np.arange(len(Output_tr[:, channel]), len(Output_tr[:, channel]) + len(Output_ts[:, channel])),
                 Target_ts[:, channel], label='Testing error', color=np.array([180, 180, 180]) / 255)
        ax2.plot(np.arange(len(Output_tr[:, channel]), len(Output_tr[:, channel]) + len(Output_ts[:, channel])),
                 Output_ts[:, channel], label='Testing error', color=color4)
        ax2.fill_betweenx(ax2.get_ylim(), end, warmup_end, color='grey', alpha=0.25)
        ax2.set_xlim(len(Output_tr[:, channel]), len(Output_tr[:, channel]) + len(Output_ts[:, channel]))
        ax2.set_xticks([400 * i for i in range(1 + int(end / 400), 1 + int(total_len / 400))])
        ax2.tick_params(axis='both', direction='in', labelsize=6)
        figure.subplots_adjust(wspace=0)

        if direct_transfer:
            plt.savefig('./Figure/Lorenz/Sim_Lorenz_{}_DT{}.svg'.format(channels[channel], fig_suffix),
                        dpi=300, format='svg',
                        transparent=True, bbox_inches='tight')
        else:
            plt.savefig('./Figure/Lorenz/Sim_Lorenz_{}_SRC{}.svg'.format(channels[channel], fig_suffix),
                        dpi=300, format='svg',
                        transparent=True, bbox_inches='tight')

        plt.show()

    # Attractor plot
    fig3d = plt.figure(figsize=(2, 2))
    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    ax = fig3d.add_subplot(111, projection='3d')
    ax.plot(Output_ts[200:, 0], Output_ts[200:, 1], Output_ts[200:, 2], color=color4)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_zticks([0, 10, 20, 30, 40])
    ax.set_xlim([-22, 22])
    ax.set_ylim([-22, 22])
    ax.set_zlim([-2, 42])
    ax.set_xlabel(r'$x$', labelpad=0.2)
    ax.set_ylabel(r'$y$', labelpad=0.2)
    ax.set_zlabel(r'$z$', labelpad=0.2)

    if direct_transfer:
        plt.savefig('./Figure/Lorenz/Sim_attractor_DT{}.svg'.format(fig_suffix), dpi=300, format='svg',
                    transparent=True, bbox_inches='tight')
    else:
        plt.savefig('./Figure/Lorenz/Sim_attractor_SRC{}.svg'.format(fig_suffix), dpi=300, format='svg',
                    transparent=True, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':

    # Check whether the folder for storing figures is created
    fig_dir = './Figure/Lorenz'
    if not os.path.exists(fig_dir):
        print('Creating new figure file directory...')
        os.makedirs(fig_dir)

    data_dir = './Data/Lorenz'
    if not os.path.exists(data_dir):
        print('Creating new data file directory...')
        os.makedirs(data_dir)

    noise = 0

    Lorenz_SRC_k3(direct_transfer=True, noise_level=noise)  # For classical framework, Figs. S13B & C, may occur divergence problem
    Lorenz_SRC_k3(direct_transfer=False, noise_level=noise)  # For the TS training framework, Fig. S13E & F



