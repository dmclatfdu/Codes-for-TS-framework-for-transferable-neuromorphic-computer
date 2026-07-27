
"""
This is the file for all the device characterization analysis.

Corresponding to Figs. 2, S7, S8, S9 & S10.

Zefeng Zhang, Research Institute of Intelligent Complex Systems, Fudan University

Last Updated: 2026/3/15
"""

import matplotlib.pyplot as plt
import pandas as pd

from base_library import *
from sim_RC_library import TiOx_SRC


def Draw_TiOx_Pulses_Stat(
        rounds=50, figdir=None
):
    Data1 = {}
    Data2 = {}
    device_serial = ['4u', '6u', '7u', '8u', '9u', '11u', '14d', '15d', '16d']  # For 5um devices
    deltaI1_record = np.zeros(9)

    # for i in range(devices):
    for i in range(len(device_serial)):
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
        delta_I_2 = I_0 - I_2

        Data1[r'{}'.format(i+1)] = delta_I_1*1e6
        Data2[r'{}'.format(i+1)] = delta_I_2*1e6

        deltaI1_record[i] = np.round(np.average(delta_I_1*1e6), 2)

    color3 = np.array([103, 149, 216])/255
    color2 = np.array([110, 167, 151])/255
    color1 = np.array([117, 185, 86])/255

    colors = []
    num_lines = 9
    for i in (deltaI1_record.argsort()).argsort():

        if i <= (num_lines/2):
            color = color1 + (color2 - color1) / ((num_lines-1) / 2) * i
        else:
            color = color2 + (color3 - color2) / ((num_lines-1) / 2) * (i-((num_lines-1)/2))
        colors.append(color)

    # Export the delta I1 statistics
    dfI1 = pd.DataFrame(Data1)
    dfI2 = pd.DataFrame(Data2)

    dfI1.to_csv('./Data/Characterization/TiOx_I1_st.csv', index=False, encoding='utf-8')

    plt.figure(figsize=(2.6, 2))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    sns.boxplot(data=dfI1.iloc[:, :], palette=colors, fliersize=1.5, saturation=1, width=0.7,
                boxprops={'linewidth':0.5}, whiskerprops={'linewidth':0.5}, medianprops={'linewidth':0.5},
                capprops={'linewidth':0.5}, flierprops={'marker':'o'})
    plt.ylabel(r'$\Delta I(t_1)$ ($\mu$A)')
    plt.xlabel('Device serial')
    plt.savefig(figdir + '/TiOx_Pulses_Stat_I1.svg', format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(4, 2.8))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    sns.boxplot(data=dfI2.iloc[:, :], palette=colors, fliersize=1.5, saturation=1, width=0.7,
                boxprops={'linewidth':0.5}, whiskerprops={'linewidth':0.5}, medianprops={'linewidth':0.5},
                capprops={'linewidth':0.5}, flierprops={'marker':'o'})
    plt.ylabel(r'$\Delta I(t_2)$ ($\mu$A)')
    plt.xlabel('Device serial')
    plt.savefig(figdir + '/TiOx_Pulses_Stat_I2.svg', format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()


def Draw_TiOx_Decay_Stat(
        rounds=20, figdir=None
):
    # collecting average current at each read pulse
    Dict = {}

    device_serial = ['4u', '6u', '7u', '8u', '9u', '11u', '14d', '15d', '16d']  # For 5um devices
    color3 = np.array([103, 149, 216])/255
    color2 = np.array([110, 167, 151])/255
    color1 = np.array([117, 185, 86])/255

    colors = []
    for i in np.array([2, 1, 4, 6, 5, 8, 3, 7, 0]):
        if i <= (9 / 2):
            color = color1 + (color2 - color1) / ((9 - 1) / 2) * i
        else:
            color = color2 + (color3 - color2) / ((9 - 1) / 2) * (i - ((9 - 1) / 2))
        colors.append(color)

    # Draw Fig2c
    plt.figure(figsize=(2, 1.8))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1

    for i in range(len(device_serial)):

        df = pd.read_csv(
            './Data/Characterization/TiOx/Decay/' + '5um_{}_Decay_W3R2_20.csv'.format(device_serial[i]),
            header=None, sep='\n'
        )
        df = df[0].str.split(',', expand=True)
        decay_time_record = []
        for j in range(rounds):
            time_point_record = []
            current_record = []
            for k in range(100):  # use 100 points in fitting decay time
                df_0 = df.iloc[178 + 9296 * j + 40 * k:189 + 9296 * j + 40 * k, 1:4]
                df_0_numpy = df_0.to_numpy()
                time = df_0_numpy[:, 0].astype(np.float64)
                current = df_0_numpy[:, 2].astype(np.float64)
                current_record.append(np.average(current))
                time_point_record.append(time[0])

            time_point_record = np.array(time_point_record)*1e2
            time_point_record = time_point_record-time_point_record[0]
            current_record = -np.array(current_record)*1e4

            # Computing the fitted decay time
            popt, pcov = curve_fit(exponential, time_point_record, current_record)
            decay_time = - 1 / popt[0]
            decay_time_record.append(decay_time)

            if j == 0:
                plt.scatter(time_point_record*1e4, current_record*100, label='Device {}'.format(i+1), s=2, color=colors[i])
                plt.plot(time_point_record*1e4, exponential(time_point_record, *popt)*100, color=colors[i])

        decay_time_record = np.array(decay_time_record) * 1e4
        Dict['{}'.format(i+1)] = decay_time_record

    # Export Decay statistics
    df_decay = pd.DataFrame(Dict)
    df_decay.to_csv('./Data/Characterization/TiOx_Decay_st.csv', index=False, encoding='utf-8')

    plt.xlim([time_point_record[0]*1e4, time_point_record[-1]*1e4])
    plt.xticks([0, 100, 200, 300, 400])
    plt.legend(frameon=False, loc='upper center', ncol=1, bbox_to_anchor=[1.2, 1.05], columnspacing=0.4,
               handletextpad=0.2)
    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel(r'Current ($\mu$A)')
    plt.savefig(figdir + '/TiOx_Nine_Decay.svg', format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

    # Draw Fig2f
    plt.figure(figsize=(2.6, 2))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    sns.boxplot(data=df_decay.iloc[:, :],  palette=colors, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth':0.5}, whiskerprops={'linewidth':0.5}, medianprops={'linewidth':0.5},
                capprops={'linewidth':0.5}, flierprops={'marker':'o'})
    plt.ylim([65, 85])
    plt.ylabel(r'$\tau$ ($\mu$s)')
    plt.xlabel('Device serial')
    plt.savefig(figdir + '/TiOx_Decay_Stat.svg', format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()


def Draw_TiOx_IV(
        rounds=1, figdir=None
):

    device_serial = ['4u', '6u', '7u', '8u', '9u', '11u', '14d', '15d', '16d']  # For 5um devices

    color3 = np.array([103, 149, 216])/255
    color2 = np.array([110, 167, 151])/255
    color1 = np.array([117, 185, 86])/255

    colors = []
    for i in np.array([2, 1, 4, 6, 5, 8, 3, 7, 0]):
        if i <= (9 / 2):
            color = color1 + (color2 - color1) / ((9 - 1) / 2) * i
        else:
            color = color2 + (color3 - color2) / ((9 - 1) / 2) * (i - ((9 - 1) / 2))
        colors.append(color)

    plt.figure(figsize=(2, 1.8))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1

    for i in range(len(device_serial)):

        df = pd.read_csv(
            './Data/Characterization/TiOx/IV/' + '5um_{}_iv_m1_25_50.csv'.format(device_serial[i]),
            header=None, sep='\n'
        )
        df = df[0].str.split(',', expand=True)

        for j in range(rounds):

            df_0 = df.iloc[254 + 456*j:455 + 456*j, 1:4]
            df_0_numpy = df_0.to_numpy()
            voltage = df_0_numpy[:, 0].astype(np.float64)
            current = df_0_numpy[:, 2].astype(np.float64)
            if j == 0:
                plt.semilogy(voltage, current, color=colors[i], alpha=0.8, label='Device {}'.format(i+1))
            else:
                plt.semilogy(voltage, current, color=colors[i], alpha=0.3)

    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.savefig(figdir + '/TiOx_IV.svg', format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()


def Draw_TiOx_Pulse(figdir=None):
    dt_pulse = 1e-6
    device_serials = ['4u', '6u', '7u', '8u', '9u', '11u', '14d', '15d', '16d']
    V_pulse = 0.5 * np.ones(2500)
    V_pulse[100:300] = 3
    V_pulse[600:800] = 3
    V_pulse[1100:1300] = 3
    V_pulse[1600:1800] = 3
    V_pulse[2100:2300] = 3

    color3 = np.array([103, 149, 216])/255
    color2 = np.array([110, 167, 151])/255
    color1 = np.array([117, 185, 86])/255

    colors = []
    for i in np.array([2, 1, 4, 6, 5, 8, 3, 7, 0]):
        if i <= (9 / 2):
            color = color1 + (color2 - color1) / ((9 - 1) / 2) * i
        else:
            color = color2 + (color3 - color2) / ((9 - 1) / 2) * (i - ((9 - 1) / 2))
        colors.append(color)

    fig, ax1 = plt.subplots(figsize=(2, 1.8))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1

    font = {
        'family': 'Arial',
        'size': 8
    }
    ax2 = ax1.twinx()
    ax2.plot(np.arange(2500) * dt_pulse * 1e3, V_pulse, label='Voltage', color='grey')

    for i in range(len(device_serials)):
        device_serial = device_serials[i]
        df = pd.read_csv(
            './Data/Characterization/TiOx/Pulse/' + '5um_{}_Pulse_W3R0.5_50.csv'.format(device_serial),
            header=None, sep='\n')
        df = df[0].str.split(',', expand=True)
        df0 = df.iloc[148:398, 1:4]
        df_numpy = np.asarray(df0.astype(np.float64))
        # Plotting
        ax1.plot((df_numpy[:, 0] - 3) * 1e3, 1e6 * (-df_numpy[:, 2]), label='Device {}'.format(device_serial), color=colors[i])
    ax1.set_ylim([-50, 750])
    ax1.tick_params(axis='y', direction='in', labelsize=8)

    ax2.set_ylim([9 / 28, 3 + 5 / 28])

    ax2.set_xlim([0, 2.5])
    ax2.tick_params(axis='y', direction='in', labelsize=8)
    ax1.tick_params(axis='x', direction='in', labelsize=8)
    ax1.set_xlabel('Time (ms)', fontdict=font)
    ax1.set_ylabel(r'Current ($\mu$A)', fontdict=font)
    ax2.set_ylabel(r'Voltage (V)', fontdict=font)

    for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
        label.set_fontname('Arial')

    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=[0.8, 1.2], columnspacing=0.4, ncol=2,
               handletextpad=0.2)
    plt.savefig(figdir + '/TiOx_Pulse.svg', format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()


def Decay_Response_TiOx(figdir=None, **kwargs):
    # T1 = TiOx_SRC(**kwargs)
    # Decay test input setting
    V_decay = 0.5 * np.ones(1400)
    V_decay[1:1001] = 3
    for i in range(100):
        V_decay[1002 + i * 4:1004 + i * 4] = 2

    # Time step
    dt_decay = 1e-6
    df = pd.read_csv(
        './Data/Characterization/TiOx/Decay/' + '5um_{}_Decay_W3R2_20.csv'.format('6u'),
        header=None, sep='\n'
    )
    df = df[0].str.split(',', expand=True)
    df_0 = df.iloc[178:189 + 4000, 1:4]
    df_0_numpy = df_0.to_numpy()
    current = -df_0_numpy[:, 2].astype(np.float64)

    # # Decay time constant fitting
    t_d = np.arange(100) * 4e-6
    popt, pcov = curve_fit(exponential, t_d, (1e6*current[:-12:40]))
    decay_time = -1 / popt[0]

    font = {
        'family': 'Arial',
        'size': 8
    }

    plt.figure(figsize=(2, 3))
    gs = gridspec.GridSpec(4, 4)
    ax1 = plt.subplot(gs[0:2, 0:4])
    ax2 = plt.subplot(gs[2:4, 0:4], sharex=ax1)
    ax1.get_xaxis().set_visible(False)

    color3 = np.array([103, 149, 216])/255
    color2 = np.array([110, 167, 151])/255
    color1 = np.array([117, 185, 86])/255


    ax1.plot(np.arange(1400) * dt_decay * 1e3, V_decay, color=color1)
    ax2.plot(np.arange(1000, 1400) * dt_decay * 1e3, 1e6 * current[:-12:10], color=color3)

    ax2.tick_params(axis='y', direction='in', labelsize=8)
    ax2.tick_params(axis='x', direction='in', labelsize=8)
    ax1.tick_params(axis='y', direction='in', labelsize=8)
    ax2.set_xlabel('Time (ms)', fontdict=font)
    ax2.set_ylabel(r'Current ($\mu$A)', fontdict=font)
    ax1.set_ylabel(r'Voltage (V)', fontdict=font)
    ax2.set_xlim([0, 1.4])
    ax1.set_ylim([0.5, 3.1])
    ax2.set_ylim([0, 700])
    for label in ax2.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
        label.set_fontname('Arial')
    plt.show()

    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    plt.figure(figsize=(3, 3))
    plt.scatter(1+1e3*t_d, 1e6*current[:-12:40], s=4, label='Experimental data')
    plt.plot(np.arange(1001, 1401) * dt_decay * 1e3,
             exponential(np.arange(400) * dt_decay, popt[0], popt[1], popt[2]), label='Fitting curve')
    plt.ylabel(r'Current ($\mu$A)')
    plt.xlabel('Time (ms)')
    plt.xlim([1, 1.4])
    plt.legend(frameon=False)
    plt.savefig(figdir + '/TiOx_Decay.svg', format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()


def Pulse_Response_TiOx(figdir=None, **kwargs):
    T1 = TiOx_SRC(**kwargs)
    device_serial = kwargs.get('device_serial', '4u')
    # The setting of the pulse test input
    V_pulse = 0.5 * np.ones(2500)
    V_pulse[100:300] = 3
    V_pulse[600:800] = 3
    V_pulse[1100:1300] = 3
    V_pulse[1600:1800] = 3
    V_pulse[2100:2300] = 3

    # time step
    dt_pulse = 1e-6

    # The model iterations and responses
    i_p, *g_p = T1.iterate_SRC(V_pulse, dt_pulse, C2C_strength=0)
    i_p += 0e-6 * np.random.randn(i_p.shape[0])

    # experiment measurement
    df = pd.read_csv('./Data/Characterization/TiOx/Pulse/5um_{}_Pulse_W3R0.5_50.csv'.format(device_serial),
                     header=None, sep='\n')
    df = df[0].str.split(',', expand=True)
    df0 = df.iloc[148:398, 1:4]
    df_numpy = np.asarray(df0.astype(np.float64))

    # Plotting
    plt.figure(figsize=(2.4, 2))
    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    plt.plot(np.arange(2500) * dt_pulse * 1e3, 1e6 * i_p, label='sim')
    plt.plot((df_numpy[:, 0] - 3) * 1e3, 1e6 * (-df_numpy[:, 2]), label='exp')

    plt.xlabel('Time (ms)')
    plt.ylabel(r'Current ($\mu$A)')
    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=[0.8, 1.15], columnspacing=0.4, ncol=2,
               handletextpad=0.2)
    plt.savefig(figdir + '/TiOx_{}_Pulse_Response.svg'.format(device_serial), format='svg', dpi=300, transparent=True,
                bbox_inches='tight')
    plt.show()

    if device_serial == '7u':
        plt.figure(figsize=(1.5, 2))
        gs = gridspec.GridSpec(3, 1)
        ax1 = plt.subplot(gs[0:1, 0])
        ax2 = plt.subplot(gs[1:3, 0], sharex=ax1)
        ax1.get_xaxis().set_visible(False)

        color3 = np.array([103, 149, 216]) / 255
        color2 = np.array([110, 167, 151]) / 255
        color1 = np.array([117, 185, 86]) / 255

        ax1.plot(np.arange(1000) * dt_pulse * 1e3, V_pulse[:1000], color=color1)
        ax2.plot((df_numpy[:100, 0] - 3) * 1e3, 1e6 * (-df_numpy[:100, 2]), color=color3)
        font = {
            'family': 'Arial',
            'size': 8
        }
        ax2.tick_params(axis='y', direction='in', labelsize=8)
        ax2.tick_params(axis='x', direction='in', labelsize=8)
        ax1.tick_params(axis='y', direction='in', labelsize=8)
        ax2.set_xlabel('Time (ms)', fontdict=font)
        ax2.set_ylabel(r'Current ($\mu$A)', fontdict=font)
        ax1.set_ylabel(r'Voltage (V)', fontdict=font)
        ax1.set_yticks([0, 1, 2, 3])
        ax2.set_xlim([0, 1])
        ax1.set_ylim([0, 3.5])
        ax2.set_ylim([-50, 700])

        for label in ax2.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
            label.set_fontname('Arial')
        plt.savefig(figdir + '/I1_I2_demo.svg'.format(device_serial), format='svg', dpi=300,
                    transparent=True,
                    bbox_inches='tight')
        plt.show()


def Nine_TiOx_Pulse_Response(figdir=None):
    device_serial_list = ['4u', '6u', '7u', '8u', '9u', '11u', '14d', '15d', '16d']
    k3_list = np.array([1.08, 1.06, 1.085, 1.085, 1.1, 1.2, 1.12, 1.16, 0.96])*1e-5

    for i in range(9):
        Pulse_Response_TiOx(figdir=figdir, device_serial=device_serial_list[i], k3=k3_list[i])
        '''
        Note: for device 7u (k3=1.085), the Fig.2d would also be plotted
        '''


def Draw_NbOx_Pulses_Stat(
        figdir, rounds=20,
):
    Data1 = {}
    Data2 = {}
    device_serial = ['a', 'b', 'c', 'd']  # For 5um devices
    deltaI1_record = np.zeros(4)

    # for i in range(devices):
    for i in range(len(device_serial)):
        I_0 = np.zeros(rounds)
        time_0 = np.zeros(rounds)
        I_1 = np.zeros(rounds)
        time_1 = np.zeros(rounds)
        I_2 = np.zeros(rounds)
        time_2 = np.zeros(rounds)
        df = pd.read_csv('./Data/Characterization/NbOx/Pulse/' + 'Device {}.csv'.format(device_serial[i]),
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
        delta_I_2 = I_0 - I_2

        Data1[r'{}'.format(i+1)] = delta_I_1*1e9  # in nA
        Data2[r'{}'.format(i+1)] = delta_I_2*1e9  # in nA

        deltaI1_record[i] = np.round(np.average(delta_I_1*1e9), 2)

    # Export I1 statistics
    df_I1 = pd.DataFrame(Data1)
    df_I2 = pd.DataFrame(Data2)
    df_I1.to_csv('./Data/Characterization/NbOx_I1_st.csv', index=False, encoding='utf-8')

    color3 = np.array([103, 149, 216])/255
    color2 = np.array([110, 167, 151])/255
    color1 = np.array([117, 185, 86])/255

    colors = []
    num_lines = 4
    for i in (deltaI1_record.argsort()).argsort():

        if i <= (num_lines/2):
            color = color1 + (color2 - color1) / ((num_lines-1) / 2) * i
        else:
            color = color2 + (color3 - color2) / ((num_lines-1) / 2) * (i-((num_lines-1)/2))
        colors.append(color)

    plt.figure(figsize=(2, 1.8))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    sns.boxplot(data=df_I1.iloc[:, :], palette=colors, fliersize=1.5, saturation=1, width=0.7,
                boxprops={'linewidth':0.5}, whiskerprops={'linewidth':0.5}, medianprops={'linewidth':0.5},
                capprops={'linewidth':0.5}, flierprops={'marker':'o'})
    plt.ylabel(r'$\Delta I(t_1)$ (nA)')
    plt.xlabel('Device serial')
    plt.savefig(figdir + '/NbOx I1 pulse stat.svg', format='svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

    plt.figure(figsize=(4, 2.8))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    sns.boxplot(data=df_I2.iloc[:, :], palette=colors, fliersize=1.5, saturation=1, width=0.7,
                boxprops={'linewidth':0.5}, whiskerprops={'linewidth':0.5}, medianprops={'linewidth':0.5},
                capprops={'linewidth':0.5}, flierprops={'marker':'o'})
    plt.ylabel(r'$\Delta I(t_2)$ (nA)')
    plt.xlabel('Device serial')
    plt.savefig(figdir + '/NbOx I2 pulse stat.svg', format='svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def Draw_NbOx_Decay_Stat(
        figdir, rounds=10
):
    # collecting average current at each read pulse
    Dict = {}

    device_serial = ['a', 'b', 'c', 'd']  # For 5um devices
    color3 = np.array([103, 149, 216])/255
    color2 = np.array([110, 167, 151])/255
    color1 = np.array([117, 185, 86])/255
    colors = []
    for i in np.array([2, 1, 4, 6, 5, 8, 3, 7, 0]):
        if i <= (9 / 2):
            color = color1 + (color2 - color1) / ((9 - 1) / 2) * i
        else:
            color = color2 + (color3 - color2) / ((9 - 1) / 2) * (i - ((9 - 1) / 2))
        colors.append(color)

    # Draw Fig2c
    plt.figure(figsize=(2, 1.8))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1

    for i in range(len(device_serial)):

        df = pd.read_csv(
            './Data/Characterization/NbOx/Decay/' + 'Device {}.csv'.format(device_serial[i]),
            header=None, sep='\n'
        )
        df = df[0].str.split(',', expand=True)
        decay_time_record = []
        for j in range(rounds):
            time_point_record = []
            current_record = []
            for k in range(5, 100):
                df_0 = df.iloc[178 + 9296 * j + 40 * k:189 + 9296 * j + 40 * k, 1:4]
                df_0_numpy = df_0.to_numpy()
                time = df_0_numpy[:, 0].astype(np.float64)
                current = df_0_numpy[:, 2].astype(np.float64)
                current_record.append(np.average(current))
                time_point_record.append(time[0])

            time_point_record = np.array(time_point_record)*1e2
            time_point_record = time_point_record-time_point_record[0]
            current_record = -np.array(current_record)*1e4

            # Computing the fitted decay time
            popt, pcov = curve_fit(exponential, time_point_record, current_record)
            decay_time = - 1 / popt[0]
            decay_time_record.append(decay_time)

            if j == 0:
                plt.scatter(time_point_record*1e4, current_record*1e5, label='Device {}'.format(i+1), s=2, color=colors[i])
                plt.plot(time_point_record*1e4, exponential(time_point_record, *popt)*1e5, color=colors[i])

        decay_time_record = np.array(decay_time_record) * 1e4
        Dict['{}'.format(i+1)] = decay_time_record

    df_decay = pd.DataFrame(Dict)
    df_decay.to_csv('./Data/Characterization/NbOx_decay_st.csv', index=False)

    plt.xlim([time_point_record[0]*1e4, time_point_record[-1]*1e4])
    plt.xticks([0, 100, 200, 300, 400])
    plt.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=[0.45, 1.28], columnspacing=0.3,
               handletextpad=0.2)
    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel(r'Current (nA)')
    plt.savefig(figdir + '/NbOx devices decay.svg', format='svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

    # Draw Fig2f
    plt.figure(figsize=(2, 1.8))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    sns.boxplot(data=pd.DataFrame(Dict).iloc[:, :],  palette=colors, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth':0.5}, whiskerprops={'linewidth':0.5}, medianprops={'linewidth':0.5},
                capprops={'linewidth':0.5}, flierprops={'marker':'o'})
    plt.ylim([100, 160])
    plt.ylabel(r'$\tau$ ($\mu$s)')
    plt.xlabel('Device serial')
    plt.savefig(figdir + '/NbOx decay stat.svg', format='svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def Draw_NbOx_IV(figdir, num_devices=4):
    device_serial = ['a', 'b', 'c', 'd']

    color3 = np.array([103, 149, 216])/255
    color2 = np.array([110, 167, 151])/255
    color1 = np.array([117, 185, 86])/255

    colors = []
    for i in np.array([0, 1, 2, 3]):
        if i <= (4 / 2):
            color = color1 + (color2 - color1) / ((4 - 1) / 2) * i
        else:
            color = color2 + (color3 - color2) / ((4 - 1) / 2) * (i - ((4 - 1) / 2))
        colors.append(color)

    plt.figure(figsize=(2, 1.8))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1

    df = pd.read_csv(
        './Data/Characterization/NbOx/IV/IV.csv',
        header=None, sep='\n'
    )
    df = df[0].str.split(',', expand=True)

    for j in range(num_devices):

        df_0 = df.iloc[254 + 456*j:455 + 456*j, 1:4]
        df_0_numpy = df_0.to_numpy()
        voltage = df_0_numpy[:, 0].astype(np.float64)
        current = df_0_numpy[:, 2].astype(np.float64)
        if j == 0:
            plt.semilogy(voltage, current, color=colors[j], alpha=1, label='Device {}'.format(j+1))
        else:
            plt.semilogy(voltage, current, color=colors[j], alpha=0.5)

    plt.ylim([0.8e-11, 1.5e-5])
    plt.xlim([0, 4.2])
    plt.xticks([0, 1, 2, 3, 4])
    plt.yticks([1e-11, 1e-9, 1e-7, 1e-5])
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.savefig(figdir + '/NbOx IV.svg', format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()


def Draw_NbOx_Pulse(figdir):
    dt_pulse = 1e-6
    device_serials = ['a', 'b', 'c', 'd']
    V_pulse = 0.5 * np.ones(2500)
    V_pulse[100:300] = 2.5
    V_pulse[600:800] = 2.5
    V_pulse[1100:1300] = 2.5
    V_pulse[1600:1800] = 2.5
    V_pulse[2100:2300] = 2.5

    color3 = np.array([103, 149, 216])/255
    color2 = np.array([110, 167, 151])/255
    color1 = np.array([117, 185, 86])/255

    colors = []
    for i in np.array([0, 1, 2, 3]):
        if i <= (4 / 2):
            color = color1 + (color2 - color1) / ((4 - 1) / 2) * i
        else:
            color = color2 + (color3 - color2) / ((4 - 1) / 2) * (i - ((4 - 1) / 2))
        colors.append(color)

    fig, ax1 = plt.subplots(figsize=(2, 1.8))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1

    font = {
        'family': 'Arial',
        'size': 8
    }
    ax2 = ax1.twinx()
    ax2.plot(np.arange(2500) * dt_pulse * 1e3, V_pulse, label='Voltage', color='grey')

    for i in range(len(device_serials)):
        device_serial = device_serials[i]
        df = pd.read_csv(
            './Data/Characterization/NbOx/Pulse/' + 'Device {}.csv'.format(device_serial),
            header=None, sep='\n')
        df = df[0].str.split(',', expand=True)
        df0 = df.iloc[148:398, 1:4]
        df_numpy = np.asarray(df0.astype(np.float64))
        # Plotting
        ax1.plot((df_numpy[:, 0] - 10) * 1e3, 1e9 * (-df_numpy[:, 2]), label='Device {}'.format(device_serial), color=colors[i])
    ax1.set_ylim([-50, 500])
    ax1.tick_params(axis='y', direction='in', labelsize=8)

    ax2.set_ylim([9 / 28, 2.5 + 5 / 28])

    ax2.set_xlim([0, 2.5])
    ax2.tick_params(axis='y', direction='in', labelsize=8)
    ax1.tick_params(axis='x', direction='in', labelsize=8)
    ax1.set_xlabel('Time (ms)', fontdict=font)
    ax1.set_ylabel(r'Current (nA)', fontdict=font)
    ax2.set_ylabel(r'Voltage (V)', fontdict=font)

    for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
        label.set_fontname('Arial')

    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=[0.8, 1.2], columnspacing=0.4, ncol=2,
               handletextpad=0.2)

    plt.savefig(figdir + '/NbOx pulse.svg', format='svg', dpi=300, bbox_inches='tight', transparent=True)

    plt.show()


def Decay_Response_NbOx(figdir, **kwargs):

    # Decay test input setting
    V_decay = 0.5 * np.ones(1400)
    V_decay[1:1001] = 4
    for i in range(100):
        V_decay[1002 + i * 4:1004 + i * 4] = 2

    # Time step
    dt_decay = 1e-6

    df = pd.read_csv(
        './Data/Characterization/NbOx/Decay/' + 'Device a.csv',
        header=None, sep='\n'
    )
    df = df[0].str.split(',', expand=True)
    df_0 = df.iloc[178:189 + 4000, 1:4]
    df_0_numpy = df_0.to_numpy()
    current = -df_0_numpy[:, 2].astype(np.float64)

    # # Decay time constant fitting
    t_d = np.arange(100) * 4e-6
    popt, pcov = curve_fit(exponential, t_d[5:], (1e9*current[200:-12:40]))
    decay_time = -1 / popt[0]

    font = {
        'family': 'Arial',
        'size': 8
    }

    plt.figure(figsize=(2, 3))
    gs = gridspec.GridSpec(4, 4)
    ax1 = plt.subplot(gs[0:2, 0:4])
    ax2 = plt.subplot(gs[2:4, 0:4], sharex=ax1)
    ax1.get_xaxis().set_visible(False)

    color3 = np.array([103, 149, 216])/255
    color2 = np.array([110, 167, 151])/255
    color1 = np.array([117, 185, 86])/255

    ax1.plot(np.arange(1400) * dt_decay * 1e3, V_decay, color=color1)
    ax2.plot(np.arange(1005, 1400) * dt_decay * 1e3, 1e9 * current[50:-12:10], color=color3)

    ax2.tick_params(axis='y', direction='in', labelsize=8)
    ax2.tick_params(axis='x', direction='in', labelsize=8)
    ax1.tick_params(axis='y', direction='in', labelsize=8)
    ax2.set_xlabel('Time (ms)', fontdict=font)
    ax2.set_ylabel(r'Current (nA)', fontdict=font)
    ax1.set_ylabel(r'Voltage (V)', fontdict=font)
    ax2.set_xlim([0, 1.4])
    ax1.set_ylim([0.5, 4.4])
    ax2.set_ylim([0, 700])
    for label in ax2.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
        label.set_fontname('Arial')
    plt.show()

    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    plt.figure(figsize=(3, 3))
    plt.scatter(1+1e3*t_d[5:], 1e9*current[200:-12:40], s=4, label='Experimental data')
    plt.plot(np.arange(1006, 1401) * dt_decay * 1e3,
             exponential(np.arange(5, 400) * dt_decay, popt[0], popt[1], popt[2]), label='Fitting curve')
    plt.ylabel(r'Current (nA)')
    plt.xlabel('Time (ms)')
    plt.xlim([1, 1.4])
    plt.legend(frameon=False)
    plt.savefig(figdir+'/NbOx decay.svg', format='svg', dpi=300,
                bbox_inches='tight', transparent=True)
    plt.show()



if __name__ == '__main__':

    # Check whether the folder for storing figures is created
    fig_dir = './Figure/Characterization'
    if not os.path.exists(fig_dir):
        print('Creating new figure file directory...')
        os.makedirs(fig_dir)


    Draw_TiOx_Pulses_Stat(figdir=fig_dir)
    Draw_TiOx_Decay_Stat(figdir=fig_dir)
    Draw_TiOx_IV(figdir=fig_dir)
    Draw_TiOx_Pulse(figdir=fig_dir)
    Decay_Response_TiOx(figdir=fig_dir)
    Nine_TiOx_Pulse_Response(figdir=fig_dir)
    #
    #
    Draw_NbOx_Pulses_Stat(figdir=fig_dir)
    Draw_NbOx_Decay_Stat(figdir=fig_dir)
    Draw_NbOx_IV(figdir=fig_dir)
    Draw_NbOx_Pulse(figdir=fig_dir)
    Decay_Response_NbOx(figdir=fig_dir)
