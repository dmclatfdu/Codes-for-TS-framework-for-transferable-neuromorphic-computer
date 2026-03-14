
"""
This is the library file for basic functions and operations in the work Make Physical Reservoir Computing Transferable

Zefeng Zhang, Research Institute of Intelligent Complex Systems & Frontier Institute for Chip and System, Fudan Univ.

"""

import os
from scipy import io
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import csv
from matplotlib.patches import Patch
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import mpl_toolkits.axisartist as AA


def exponential(x, k, b1, b2):
    """
    The exponential function used for parameter estimation
    """
    return b1 + np.exp(x * k + b2)

def double_exp(x, k1, k2, b1, b2, b3):
    return b3 + b1 * np.exp(k1 * x) + b2 * np.exp(k2 * x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def swish(x):
    return x*sigmoid(x)

def RK_iteration(f, X_p, h, *args):
    """

    :param f: The function, the parameters need to be specified first
    :param X_p: The states and time point of the previous iteration, in the form of (x_p, t_p), ndarray form (N+1,)
    :param h: The iteration step, AKA the dt in the differential equation
    :param args: Other parameters defining the derivative,
                  keep the order of the parameters consistent with that in the velocity field function
    :return: X_n, the states and time point of the next iteration, in the form of (x_n, t_n), ndarray form (N+1,)
    """
    x_p, t_p = X_p[:-1], X_p[-1]
    k1 = f(x_p, t_p, *args)
    k2 = f(x_p + h / 2 * k1, t_p + h / 2, *args)
    k3 = f(x_p + h / 2 * k2, t_p + h / 2, *args)
    k4 = f(x_p + h * k3, t_p + h, *args)

    X_n = np.zeros(X_p.shape)
    x_n = x_p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    t_n = t_p + h
    X_n[:-1] = x_n
    X_n[-1] = t_n
    return X_n

def nrmse(series_1, series_2):
    """

    :param series_1: the data series used as the reference
    :param series_2: the data series as the approximation of series_1
    :return: the NRMSE value of the two series
    """
    nrmse_result = np.sqrt(nmse(series_1, series_2))
    return nrmse_result

def nmse(series_1, series_2):
    """

    :param series_1: the data series used as the reference
    :param series_2: the data series as the approximation of series_1
    :return: the NMSE value of the two series
    """
    nmse_result = mse(series_1, series_2) / np.var(series_1)
    return nmse_result

def mse(series_1, series_2):
    """

    :param series_1: the data series used as the reference
    :param series_2: the data series as the approximation of series_1
    :return: the NMSE value of the two series
    """
    mse_result = np.mean((series_2 - series_1) ** 2)
    return mse_result

def create_mask(
        out_dim,  # also the num_node
        **kwargs
):
    in_dim, abs_value = kwargs.get('in_dim', 1), kwargs.get('abs_value', 0.1)
    mask = np.random.randint(0, 2, (in_dim, out_dim))
    mask = abs_value * (2 * mask - 1)

    return mask

def mackey_glass_func(dt, x, x_tau, a, b, c):
    """

    For data generation of the physical experiment
    """
    x_next = x + dt * (-b * x + a * x_tau / (1 + x_tau ** c))

    return x_next

def create_MG_signal_file(
        file_num=20,
        each_length=80,
        dt=1,
        overlap=20,
        mask_abs=0.1,
        in_dim=1,
        out_dim=10,
        point_each_step=2,
        single_pulsewidth=20e-6,
        scaling=[2, 2.5],
        pred_shift=None,
        warm_up=None,
        tau=None,
        initial=None,
        a=None,
        b=None,
        c=None

):
    """

    For data generation of the physical experiment
    """
    point_number = each_length * file_num + overlap

    # Mask creation
    mask = np.random.randint(2, size=(in_dim, out_dim))
    mask = mask * 2 - 1
    mask = mask_abs * mask
    print(mask)

    # Default Settings
    if a is None:
        a = 0.2
    if b is None:
        b = 0.1
    if c is None:
        c = 10
    if tau is None:
        tau = 18

    # Points in simulation
    if warm_up is None:  # points for merely warmup, default to be identical length of the needed length
        warm_up = max(int(1 * point_number), 1000)
    if pred_shift is None:
        # pred_shift = 1  # Points for direct prediction shift
        pred_shift = 1
    time_total = (pred_shift + point_number + warm_up) * dt

    x_initial_len = int(tau / dt)
    running_len = (pred_shift + point_number + warm_up)
    x_total_len = x_initial_len + running_len

    # Initialization for MG DDE
    t = np.array([np.arange(0, time_total, dt)]).T
    x_record = np.zeros((x_total_len, 1)) + 0.01
    if initial is None:
        pass
    else:
        x_record[:int(tau / dt), 0] = initial

    for i in range(running_len):
        x_record[i + x_initial_len, 0] = mackey_glass_func(dt, x_record[i + x_initial_len - 1], x_record[i], a, b, c)

    x_base = x_record[x_initial_len + warm_up:-pred_shift]
    x_target = x_record[x_initial_len + warm_up + pred_shift:]
    x_awg = np.dot(x_base, mask)

    # Scaling
    x_awg = (x_awg - (x_awg.min() + x_awg.max()) / 2) / (x_awg.max() - x_awg.min()) * (scaling[1] - scaling[0]) + (
            scaling[0] + scaling[1]) / 2

    for i in range(file_num):
        x_in = x_base[i * each_length:(i + 1) * each_length + overlap]
        x_out = x_target[i * each_length:(i + 1) * each_length + overlap]
        x_awg_in = x_awg[i * each_length:(i + 1) * each_length + overlap]
        time = t[i * each_length:(i + 1) * each_length + overlap]

        with open(os.getcwd() + '/' + 'device num {}.csv'.format(i), mode='w', encoding='UTF-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(np.array([x_in.flatten(), x_out.flatten(), time.flatten()]).T)

        # Write signal
        t_sig_in = np.arange(0,
                             single_pulsewidth * (
                                     each_length + overlap) * out_dim + single_pulsewidth / 2 / point_each_step,
                             single_pulsewidth / point_each_step)
        with open(os.getcwd() + '/' + 'device awg time num {}.csv'.format(i), mode='w', encoding='UTF-8',
                  newline='') as f:
            writer = csv.writer(f)
            writer.writerows(np.array([t_sig_in]).T)
        with open(os.getcwd() + '/' + 'device awg input num {}.csv'.format(i), mode='w', encoding='UTF-8',
                  newline='') as f:
            writer = csv.writer(f)
            X_rec = x_awg_in
            x_awg_in = x_awg_in.flatten()
            x_sig_in = np.ones(((each_length + overlap) * out_dim) * point_each_step + 1) * scaling[0]

            for j in range((each_length + overlap) * out_dim):
                # if j == 0:
                #     previous = scaling[0]
                # else:
                #     previous = x_sig_in[j*point_each_step+1]
                # x_sig_in[j*point_each_step+2] = (previous + x_awg_in[j]) / 2
                # x_sig_in[j*point_each_step+3:(j+1)*point_each_step+2] = x_awg_in[j]
                #
                x_sig_in[j * point_each_step + 1:(j + 1) * point_each_step + 1] = x_awg_in[j]

            writer.writerows(np.array([x_sig_in]).T)

class MG_generator:

    """

    For data generation in the numerical simulation
    """

    def __init__(self, a, b, c, tau, shift=1):
        # Note: set shift as a positive integer
        self.a, self.b, self.c, self.tau, self.shift = a, b, c, tau, shift

    @staticmethod
    def mg_func(x, x_tau, a, b, c, dt):
        x_next = x + dt * (-b * x + a * x_tau / (1 + x_tau ** c))
        return x_next

    def iterate(self, dt, length, initial=None):

        # The initial conditions
        if initial is None:
            initial = 0.01 * np.ones(int(self.tau / dt))
        elif len(initial) < int(self.tau / dt):
            initial = np.r_[0.01 * np.ones(int(self.tau / dt) - len(initial)), initial]
        elif len(initial) > int(self.tau / dt):
            initial = initial[-len(self.tau / dt):]

        # The setting of the warmup steps
        warmup_steps = 2000

        # state record
        x_record = np.zeros(len(initial) + length + warmup_steps)
        x_record[:len(initial)] = initial

        # iterating process
        for i in range(length + warmup_steps):
            x_record[len(initial) + i] = MG_generator.mg_func(
                x_record[len(initial) + i - 1], x_record[i], self.a, self.b, self.c, dt
            )

        # Return the signal input and the target
        return x_record[-length - self.shift:-self.shift], np.array([x_record[-length:]]).T

class Lorenz_generator:

    def __init__(self, **kwargs):
        self.length = kwargs.get('length', 10000)
        self.h = kwargs.get('h', 0.02)
        self.warmup = kwargs.get('warmup', 1000)

    @staticmethod
    def lorenz63(x_p, t):
        x, y, z = x_p
        dxdt = np.array([10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 * z / 3])
        return dxdt

    def series(self):

        length = self.length
        warmup = self.warmup

        initials = np.array([-3, -2.7, 17])
        X = np.zeros((length+warmup+1, 4))
        X[0:3, 0] = initials

        for i in range(warmup+length):
            X[i+1, :] = RK_iteration(Lorenz_generator.lorenz63, X[i, :], self.h)

        Input, Target = X[warmup-1:length+warmup-1, :-1], X[warmup:length+warmup, :-1]

        return Input, Target

def p_to_stars(p, show_ns=False):
    if np.isnan(p):
        return "NA"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s." if show_ns else ""

def add_sig_bracket(ax, x1, x2, y, h, text, lw=0.9, fs=8):
    """画“秃宝盖”括号并写星号"""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], linewidth=lw, color="k", clip_on=False)
    if text:
        ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom", fontsize=fs)

def stats_summary_1d(x):
    """返回 robust 统计量：median, IQR, P90, P95, n"""
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return {"n": 0, "median": np.nan, "iqr": np.nan, "p90": np.nan, "p95": np.nan}

    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    p90 = np.percentile(x, 90)
    p95 = np.percentile(x, 95)
    return {"n": n, "median": med, "iqr": iqr, "p90": p90, "p95": p95}

def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



