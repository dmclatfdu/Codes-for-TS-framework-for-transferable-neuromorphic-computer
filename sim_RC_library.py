"""
This is the library file for the memristor RC and TS training based on the TiOx memristor dynamics.

It includes the following modules:
(1) Building the dynamical model for the TiOx device, create the model with class TiOx_SRC();
(2) Examining the accuracy of the model through two simulated experiments: Pulse Response and Decay Response;
(3) Standard benchmark experiment, here we use the Mackey-Glass chaotic series one-step prediction;
(4) Real-world demo, here we use the MIT-BIH Arrhythmia Database for the arrhythmia detection.

Zefeng Zhang, Research Institute of Intelligent Complex Systems and ISTBI, Fudan Univ.

"""
from base_library import *
import h5py


def signal_process(
        signal,
        target,
        mask,
        fuse=False,
        **kwargs
):
    split = kwargs.get('split', 0.5)
    voltage_range = kwargs.get('voltage_range', [2, 2.5])

    signal = np.array([signal]).T
    signal_p = np.dot(signal, mask)
    signal_p_f = signal_p.flatten()

    # Re-scaling
    sig_up, sig_down = np.max(signal_p_f), np.min(signal_p_f)
    signal_p_f = (signal_p_f - (sig_up + sig_down) / 2) / (sig_up - sig_down) * (voltage_range[1] - voltage_range[0]) + \
                 (voltage_range[0] + voltage_range[1]) / 2

    if not fuse:
        split_point = int(split * len(signal_p_f))
        # Split the signal and target
        signal_train, signal_test = signal_p_f[:split_point], signal_p_f[split_point:]
        target_train, target_test = target[:int(split * len(target))], target[int(split * len(target)):]
        return signal_train, signal_test, target_train, target_test
    else:
        return signal_p_f, target


def create_maskh5(
        file_name, in_dim, out_dim, abs_value=0.1
):
    mask = np.random.randint(2, size=(in_dim, out_dim))
    mask = mask*2 - 1
    mask = abs_value*mask

    with h5py.File(file_name, 'w') as wfile:
        wfile.create_dataset('mask', data=mask)

    return None


def RC_settings(
        **kwargs
):
    ridge_alpha, train_mode, test_mode \
        = kwargs.get('ridge_alpha', 0), kwargs.get('train_mode', 'mix'), kwargs.get('test_mode', 'single')
    direct_noise, direct_noise_level = kwargs.get('direct_noise', False), kwargs.get('direct_noise_level', 0)

    return ridge_alpha, train_mode, test_mode, direct_noise, direct_noise_level


class TiOx_SRC:  # The SRC stands for the switching RC operation in the temporal-switch framework

    def __init__(self, **kwargs):
        self.k1 = kwargs.get('k1', 25e5)
        self.k2 = kwargs.get('k2', 3)
        self.k3 = kwargs.get('k3', 1.08e-5)
        self.k4 = kwargs.get('k4', 1.5)
        self.tau = kwargs.get('tau', 80e-6)
        self.tau0 = kwargs.get('tau0', 500e-3)
        self.g00 = kwargs.get('g00', 0.5)
        self.g0 = self.g00
        self.g = self.g00

    @staticmethod
    def TiOx_dynamic(x_p, t_p, V, tau, tau0, k1, k2, k3, k4, g00):

        """
        TiOx model: The dynamical model describing the behavior of the TiOx device
        """

        g, g0 = x_p[0], x_p[1]
        window_g = 1 - (1 - 2 * g) ** 2  # p1 = 2
        window_g0 = 1 - (-10 * (g0 - 1 / 2)) ** 2
        dgdt = -1 / tau * (g - g0) + k1 * k3 * g * np.exp(k4 * V) * window_g
        dg0dt = -1 / tau0 * (g0 - g00) + k2 * np.exp(V) * window_g0
        derivative = np.array([dgdt, dg0dt])
        return derivative

    def iterate_SRC(self, V, dt, **kwargs):
        # Parameters
        k1 = kwargs.get('k1', self.k1)
        k2 = kwargs.get('k2', self.k2)
        k3_0 = kwargs.get('k3', self.k3)
        k4 = kwargs.get('k4', self.k4)
        tau = kwargs.get('tau', self.tau)
        tau0 = kwargs.get('tau0', self.tau0)
        g00 = kwargs.get('g00', self.g00)
        tqdm_on = kwargs.get('tqdm_on', True)
        virtual_nodes = kwargs.get('virtual_nodes', 10)
        C2C_strength = kwargs.get('C2C_strength', 0.01e-5)
        clear = kwargs.get('clear', False)  # Clear the memory before the processing of given signal

        g_record = np.zeros(V.shape)
        g0_record = np.zeros(V.shape)
        i_record = np.zeros(V.shape)
        multiple_iteration = kwargs.get('multiple_iteration', 1)
        Dt = dt / multiple_iteration

        if clear:
            self.g0 = self.g00
            self.g = self.g00

        # Counter
        if tqdm_on:
            AAA = tqdm(range(len(V)))
        else:
            AAA = range(len(V))
        for i in AAA:

            # if i % virtual_nodes == 0:
            #     k3 = k3_0 + C2C_strength * np.random.randn()

            k3 = k3_0 + C2C_strength * np.random.randn()

            for j in range(multiple_iteration):
                self.g, self.g0, _ = RK_iteration(TiOx_SRC.TiOx_dynamic, np.array([self.g, self.g0, 0]),
                                                  Dt, V[i], tau, tau0, k1, k2, k3, k4, g00)

            i_current = k3 * self.g * np.exp(k4 * V[i]) * np.sign(V[i])
            i_record[i] = i_current
            g_record[i] = self.g
            g0_record[i] = self.g0

        return i_record, g_record, g0_record

    def iterate_one_step(self, V, dt, k3, **kwargs):
        # For continuous prediction of Lorenz series

        # The V as input here should has the length of num_node, V is the masked input of the previous prediction value
        g_record = np.zeros(V.shape)
        g0_record = np.zeros(V.shape)
        i_record = np.zeros(V.shape)
        multiple_iteration = kwargs.get('multiple_iteration', 1)
        Dt = dt / multiple_iteration

        k1 = kwargs.get('k1', self.k1)
        k2 = kwargs.get('k2', self.k2)
        tau = kwargs.get('tau', self.tau)
        tau0 = kwargs.get('tau0', self.tau0)
        g00 = kwargs.get('g00', self.g00)
        k4 = kwargs.get('k4', self.k4)

        for i in tqdm(range(len(V))):

            self.g, self.g0, _ = RK_iteration(TiOx_SRC.TiOx_dynamic, np.array([self.g, self.g0, 0]),
                                              Dt, V[i], tau, tau0, k1, k2, k3, k4, g00)

            i_current = k3 * self.g * np.exp(k4 * V[i]) * np.sign(V[i])
            i_record[i] = i_current
            g_record[i] = self.g
            g0_record[i] = self.g0

        return i_record, g_record, g0_record

