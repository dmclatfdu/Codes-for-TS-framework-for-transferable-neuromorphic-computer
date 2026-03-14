# This is the signal generator for voltage signal used in the experimental RC spoken digit recognition task

import csv
import librosa
import numpy as np
import pandas as pd
import os
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import mpl_toolkits.axisartist as AA


def create_Voice_signal_file(
        scaling=[1.8, 2.3],
        in_dim=32,
        out_dim=40,  # please set as smaller than 40
        pw=20e-6,  # pulsewidth (in second)
        point_padding=2,
        mean_row=None,
        max_row=None,
        min_row=None,
        data_dir=None
):
    if not os.path.exists(r'{}/mask_I{}O{}.csv'.format(data_dir, in_dim, out_dim)):
        create_mask_file(in_dim, out_dim, data_dir=data_dir)
        print('Creating mask ...')
    else:
        print('Mask already exists.')

    mask = pd.read_csv(r'{}/mask_I{}O{}.csv'.format(data_dir, in_dim, out_dim), header=None).values

    steps_rec = []

    for i in range(5):
        for j in range(10):
            for k in range(10):
                _select = '0{}f{}set{}'.format(j, i+1, k)
                filename = r'.\Data\Voice\Voice Source Data\train\f{}\{}.wav'.format(i+1, _select)

                _audio, sr = librosa.load(filename, sr=None)
                _audio_rs = librosa.resample(_audio, orig_sr=12500, target_sr=8000)
                _feature = librosa.feature.mfcc(y=_audio_rs, sr=8000, dct_type=2, n_mfcc=in_dim).T

                if (i+j+k) != 0 or mean_row is not None or min_row is not None or max_row is not None:
                    pass
                else:
                    max_row = np.max(_feature, axis=0)
                    min_row = np.min(_feature, axis=0)
                    mean_row = np.mean(_feature, axis=0)

                _feature = (_feature - mean_row) / (max_row - min_row)
                _feature = _feature[:25, :]
                steps = len(_feature[:, 0])

                steps_rec.append(steps)

                # masking and rescaling
                masked_input = _feature @ mask
                if (i + j + k) == 0:
                    _min, _max = np.min(masked_input), np.max(masked_input)
                else:
                    pass

                rescaled_input = ((masked_input - (_min + _max) / 2)
                                  / (_max - _min) * (scaling[1] - scaling[0]) +
                                  (scaling[1] + scaling[0]) / 2)
                rescaled_input = rescaled_input.flatten()

                full_length_input = np.zeros(2000+1)
                full_length_input[0] = scaling[0]
                for position in range(steps*out_dim):
                    full_length_input[position*point_padding+1: (position+1)*point_padding+1] = rescaled_input[position]
                times = np.arange(0, pw*25*out_dim + pw/(2*point_padding), pw/point_padding)

                full_signal = np.c_[times, full_length_input]

                with open(r'{}/awg_{}.csv'.format(data_dir, _select), mode='w',
                          encoding='UTF-8', newline='') as f_tr:
                    writer = csv.writer(f_tr)
                    writer.writerows(full_signal)

    # record the actual steps
    steps_rec = np.array([steps_rec]).T
    with open(r'{}/steps_rec.csv'.format(data_dir), mode='w',encoding='UTF-8', newline='') as f_tr:
        writer = csv.writer(f_tr)
        writer.writerows(steps_rec)


def create_mask_file(in_dim, out_dim, mask_abs=0.1, data_dir=None):
    # Mask creation
    mask = np.random.randint(2, size=(in_dim, out_dim))
    mask = mask * 2 - 1
    mask = mask_abs * mask

    with open(r'{}/mask_I{}O{}.csv'.format(data_dir, in_dim, out_dim), mode='w',
              encoding='UTF-8', newline='') as f_tr:
        writer = csv.writer(f_tr)
        writer.writerows(mask)

    return None


if __name__ == '__main__':

    # Check whether the folder for storing figures is created
    devices = ['NbOx', 'TiOx']
    for Device in devices:
        fig_dir = './Figure/Voice/Exp/{}'.format(Device)
        data_dir1 = './Data/Voice/Exp/{}/inputs'.format(Device)
        data_dir2 = './Data/Voice/Exp/{}/outputs'.format(Device)
        if not os.path.exists(fig_dir):
            print('Creating new figure file directory...')
            os.makedirs(fig_dir)
        if not os.path.exists(data_dir1):
            print('Creating new data file directory...')
            os.makedirs(data_dir1)
        if not os.path.exists(data_dir2):
            print('Creating new data file directory...')
            os.makedirs(data_dir2)

        # create_Voice_signal_file(data_dir=data_dir1)  # please disable this function when the voice signal file is generated
        # The step_rec.csv file is provided together with the already generated signal file

