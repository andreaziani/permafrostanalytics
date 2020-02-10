from collections import Counter

import scipy
import numpy as np
from scipy.fftpack import fft


# calculates entropy on the measurements
def calculate_entropy(v):
    counter_values = Counter(v).most_common()
    probabilities = [elem[1] / len(v) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy


# extracts statistical features (we decided to exclude most of the features for computational reasons)
def min_max_estractor(row):
    return [np.max(row), np.mean(row), np.min(row), np.var(row), np.mean(row ** 2)]  # calculate_entropy(row),
    # np.percentile(row, 1), np.percentile(row, 5), np.percentile(row, 25),
    # np.percentile(row, 95), np.percentile(row, 95), np.percentile(row, 99)]


# computes fourier transform of the signal and extracts features
def fourier_extractor(x):
    sampling_freq = 250
    N = len(x)
    f_values = np.linspace(0.0, sampling_freq / 2, N // 2)
    fft_values_ = fft(x)
    fft_values = 2.0 / N * np.abs(fft_values_[0:N // 2])

    #values determined empirically
    coeff_0 = fft_values[0]  # coefficient at 0Hz
    peak_70 = 0  # coefficient around 70 Hz
    coeff = np.zeros(20)  # max coefficient from each 2 Hz interval (0-40)
    integral40 = 0  # integral from 0 to 40 Hz
    integral125 = np.mean(fft_values)  # integral over the whole transform
    for i in range(0, len(f_values)):
        if f_values[i] > 69 and f_values[i] < 72 and fft_values[i] > peak_70:
            peak_70 = fft_values[i]
        if f_values[i] < 40:
            integral40 += fft_values[i]
            if fft_values[i] > coeff[int(f_values[i] / 2)]:
                coeff[int(f_values[i] / 2)] = fft_values[i]
    return list(coeff) + [coeff_0, peak_70, integral40, integral125]


# extracts features from an hour worth of seismic data from three sensors
def transform_hour(data):
    data = np.array(data)[0][-3:]
    features = []
    #for first_dimension in data:
    for row in data:
        for extractor in [min_max_estractor]:  # , fourier_extractor]:  #fourier features were not included
            for element in extractor(row):
                features.append(element)
    features = np.array(features)
    return features