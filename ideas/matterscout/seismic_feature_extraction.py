#these functionalities have been moved and integrated with anomaly_detection.py
#this was the first script in which we attempted to extract features from the sataset, focusing on seismic measurements alone

#extracted features are written to a csv file and later used

import scipy
import stuett
from stuett.global_config import get_setting, setting_exists, set_setting
from sklearn import svm
import numpy as np
import pandas as pd
from skimage import io as imio
import io
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import anomaly_visualization
from dateutil import rrule
from datetime import date, timedelta
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.fftpack import fft
import time

account_name = (
    get_setting("azure")["account_name"]
    if setting_exists("azure")
    else "storageaccountperma8980"
)
account_key = (
    get_setting("azure")["account_key"] if setting_exists("azure") else None
)
store = stuett.ABSStore(
    container="hackathon-on-permafrost",
    prefix="seismic_data/4D/",
    account_name=account_name,
    account_key=account_key,
)
rock_temperature_file = "MH30_temperature_rock_2017.csv"
derived_store = stuett.ABSStore(
    container="hackathon-on-permafrost",
    prefix="timeseries_derived_data_products",
    account_name=account_name,
    account_key=account_key,
)
image_store = stuett.ABSStore(
    container="hackathon-on-permafrost",
    prefix="timelapse_images_fast",
    account_name=account_name,
    account_key=account_key,
)

#calculates entropy on the measurements
def calculate_entropy(v):
    counter_values = Counter(v).most_common()
    probabilities = [elem[1] / len(v) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy

#extracts statistical features (we decided not to include entropy)
def min_max_estractor(row):
    return  [np.min(row), np.max(row), np.var(row), np.mean((row-np.mean(row))**2), np.mean(row**2),#calculate_entropy(row),
            np.percentile(row, 1), np.percentile(row, 5), np.percentile(row, 25),
            np.percentile(row, 95), np.percentile(row,95), np.percentile(row, 99)]

#computes fourier transform of the signal and extracts features
def fourier_extractor(x):
    sampling_freq = 250
    N=len(x)
    f_values = np.linspace(0.0, sampling_freq/2, N//2)
    fft_values_ = fft(x)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])

    #the following values were determined empirically
    coeff_0=fft_values[0] #coefficient at 0Hz
    peak_70=0 #coefficient around 70 Hz
    coeff = np.zeros(20) #max coefficient from each 2 Hz interval (0-40)
    integral40 = 0 #integral from 0 to 40 Hz
    integral125 = np.avg(fft_values) #integral over the whole transform
    for i in range(0, len(f_values)):
        if f_values[i]>69 and f_values[i]<72 and fft_values[i]>peak_70:
            peak_70=fft_values[i]
        if f_values[i]<40:
            integral40+=fft_values[i]
            if fft_values[i] > coeff[int(i/2)]:
                coeff[int(i/2)]=fft_values[i]
    return coeff + [coeff_0, peak_70, integral40, integral125]

#extracts features from an hour worth of seismic data from three sensors
def transform_hour(data):
    data = np.array(data)
    features=[]
    print('Extracting features')
    for extractor in [min_max_estractor]:#, fourier_extractor]:     #we choose to exclude Fourier features for computational efficiency
        for element in extractor(data):
            features.append(element)
    return features

# Load the data source and apply transformations
def load_seismic_source(start, end):
    output = []
    dates = []
    for i, date in enumerate(rrule.rrule(rrule.HOURLY, dtstart=start, until=end)):
        try:
            seismic_node = stuett.data.SeismicSource(
                store=store,
                station="MH36",
                channel=["EHE", "EHN", "EHZ"],
                start_time=date,
                end_time=date + timedelta(hours=1),
            )
            dates.append(date)
            output.append(transform_hour(seismic_node()))
            print('Extracted ' + str(i))
        except:
            print('Error')
    return dates, output

dates, seismic_data = np.array(load_seismic_source(start=date(2017, 1, 1), end=date(2018, 1, 1)))   #test time window
seismic_df = pd.DataFrame(seismic_data)
seismic_df["date"] = dates
seismic_df.set_index("date")
seismic_df.to_csv('seismic_features.csv')