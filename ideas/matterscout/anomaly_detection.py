import stuett
import torch
import numpy as np
import scipy
import argparse
import datetime as dt
import os
import pandas as pd
import xarray as xr

from datasets import SeismicDataset, DatasetFreezer, DatasetMerger
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

import stuett
from stuett.global_config import get_setting, setting_exists, set_setting

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from ignite.metrics import Accuracy
from sklearn import svm

from pathlib import Path

from PIL import Image

import numpy as np
import json
import pandas as pd
import os
from skimage import io as imio
import io, codecs

from models import SimpleCNN
import anomaly_visualization
from dateutil import rrule
from datetime import date, timedelta
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

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

def get_seismic_transform():
    def to_db(x,min_value=1e-10,reference=1.0):
        value_db = 10.0 * xr.ufuncs.log10(xr.ufuncs.maximum(min_value, x))
        value_db -= 10.0 * xr.ufuncs.log10(xr.ufuncs.maximum(min_value, reference))
        return value_db

    spectrogram = stuett.data.Spectrogram(nfft=512, stride=512, dim="time", sampling_rate=1000)

    transform = transforms.Compose([
        lambda x: x / x.max(),                          # rescale to -1 to 1
        spectrogram,                                    # spectrogram
        lambda x: to_db(x).values.squeeze(),
        lambda x: Tensor(x)
        ])

    return transform


def transform_hour(data):
    pass

def transform_minute(data):
    pass


# Load the data source
def load_seismic_source(start, end):
    output = []
    dates = []
    for date in rrule.rrule(rrule.HOURLY, dtstart=start, until=end):
        seismic_node = stuett.data.SeismicSource(
            store=store,
            station="MH36",
            channel=["EHE", "EHN", "EHZ"],
            start_time=date,
            end_time=date + timedelta(hours=1),
        )
        dates.append(date)
        output.append(transform_hour(seismic_node()))
    return dates, output

def load_image_source():
    image_node = stuett.data.MHDSLRFilenames(
        store=store,
        force_write_to_remote=True,
        as_pandas=False,
    )
    return image_node, 3

transform = get_seismic_transform()
dates, seismic_data = np.array(load_seismic_source(start=date(2017, 1, 1), end=date(2018, 1, 1)))
seismic_df = pd.DataFrame(seismic_data)
seismic_df["date"] = dates
seismic_df.set_index("date")

rock_temperature_node = stuett.data.CsvSource(rock_temperature_file,store=store)
rock_temperature = rock_temperature_node()


n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers


anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Isolation Forest", IsolationForest(behaviour='new',
                                         contamination=outliers_fraction,
                                         random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction))]

dataset = rock_temperature
for name, algorithm in anomaly_algorithms:
    y_pred = algorithm.fit_predict(dataset.values)
    for date in dataset[y_pred].index:
        start = date - timedelta(hours=1)
        end = date + timedelta(hours=1)
        images_df = anomaly_visualization.get_images_from_timestamps(image_store, start, end)
        for key in images_df["filename"]:
            img = imio.imread(io.BytesIO(store[key]))
            print(type(img))
