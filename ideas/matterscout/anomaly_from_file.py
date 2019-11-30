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
from sklearn.impute import SimpleImputer
import time
import os


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
prec_file = "MH25_vaisalawxt520prec_2017.csv"
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


def get_seismic_data(date):
    return np.array(stuett.data.SeismicSource(
        store=store,
        station="MH36",
        channel=["EHE", "EHN", "EHZ"],
        start_time=date,
        end_time=date + timedelta(hours=1),
    )())

prec_node = stuett.data.CsvSource(prec_file, store=derived_store)
prec = prec_node().to_dataframe()
prec = prec.reset_index('name').drop(["unit"], axis=1).pivot(columns='name', values='CSV').drop(["position"], axis=1)

data = []
for data_file in os.listdir("raw_data"):
    data.append(pd.read_csv(data_file))

dataset = pd.concat(data)

algorithm = IsolationForest(behaviour='new',
                            contamination=0.05,
                            random_state=42)

y_pred = algorithm.fit_predict(dataset.values)

os.makedirs("data/normal/", exist_ok=True)
normals = dataset[y_pred > 0]
prec.loc[normals.index].median(axis=0).to_csv("data/normal/precipitation_data.csv")
normal_seismic = []
for normal_data in normals.index:
    normal_seismic.append(get_seismic_data(normal_data)[0])
normal_seismic = np.median(np.array(normal_seismic), axis=0)
normal_seismic = pd.DataFrame(np.transpose(normal_seismic), columns=["EHE", "EHN", "EHZ"])
normal_seismic.to_csv("data/normal/seismic_data.csv", header=True)

scores = algorithm.decision_function(dataset[y_pred < 0].values)
scores_min = scores.min()
scores_max = scores.max()
for date in dataset[y_pred < 0].index:

    os.makedirs("data/{}/images/".format(date), exist_ok=True)
    score = (algorithm.decision_function(
        dataset.loc[date].values.reshape((1, len(dataset.columns)))) - scores_min) * 5 / (scores_max - scores_min)
    with open("data/{}/score.txt".format(date), "w") as f:
        f.write(str(score[0]))

    print("event at {}".format(date))
    # print(dataset.loc[date])
    prec.loc[date].to_csv("data/{}/precipitation_data.csv".format(date))

    sism = pd.DataFrame(np.transpose(get_seismic_data(date)[0]), columns=["EHE", "EHN", "EHZ"])
    sism["date"] = np.array([d for d in pd.date_range(date, date + timedelta(hours=1), freq='4ms')])
    sism.to_csv("data/{}/seismic_data.csv".format(date), header=True)

    # print(dataset.describe())
    start = str(date - timedelta(minutes=10))
    end = str(date + timedelta(minutes=60))

    images_df = anomaly_visualization.get_images_from_timestamps(image_store, start, end)()
    for key in images_df["filename"]:
        img = imio.imread(io.BytesIO(image_store[key]))
        imshow(img)
        print("data/{}/images/{}.png".format(date, key.split("/")[1]))
        imio.imsave("data/{}/images/{}.png".format(date, key.split("/")[1]), img)
        plt.show()