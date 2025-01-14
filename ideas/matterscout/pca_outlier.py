#this is an adaptation of the anomaly_detection script to also perform PCA
#it was proposet to represent events as a point cloud using this technique, but the idea was abandoned for time restrictions
#most lines used here are commented in the original script

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
from datetime import datetime
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.fftpack import fft
from sklearn.impute import SimpleImputer
import time
import os
from sklearn.decomposition import PCA

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

def get_images_from_timestamps(store, start, end):
    return stuett.data.MHDSLRFilenames(store=store,
                                       start_time=start,
                                       end_time=end,
                                       as_pandas=True)

def get_seismic_data(date):
    d = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    return np.array(stuett.data.SeismicSource(
        store=store,
        station="MH36",
        channel=["EHE", "EHN", "EHZ"],
        start_time=d,
        end_time=d + timedelta(hours=1),
    )())

#retrieve local data
data = []
for data_file in os.listdir('.'):
    print(data_file)
for data_file in os.listdir("raw_data"):
    print(os.path.join(data_file))
    data.append(pd.read_csv(os.path.join("raw_data", data_file)))

dataset = pd.concat(data)
dataset = dataset.set_index("date")
prec = dataset[["hail_accumulation","hail_duration","hail_intensity","hail_peak_intensity","rain_accumulation","rain_duration","rain_intensity","rain_peak_intensity"]]
print(dataset)

#perform PCA
pca = PCA(n_components=3)
transformed_dataset = pca.fit_transform(dataset.values)
result = pd.DataFrame(transformed_dataset, index=dataset.index)
print(result)
result.to_csv('pca.csv')

#COMMENT OUT to also try predicting
exit(0)

#it is possible to train the isolation forest on a dataset with recuded dimensions, althought we chose not to expore this possibility
algorithm = IsolationForest(behaviour='new',
                            contamination=0.01,
                            random_state=42, verbose=1)

y_pred = algorithm.fit_predict(dataset.values)
'''
os.makedirs("data/normal/", exist_ok=True)
normals = dataset[y_pred > 0].sample(100)
prec.loc[normals.index].median(axis=0).to_csv("data/normal/precipitation_data.csv")
normal_seismic = []
for normal_data in normals.index:
    normal_seismic.append(get_seismic_data(normal_data)[0])
normal_seismic = np.median(np.array(normal_seismic), axis=0)
normal_seismic = pd.DataFrame(np.transpose(normal_seismic), columns=["EHE", "EHN", "EHZ"])
normal_seismic.to_csv("data/normal/seismic_data.csv", header=True)
'''
scores = algorithm.decision_function(dataset[y_pred < 0].values)
scores_min = scores.min()
scores_max = scores.max()
for date in dataset[y_pred < 0].index:
    d = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    os.makedirs("data/{}/images/".format(date), exist_ok=True)
    score = (algorithm.decision_function(
        dataset.loc[date].values.reshape((1, len(dataset.columns)))) - scores_min) * 5 / (scores_max - scores_min)
    with open("data/{}/score.txt".format(date), "w") as f:
        f.write(str(score[0]))

    print("event at {}".format(date))
    # print(dataset.loc[date])
    prec.loc[date].to_csv("data/{}/precipitation_data.csv".format(date))

    sism = pd.DataFrame(np.transpose(get_seismic_data(date)[0]), columns=["EHE", "EHN", "EHZ"])
    sism["date"] = np.array([d for d in pd.date_range(d, d + timedelta(hours=1), freq='4ms')])
    sism.to_csv("data/{}/seismic_data.csv".format(date), header=True)

    # print(dataset.describe())
    start = str(d - timedelta(minutes=10))
    end = str(d + timedelta(minutes=60))

    images_df = anomaly_visualization.get_images_from_timestamps(image_store, start, end)()
    for key in images_df["filename"]:
        img = imio.imread(io.BytesIO(image_store[key]))
        #imshow(img)
        print("data/{}/images/{}.png".format(date, key.split("/")[1]))
        imio.imsave("data/{}/images/{}.png".format(date, key.split("/")[1]), img)
        #plt.show()