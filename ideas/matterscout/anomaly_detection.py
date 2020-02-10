#this is the main machine learning script in our project
#it can extract features from the dataset and perform anomaly detection (although more work is needed to verify its actual performance)
#the first part performs feature extraction and the second one finds outliers in the freshly extracted dataset
#it is easy to modify the script to skip the first part and perform anomaly detection on datasets previously written to files.

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
import shutil


#clear working folder
try:
    folder = 'data'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
except:
    print('First time running')

#load rock temperature data (it was NOT used in this script but can easily be added)
rock_temperature_node = stuett.data.CsvSource(rock_temperature_file, store=derived_store)
rock_temperature = rock_temperature_node().to_dataframe()
rock_temperature = rock_temperature.reset_index('name').drop(["unit"], axis=1)
rock_temperature = rock_temperature.pivot(columns='name', values='CSV').drop(["position"], axis=1)
rock_temperature.index.rename("date")

#load precipitation data
prec_node = stuett.data.CsvSource(prec_file, store=derived_store)
prec = prec_node().to_dataframe()
prec = prec.reset_index('name').drop(["unit"], axis=1).pivot(columns='name', values='CSV').drop(["position"], axis=1)

#load seismic data specifying the time window
dates, seismic_data = load_seismic_source(start=date(2017, 11, 1), end=date(2017, 11, 30))
seismic_data = np.array(seismic_data)
seismic_df = pd.DataFrame(seismic_data)
print(seismic_df.describe())
seismic_df["date"] = dates
seismic_df = seismic_df.set_index("date")

#join the two datasets
dataset = seismic_df.join(prec)
dataset = dataset.fillna(0)
print(dataset.describe())

#save the joined dataset
dataset.to_csv("seismic_11.csv")

# COMMENT OUT if interested in making predictions
exit(0)

# COMMENT OUT if interested in loading preprocessed datasets stored locally
'''
data = []
for data_file in os.listdir('.'):
    print(data_file)
for data_file in os.listdir("raw_data"):
    print(os.path.join(data_file))
    data.append(pd.read_csv(os.path.join("raw_data", data_file)))

dataset = pd.concat(data)
dataset = dataset.set_index("date")
'''

#anomaly detection begins here
n_samples = 300
outliers_fraction = 0.05
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

#we experimented with different algorithms and chose an isolation forest
"""
anomaly_algorithms = [
    #("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    #("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
    #                                  gamma=0.1)),
    #("Isolation Forest", IsolationForest(behaviour='new',
    #                                     contamination=outliers_fraction,
    #                                     random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction))]
"""
anomaly_algorithms = [("Isolation Forest", IsolationForest(behaviour='new',
                                                           contamination=outliers_fraction,
                                                           random_state=42))]

for name, algorithm in anomaly_algorithms:
    #compute predictions
    y_pred = algorithm.fit_predict(dataset.values)

    #saverage an example of an "average" data point
    os.makedirs("data/normal/", exist_ok=True)
    normals = dataset[y_pred > 0]
    prec.loc[normals.index].median(axis=0).to_csv("data/normal/precipitation_data.csv")
    normal_seismic = []
    for normal_data in normals.index:
        normal_seismic.append(get_seismic_data(normal_data)[0])
    normal_seismic = np.median(np.array(normal_seismic), axis=0)
    normal_seismic = pd.DataFrame(np.transpose(normal_seismic), columns=["EHE", "EHN", "EHZ"])
    normal_seismic.to_csv("data/normal/seismic_data.csv", header=True)

    #compute anomaly scores and rescale them
    scores = algorithm.decision_function(dataset[y_pred < 0].values)
    scores_min = scores.min()
    scores_max = scores.max()
    #for each anomaly
    for date in dataset[y_pred < 0].index:
        #retrieve images
        os.makedirs("data/{}/images/".format(date), exist_ok=True)
        #compute an "anomaly score" from 1 to 5
        score = (algorithm.decision_function(
            dataset.loc[date].values.reshape((1, len(dataset.columns)))) - scores_min) * 5 / (scores_max - scores_min)
        with open("data/{}/score.txt".format(date), "w") as f:
            f.write(str(score[0]))

        #save measurements for outliers
        print("event at {}".format(date))
        prec.loc[date].to_csv("data/{}/precipitation_data.csv".format(date))

        sism = pd.DataFrame(np.transpose(get_seismic_data(date)[0]), columns=["EHE", "EHN", "EHZ"])
        sism["date"] = np.array([d for d in pd.date_range(date, date + timedelta(hours=1), freq='4ms')])
        sism.to_csv("data/{}/seismic_data.csv".format(date), header=True)

        start = str(date - timedelta(minutes=10))
        end = str(date + timedelta(minutes=60))

        images_df = anomaly_visualization.get_images_from_timestamps(image_store, start, end)()
        for key in images_df["filename"]:
            img = imio.imread(io.BytesIO(image_store[key]))
            #imshow(img)
            print("data/{}/images/{}.png".format(date, key.split("/")[1]))
            imio.imsave("data/{}/images/{}.png".format(date, key.split("/")[1]), img)
            #plt.show()