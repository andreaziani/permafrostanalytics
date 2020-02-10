from sklearn import svm
import numpy as np
import pandas as pd
from skimage import io as imio
import io
from datetime import timedelta
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import os

from data_load import get_seismic_data, get_images_from_timestamps, image_store, DATA_FOLDER

n_samples = 300
outliers_fraction = 0.05
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

#we experimented with different algorithms and chose an isolation forest
anomaly_algorithms = {"Robust_covariance": EllipticEnvelope(contamination=outliers_fraction),
                      "One-Class_SVM": svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1),
                      "Isolation_Forest": IsolationForest(behaviour='new',
                                          contamination=outliers_fraction,
                                          random_state=42),
                      "Local_Outlier_Factor": LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)}


def predict_outlier_labels(anomaly_algorithm, dataset):
    return anomaly_algorithm.fit_predict(dataset.values)


def score_dataset(anomaly_algorithm, dataset, y_pred):
    # compute anomaly scores and rescale them
    scores = anomaly_algorithm.decision_function(dataset[y_pred < 0].values)
    scores_min = scores.min()
    scores_max = scores.max()
    scores = {}
    # for each anomaly
    for date in dataset[y_pred < 0].index:
        # compute an "anomaly score" from 1 to 5
        scores[date] = (anomaly_algorithm.decision_function(
            dataset.loc[date].values.reshape((1, len(dataset.columns)))) - scores_min) * 5 / (scores_max - scores_min)[0]
    return scores


def save_normal_data(dataset, y_pred, prec):
    # saverage an example of an "average" data point
    os.makedirs(DATA_FOLDER + "/normal/", exist_ok=True)
    normals = dataset[y_pred > 0]
    prec.loc[normals.index].median(axis=0).to_csv("data/normal/precipitation_data.csv")
    normal_seismic = []
    for normal_data in normals.index:
        normal_seismic.append(get_seismic_data(normal_data)[0])
    normal_seismic = np.median(np.array(normal_seismic), axis=0)
    normal_seismic = pd.DataFrame(np.transpose(normal_seismic), columns=["EHE", "EHN", "EHZ"])
    normal_seismic.to_csv(DATA_FOLDER + "/normal/seismic_data.csv", header=True)


def save_anomalies(dataset, anomaly_algorithm, y_pred, prec):
    scores = score_dataset(anomaly_algorithm, dataset, y_pred)
    for date in dataset[y_pred < 0].index:
        # retrieve images
        os.makedirs(DATA_FOLDER + "/{}/images/".format(date), exist_ok=True)
        with open(DATA_FOLDER + "/{}/score.txt".format(date), "w") as f:
            f.write(str(scores[date]))

        # save measurements for outliers
        print("event at {}".format(date))
        prec.loc[date].to_csv(DATA_FOLDER + "/{}/precipitation_data.csv".format(date))

        sism = pd.DataFrame(np.transpose(get_seismic_data(date)[0]), columns=["EHE", "EHN", "EHZ"])
        sism["date"] = np.array([d for d in pd.date_range(date, date + timedelta(hours=1), freq='4ms')])
        sism.to_csv(DATA_FOLDER + "/{}/seismic_data.csv".format(date), header=True)

        start = str(date - timedelta(minutes=10))
        end = str(date + timedelta(minutes=60))

        images_df = get_images_from_timestamps(start, end)()
        for key in images_df["filename"]:
            img = imio.imread(io.BytesIO(image_store[key]))
            # imshow(img)
            print(DATA_FOLDER + "/{}/images/{}.png".format(date, key.split("/")[1]))
            imio.imsave(DATA_FOLDER + "/{}/images/{}.png".format(date, key.split("/")[1]), img)
            # plt.show()