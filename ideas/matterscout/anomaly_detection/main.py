from datetime import date
import os
import shutil
import argparse
from data_load import DATA_FOLDER, create_dataset, load_dataset_from_filesystem, load_precipitation
from detection import predict_outlier_labels, save_normal_data, save_anomalies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", type=bool, default=False, help="clear working folder")
    parser.add_argument("--create", type=bool, default=False,
                        help="create dataset if true, load it from file system otherwise")
    parser.add_argument("--anomaly_algorithm", type=str, default="Isolation_Forest",
                        choices=["Robust_covariance", "One-Class_SVM", "Isolation_Forest", "Local_Outlier_Factor"],
                        help="anomaly detection algorithm to be used")
    parser.add_argument("--save_data", type=bool, default=False,
                        help="save data to file system")
    parser.add_argument("--save_dataset", type=bool, default=False,
                        help="save dataset to file system")
    parser.add_argument("--save_dataset_file", type=str, default="seismic.csv",
                        help="file where to save dataset if save_dataset is true")

    args = parser.parse_args()

    if args.clear:
        # clear working folder
        try:
            folder = DATA_FOLDER
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
    if args.create:
        dataset = create_dataset(start_date=date(2017, 11, 1), end_date=date(2017, 11, 30))
        if args.save_dataset:
            dataset.to_csv(args.save_dataset_file)
    else:
        dataset = load_dataset_from_filesystem()
    prec = load_precipitation()
    if args.save_data:
        y_pred = predict_outlier_labels(args.anomaly_algorithm, dataset)
        save_normal_data(dataset, y_pred, prec)
        save_anomalies(dataset, args.anomaly_algorithm, y_pred, prec)


if __name__ == '__main__':
    main()