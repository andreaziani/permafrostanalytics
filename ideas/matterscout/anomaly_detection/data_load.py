import stuett
from stuett.global_config import get_setting, setting_exists, set_setting
import numpy as np
import pandas as pd
from datetime import timedelta
import os

DATA_FOLDER = "../data"

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


#returns images captured at a particular time
def get_images_from_timestamps(start, end):
    return stuett.data.MHDSLRFilenames(store=image_store,
                                       start_time=start,
                                       end_time=end,
                                       as_pandas=True)


#reads seismic data
def get_seismic_data(date):
    return np.array(stuett.data.SeismicSource(
        store=store,
        station="MH36",
        channel=["EHE", "EHN", "EHZ"],
        start_time=date,
        end_time=date + timedelta(hours=1),
    )())


# loads the data source and applies the transformations
def load_seismic_source(start, end, transform):
    output = []
    dates = []
    for date in pd.date_range(start, end, freq='1H'):
        print(date)
        try:
            seismic_node = stuett.data.SeismicSource(
                store=store,
                station="MH36",
                channel=["EHE", "EHN", "EHZ"],
                start_time=date,
                end_time=date + timedelta(hours=1),
            )
            newline = transform(seismic_node())
            output.append(newline)
            dates.append(date)
        except:
            pass
    return dates, output


#loads image source
def load_image_source():
    image_node = stuett.data.MHDSLRFilenames(
        store=store,
        force_write_to_remote=True,
        as_pandas=False,
    )
    return image_node, 3


def load_temperature():
    rock_temperature_node = stuett.data.CsvSource(rock_temperature_file, store=derived_store)
    rock_temperature = rock_temperature_node().to_dataframe()
    rock_temperature = rock_temperature.reset_index('name').drop(["unit"], axis=1)
    rock_temperature = rock_temperature.pivot(columns='name', values='CSV').drop(["position"], axis=1)
    rock_temperature.index.rename("date")
    return rock_temperature


#load precipitation data
def load_precipitation():
    prec_node = stuett.data.CsvSource(prec_file, store=derived_store)
    prec = prec_node().to_dataframe()
    prec = prec.reset_index('name').drop(["unit"], axis=1).pivot(columns='name', values='CSV').drop(["position"], axis=1)
    return prec


#load seismic data specifying the time window
def load_seismic(start_date, end_date, transform):
    dates, seismic_data = load_seismic_source(start_date, end_date, transform)
    seismic_data = np.array(seismic_data)
    seismic_df = pd.DataFrame(seismic_data)
    print(seismic_df.describe())
    seismic_df["date"] = dates
    seismic_df = seismic_df.set_index("date")
    return seismic_df


# There is possible support for extending with different types of data
# but currently the data creation only works with precipitation and seismic data
def create_dataset(start_date, end_date, precipitaion_data=True, rock_temperature_data=False, verbose=False):
    dataset = load_seismic(start_date, end_date)
    if precipitaion_data:
        dataset = dataset.join(load_precipitation())
    if rock_temperature_data:
        dataset = dataset.join(load_temperature())
    dataset = dataset.fillna(0)
    if verbose:
        print(dataset.describe())
    return dataset


def load_dataset_from_filesystem(verbose=False):
    data = []
    if verbose:
        for data_file in os.listdir('.'):
            print(data_file)
    for data_file in os.listdir("raw_data"):
        if verbose:
            print(os.path.join(data_file))
        data.append(pd.read_csv(os.path.join("raw_data", data_file)))

    dataset = pd.concat(data)
    dataset = dataset.set_index("date")
    if verbose:
        dataset.summary()
    return dataset
