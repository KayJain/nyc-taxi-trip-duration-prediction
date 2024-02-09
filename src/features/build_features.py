import sys
import yaml
import click
import pathlib

import numpy as np
import pandas as pd

class BuildFeatures():
    def __init__(self, input_train_data_path, input_test_data_path, output_data_path):
        self.input_train_data_path = input_train_data_path
        self.input_test_data_path = input_test_data_path
        self.output_data_path = output_data_path

    def read_data(self):
        '''This methods reads the data from input data path (csv file) to pandas dataframe.'''
        self.train = pd.read_csv(self.input_train_data_path)
        self.test = pd.read_csv(self.input_test_data_path)

    def datetime_conversion(self):
        self.train['pickup_datetime'] = pd.to_datetime(self.train['pickup_datetime'])
        self.train['dropoff_datetime'] = pd.to_datetime(self.train['dropoff_datetime'])

        self.test['pickup_datetime'] = pd.to_datetime(self.test['pickup_datetime'])
        self.test['dropoff_datetime'] = pd.to_datetime(self.test['dropoff_datetime'])

    def crete_datetime_features(self):
        '''This methods creates the day, hour etc features from datetime feature column.'''
        self.train['pickup_weekday'] = self.train['pickup_datetime'].dt.dayofweek
        self.train['pickup_hour'] = self.train['pickup_datetime'].dt.hour
        weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        self.train['pickup_weekday'] = self.train['pickup_weekday'].map(lambda x: weekday[x])

        self.test['pickup_weekday'] = self.test['pickup_datetime'].dt.dayofweek
        self.test['pickup_hour'] = self.test['pickup_datetime'].dt.hour
        weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        self.test['pickup_weekday'] = self.test['pickup_weekday'].map(lambda x: weekday[x])

    def haversine_array(self, lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        AVG_EARTH_RADIUS = 6371  # in km
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h

    def dummy_manhattan_distance(self, lat1, lng1, lat2, lng2):
        a = self.haversine_array(lat1, lng1, lat1, lng2)
        b = self.haversine_array(lat1, lng1, lat2, lng1)
        return a + b

    def bearing_array(self, lat1, lng1, lat2, lng2):
        AVG_EARTH_RADIUS = 6371  # in km
        lng_delta_rad = np.radians(lng2 - lng1)
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
        return np.degrees(np.arctan2(y, x))
    
    def create_distuance_feature(self):
        self.train['distance_haversine'] = self.haversine_array(self.train['pickup_latitude'].values, self.train['pickup_longitude'].values, self.train['dropoff_latitude'].values, self.train['dropoff_longitude'].values)
        self.train['distance_dummy_manhattan'] = self.dummy_manhattan_distance(self.train['pickup_latitude'].values, self.train['pickup_longitude'].values, self.train['dropoff_latitude'].values, self.train['dropoff_longitude'].values)
        self.train['direction'] = self.bearing_array(self.train['pickup_latitude'].values, self.train['pickup_longitude'].values, self.train['dropoff_latitude'].values, self.train['dropoff_longitude'].values)

        self.test['distance_haversine'] = self.haversine_array(self.test['pickup_latitude'].values, self.test['pickup_longitude'].values, self.test['dropoff_latitude'].values, self.test['dropoff_longitude'].values)
        self.test['distance_dummy_manhattan'] = self.dummy_manhattan_distance(self.test['pickup_latitude'].values, self.test['pickup_longitude'].values, self.test['dropoff_latitude'].values, self.test['dropoff_longitude'].values)
        self.test['direction'] = self.bearing_array(self.test['pickup_latitude'].values, self.test['pickup_longitude'].values, self.test['dropoff_latitude'].values, self.test['dropoff_longitude'].values)

    def remove_unwanted_features(self):
        self.train = self.train.drop(columns=['id', 'pickup_datetime', 'dropoff_datetime'])
        self.test = self.test.drop(columns=['id', 'pickup_datetime', 'dropoff_datetime'])

    def save_data(self):
        self.train.to_csv(self.output_data_path + '/train.csv', index=False)
        self.test.to_csv(self.output_data_path + '/test.csv', index=False)

    def create_features(self):
        self.read_data()
        self.datetime_conversion()
        self.crete_datetime_features()
        self.create_distuance_feature()
        self.remove_unwanted_features()
        self.save_data()

@click.command()
@click.argument('input_train_path', type=click.Path())
@click.argument('input_test_path', type=click.Path())
@click.argument('output_path', type=click.Path())
def main(input_train_path, input_test_path, output_path):
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_train_data_path = home_dir.as_posix() + input_train_path
    input_test_data_path = home_dir.as_posix() + input_test_path
    output_data_path = home_dir.as_posix() + output_path

    features = BuildFeatures(input_train_data_path, input_test_data_path, output_data_path)
    features.create_features()

if __name__ == '__main__':
    main()