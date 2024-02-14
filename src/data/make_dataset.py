import sys
import pathlib
import click
import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion():
    def __init__(self, input_data_path, output_data_path, seed, test_size):
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.seed = seed
        self.test_size = test_size

    def read_data(self):
        self.df = pd.read_csv(self.input_data_path)

    def outlier_removal(self):
        #city_long_border = (-74.03, -73.75)
        #city_lat_border = (40.63, 40.85)
        self.df = self.df[(self.df['pickup_longitude'] < -73.75) & (self.df['pickup_longitude'] > -74.03)]
        self.df = self.df[(self.df['pickup_latitude'] < 40.85) & (self.df['pickup_latitude'] > 40.63)] 
        self.df = self.df[(self.df['dropoff_longitude'] < -73.75) & (self.df['dropoff_longitude'] > -74.03)] 
        self.df = self.df[(self.df['dropoff_latitude'] < 40.85) & (self.df['dropoff_latitude'] > 40.63)]  

    def split_data(self):
        self.train, self.test = train_test_split(self.df, test_size=self.test_size, random_state=self.seed)

    def save_data(self):
        print(self.train.head())
        print(self.train.shape)
        self.train.to_csv(self.output_data_path + '/train.csv', index=False)
        self.test.to_csv(self.output_data_path + '/test.csv', index=False)

    def preprocess(self):
        self.read_data()
        self.outlier_removal()
        self.split_data()
        self.save_data()

@click.command()
@click.argument('input_path', type=click.Path())
@click.argument('output_path', type=click.Path())
def main(input_path, output_path):
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    params_file = home_dir.as_posix() + './params.yaml'
    params = yaml.safe_load(open(params_file)).get('make_dataset', {})

    seed = params.get('seed')
    test_size = params.get('test_size')

    if seed is None or test_size is None:
        print('ERROR : seed or test_size not found in params file')
        sys.exit(1)

    input_data_path = home_dir.as_posix() + input_path
    output_data_path = home_dir.as_posix() + output_path
    data_ingestion = DataIngestion(input_data_path, output_data_path, seed, test_size)
    data_ingestion.preprocess()

if __name__ == '__main__':
    main()