import sys
import pathlib
import pickle
import click
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Training():
    def __init__(self, input_train_data_path, input_test_data_path, output_model_path):
        self.input_train_data_path = input_train_data_path
        self.output_model_path = output_model_path
        self.input_test_data_path = input_test_data_path

    def read_data(self):
        self.df_train = pd.read_csv(self.input_train_data_path)
        self.df_test = pd.read_csv(self.input_test_data_path)

    def split_feat_target(self):
        self.y_train = self.df_train['trip_duration']
        self.x_train = self.df_train.drop(columns=['trip_duration'])

        self.y_test = self.df_test['trip_duration']
        self.x_test = self.df_test.drop(columns=['trip_duration'])

    def get_metrics(self, x, y_actual, y_predict):
        mse = round(mean_squared_error(y_actual, y_predict), 2)
        rmse = round(np.sqrt(mean_squared_error(y_actual, y_predict)), 2)
        mae = round(mean_absolute_error(y_actual, y_predict), 2)
        rmspe = round(np.sqrt(np.mean(np.square(((y_actual - y_predict) / y_actual)))), 3)
        r2 = round(r2_score(y_actual, y_predict), 2)
        adjr2 = round(1 - (1 - r2_score(y_actual, y_predict)) * ((x.shape[0] - 1) / (x.shape[0] - x.shape[1] - 1)), 2)

        score_dict = {
            'Mean Square Error': mse,
            'Root Mean Square Error': rmse,
            'Mean Absolute Error': mae,
            'Root Mean Square Percentage Error': rmspe,
            'R2 Score': r2,
            'Adjusted R2 Score': adjr2
        }

        return score_dict

    def create_pipeline(self):
        self.trf1 = ColumnTransformer([
            ('ohe', OneHotEncoder(sparse_output=False), ['vendor_id', 'pickup_weekday', 'store_and_fwd_flag'])
        ],
        remainder='passthrough')
        self.trf1.set_output(transform='pandas')
        #self.model = RandomForestRegressor(n_estimators=100, criterion='squared_error',min_samples_split=2,min_samples_leaf=1 ,verbose=True, max_features='log2', max_depth=6 ,n_jobs=-1)
        #self.model = RandomForestRegressor(n_estimators=100, criterion='squared_error',min_samples_split=15,min_samples_leaf=1 ,verbose=True, max_features='log2', max_depth=16 ,n_jobs=-1)
        #self.model = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, min_samples_split=15, max_features='log2', max_depth=90, bootstrap=True, n_jobs=-1)
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10 ,n_jobs=-1, max_features='sqrt' ,verbose=2)

        self.pipe = Pipeline([
            ('preprocess', self.trf1),
            ('regressor', self.model)
        ], verbose=True)

        self.pipe.fit(self.x_train, self.y_train)

    def prediction(self):
        self.y_train_pred = self.pipe.predict(self.x_train)
        self.y_test_pred = self.pipe.predict(self.x_test)

    def save_model(self):
        pickle.dump(self.pipe, open(self.output_model_path + '/base_model', 'wb'))

    def start_process(self):
        self.read_data()
        self.split_feat_target()
        self.create_pipeline()
        self.prediction()
        self.training_scores = self.get_metrics(self.x_train, self.y_train, self.y_train_pred)
        self.test_scores = self.get_metrics(self.x_test, self.y_test, self.y_test_pred)
        print('Training scores : ', self.training_scores)
        print('Test scores : ', self.test_scores)
        self.save_model()

@click.command()
@click.argument('input_train_path', type=click.Path())
@click.argument('input_test_path', type=click.Path())
@click.argument('output_model', type=click.Path())
def main(input_train_path, input_test_path ,output_model):
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_train_data_path = home_dir.as_posix() + input_train_path
    input_test_data_path = home_dir.as_posix() + input_test_path
    output_model_path = home_dir.as_posix() + output_model

    train = Training(input_train_data_path, input_test_data_path, output_model_path)
    train.start_process()

if __name__ == '__main__':
    main()