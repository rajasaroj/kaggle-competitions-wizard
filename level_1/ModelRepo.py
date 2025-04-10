import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np


class MyModel:

    def __init__(self):
        self.model_1 = RandomForestRegressor(n_estimators=50)
        self.model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
        self.model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
        self.model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
        self.model_5 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
        self.model_6 = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, random_state=0)
        self.model_7 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', max_depth=10,
                                             min_samples_split=10, random_state=0)
        self.models = [self.model_1, self.model_2, self.model_3, self.model_4, self.model_5, self.model_6, self.model_7]

    def score_and_evaluate(self, x_train, y_train, x_val, y_val):
        dt = {}
        i = 0
        for model in self.models:
            i += 1
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            mae = mean_absolute_error(y_val, y_pred)
            dt['model_' + str(i)] = mae
            # print(f"model={i} mae={mae}")

        ls = dict(sorted(dt.items(), key=lambda item: item[1], reverse=False))
        for x in ls.items():
            print(f"{x[0]} mae={str(x[1])}")
