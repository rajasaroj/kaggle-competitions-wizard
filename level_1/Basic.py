# Predicting performs just by imputing values and using one hot encoder

import matplotlib.pyplot as plt
import pandas as pd
from ModelRepo import MyModel
from sklearn.model_selection import train_test_split

features = ['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X_FULL = pd.read_csv(r"/resources/data/titanic/train.csv")
X_TEST_FULL = pd.read_csv(r"/resources/data/titanic/test.csv")
X_FULL = X_FULL.dropna(subset=features, axis=0)
Y = X_FULL.Survived
x_train = X_FULL[features]
x_train = x_train.drop("Survived", axis=1)

print(x_train.columns)

# exclude object columns and drop the NA rows
object_columns = x_train.select_dtypes(include=['object']).columns
x_train_reduced = x_train.drop(object_columns, axis=1)

# Split test train set
x_train, x_valid, y_train, y_valid = train_test_split(x_train_reduced, Y, train_size=0.8, test_size=0.2, random_state=0)

my_model = MyModel()
my_model.score_and_evaluate(x_train, y_train, x_valid, y_valid)

# model_7 mae=0.31888111888111886
# model_6 mae=0.38498964024846993
# model_4 mae=0.3889625690130802
# model_5 mae=0.3889625690130802
# model_1 mae=0.3949953379953379
# model_3 mae=0.40311188811188814
# model_2 mae=0.40477433677433683