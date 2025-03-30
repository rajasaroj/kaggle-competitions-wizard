# Predicting performs just by using Pclass,Age,Fare,Parch,SibSp and droping all the NA rows

import matplotlib.pyplot as plt
import pandas as pd
from ModelRepo import MyModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

features = ['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X_FULL = pd.read_csv(r"C:\Users\Home\pythonProject\kaggle\TitanicSurvivor\resources\data\titanic\train.csv")
X_TEST_FULL = pd.read_csv(r"C:\Users\Home\pythonProject\kaggle\TitanicSurvivor\resources\data\titanic\test.csv")
X_FULL = X_FULL.dropna(subset=['Survived'], axis=0)
Y = X_FULL.Survived
X_FULL = X_FULL[features[1:]]

print(features[1:])


x_train, x_valid, y_train, y_valid = train_test_split(X_FULL, Y, train_size=0.8, test_size=0.2, random_state=0)

my_imputer = SimpleImputer(strategy='most_frequent')
x_train_imp = pd.DataFrame(my_imputer.fit_transform(x_train, y_train))
x_valid_imp = pd.DataFrame(my_imputer.transform(x_valid))

# restore the columns
x_train_imp.columns = x_train.columns
x_valid_imp.columns = x_valid.columns

# Create mask to count null column wise
# missing_val_column_mask = (x_train_imp.isnull().sum())
# print(missing_val_column_mask[missing_val_column_mask > 0])

# get object columns for encoding
obj_columns_mask = (X_FULL.dtypes == 'object')
obj_cols = list(obj_columns_mask[obj_columns_mask].index)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
x_train_imp_oh = pd.DataFrame(OH_encoder.fit_transform(x_train_imp[obj_cols]))
x_valid_imp_oh = pd.DataFrame(OH_encoder.transform(x_valid_imp[obj_cols]))

# One-hot encoding removed index; put it back
x_train_imp_oh.index = x_train_imp.index
x_valid_imp_oh.index = x_valid_imp.index

# Remove categorical columns (will replace with one-hot encoding)
x_num_train = x_train_imp.drop(obj_cols, axis=1)
x_num_valid = x_valid_imp.drop(obj_cols, axis=1)

# Add one-hot encoded columns to numerical features
x_train_imp_oh = pd.concat([x_train_imp_oh, x_num_train], axis=1)
x_valid_imp_oh = pd.concat([x_valid_imp_oh, x_num_valid], axis=1)

# Ensure all columns have string type
x_train_imp_oh.columns = x_train_imp_oh.columns.astype(str)
x_valid_imp_oh.columns = x_valid_imp_oh.columns.astype(str)


my_model = MyModel()
my_model.score_and_evaluate(x_train_imp_oh, y_train, x_valid_imp_oh, y_valid)


# model_7 mae=0.20237430167597767
# model_1 mae=0.2122491764269615
# model_2 mae=0.21394122052592313
# model_3 mae=0.2168715083798883
# model_6 mae=0.23123290415826242
# model_4 mae=0.2412382528191676
# model_5 mae=0.2412382528191676