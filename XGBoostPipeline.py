import matplotlib.pyplot as plt
import pandas as pd
from ModelRepo import MyModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X_FULL = pd.read_csv(r"C:\Users\Home\pythonProject\kaggle\TitanicSurvivor\resources\data\titanic\train.csv")
X_TEST_FULL = pd.read_csv(r"C:\Users\Home\pythonProject\kaggle\TitanicSurvivor\resources\data\titanic\test.csv")
X_FULL = X_FULL.dropna(subset=['Survived'], axis=0)
Y = X_FULL.Survived
X_FULL = X_FULL[features[1:]]

print(features[1:])

x_train, x_valid, y_train, y_valid = train_test_split(X_FULL, Y, train_size=0.8, test_size=0.2, random_state=0)

categorical_cols = [col for col in x_train.columns if x_train[col].nunique() < 10 and x_train[col].dtype == 'object']
numerical_cols = [col for col in x_train.columns if x_train[col].dtype in ['int64', 'float64']]

print("categorical_cols", categorical_cols)
print("numerical_cols", numerical_cols)

# Create Numerical Transfromer to impute numerical columns
numerical_transformer = SimpleImputer(strategy='most_frequent')

# Create Categorical Transformer to impute categorical columns and impute them with most frequent value if they
# contain null
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

# compose your preprocessing process with Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

my_model_1 = XGBClassifier(eval_metric='logloss', random_state=0)
my_model_2 = XGBClassifier(eval_metric='logloss', n_estimators=500, learning_rate=0.01, random_state=0)
my_model_3 = XGBClassifier(n_estimators=10000, max_depth=20, learning_rate=0.01, eval_metric='logloss', random_state=0)

def mae_without_cross_validation():

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', my_model_3)
    ])

    clf.fit(x_train, y_train)
    pred = clf.predict(x_valid)

    print('MAE without cross validation:', accuracy_score(y_valid, pred))

    # Classifiers Accuracy
    # my_model_1 = 0.8491620111731844
    # my_model_2 = 0.8547486033519553
    # my_model_3 = 0.8603351955307262

mae_without_cross_validation()

# 86% Accuracy Achieved
