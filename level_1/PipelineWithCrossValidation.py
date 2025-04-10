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

features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X_FULL = pd.read_csv(r"/resources/data/titanic/train.csv")
X_TEST_FULL = pd.read_csv(r"/resources/data/titanic/test.csv")
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

model_7 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', max_depth=10,
                                min_samples_split=10, random_state=0)


# Bundle Preprocessing and modeling in a pipeline
def mae_without_cross_validation():
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_7)
    ])

    clf.fit(x_train, y_train)
    pred = clf.predict(x_valid)

    print('MAE without cross validation:', mean_absolute_error(y_valid, pred))
    # Answer MAE: 0.20083798882681567


mae_without_cross_validation()





# def mae_with_cross_validation(X_FULL, Y, n_estimator):
#
#     model = RandomForestRegressor(n_estimators=n_estimator, criterion='absolute_error', max_depth=10,
#                                     min_samples_split=10, random_state=0)
#     my_pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('model', model)
#     ])
#
#     # Multiply by -1 since sklearn calculates *negative* MAE
#     scores = -1 * cross_val_score(my_pipeline, X_FULL, Y,
#                                   cv=5,
#                                   scoring='neg_mean_absolute_error')
#
#     print("Average MAE score:", scores.mean())
#     return scores.mean()
#
#
# results = {x: mae_with_cross_validation(X_FULL, Y, x) for x in range(50, 450, 50)}
# plt.plot(list(results.keys()), list(results.values()))
# plt.show()

