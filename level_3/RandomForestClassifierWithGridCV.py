from feature_engineering import AdvanceFeatureEngineering as adf
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


baseline = pd.read_csv(r"../resources/data/titanic/gender_submission.csv")

X_FULL = adf.get_full_df(r"../resources/data/titanic/train.csv")
X_TEST_FULL = adf.get_full_df(r"../resources/data/titanic/test.csv")

X_FULL = X_FULL.dropna(subset=['Survived'], axis=0)
Y = X_FULL.Survived

# Final features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',  # Default Features
            'familySize', 'Alone', 'Title', 'Deck',  # Basic Derived Features
            'TicketGroupSize', 'FarePerPerson']  # Advance Derived Features

X_FULL = X_FULL[features]
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

# Create Model
rf_model = RandomForestClassifier(random_state=0)

# Create Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

# Create GridCV Search Params
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [5, 10, None],
    'model__min_samples_split': [2,5,10],
    'model__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1,
                           n_jobs=-1)  # CV = Cross Validation
grid_search.fit(X_FULL, Y)

print('Best Param: ', grid_search.best_params_)
print('Best Estimator: ', grid_search.best_estimator_)
print('Best Train Score: ', grid_search.best_score_)


def compare_accuracy_with_gender_submission(test_preds):
    # Your predictions DataFrame
    submission = pd.DataFrame({
        'PassengerId': X_TEST_FULL['PassengerId'],
        'Survived': test_preds
    })

    # Merge your prediction with the baseline
    comparison = submission.merge(baseline, on='PassengerId', suffixes=('_model', '_baseline'))

    accuracy_vs_baseline = accuracy_score(comparison['Survived_baseline'], comparison['Survived_model'])
    print(f"Agreement with gender_submission.csv: {accuracy_vs_baseline:.4f}")


rf_best_model = grid_search.best_estimator_
test_preds = rf_best_model.predict(X_TEST_FULL)

compare_accuracy_with_gender_submission(test_preds)

# Results:
# Best Train Score:  0.8327788588286987
# Agreement with gender_submission.csv: 0.8947

def write_to_file(test_preds):
    submission = pd.DataFrame({
        'PassengerId': X_TEST_FULL['PassengerId'],
        'Survived': test_preds
    })

    submission.to_csv('submission.csv', index=False)
    print("submission.csv created ✅")

write_to_file(test_preds.astype(int))