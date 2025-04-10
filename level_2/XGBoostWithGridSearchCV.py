# 🎯 Goal: Tune Hyperparameters of XGBClassifier
# Instead of guessing values like n_estimators=10000 or max_depth=20, we’ll use GridSearchCV to:
# Try many parameter combinations
# Use cross-validation to test them properly
# Find the best config that gives the highest validation accuracy


from feature_engineering.AdvanceFeatureEngineering import FULL_DF
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

baseline = pd.read_csv(r"../resources/data/titanic/gender_submission.csv")

X_FULL = FULL_DF[FULL_DF['set'] == 'train'].drop(columns=['set'])
X_TEST_FULL = FULL_DF[FULL_DF['set'] == 'test'].drop(columns=['set'])
X_FULL = X_FULL.dropna(subset=['Survived'], axis=0)
Y = X_FULL.Survived

# Final features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',  # Default Features
            'familySize', 'Alone', 'Title', 'Deck',  # Basic Derived Features
            'TicketGroupSize', 'FarePerPerson', 'SurnameSurvivalRate']  # Advance Derived Features

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
my_model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=0)

# Create Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', my_model)
])

# Create GridCV Search Params
param_grid = {
    'model__n_estimators': [100, 600],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1,
                           n_jobs=-1)  # CV = Cross Validation
grid_search.fit(X_FULL, Y)

print('Best Param: ', grid_search.best_params_)
print('Best Estimator: ', grid_search.best_estimator_)
print('Best Train Score: ', grid_search.best_score_)


# Best Train Score:  0.9764358797313413

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


best_model = grid_search.best_estimator_
test_preds = best_model.predict(X_TEST_FULL)

compare_accuracy_with_gender_submission(test_preds)

# Results:
# Best Train Score:  0.9764358797313413
# Agreement with gender_submission.csv: 0.7943

# 🤔 Interpreting These Results
# 🔹 High Train Accuracy (~97.6%)
# This is expected since:
# You used extensive features (Title, Deck, SurnameSurvivalRate, etc.)
# You tuned model hyperparameters with GridSearchCV
# This score comes from cross-validation, so it’s reliable
#
# 🔹 Test Agreement with Baseline: ~79%
# This means your model agrees with gender_submission.csv on ~79% of test predictions.
#
# ✅ Good news: You’re not blindly copying the baseline
# ✅ Even better: You’re learning different and potentially better rules