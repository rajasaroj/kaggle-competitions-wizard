import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

baseline = pd.read_csv(r"/resources/data/titanic/gender_submission.csv")
X_FULL = pd.read_csv(r"/resources/data/titanic/train.csv")
X_TEST_FULL = pd.read_csv(r"/resources/data/titanic/test.csv")
X_FULL = X_FULL.dropna(subset=['Survived'], axis=0)
Y = X_FULL.Survived

# Feature Engineering
X_FULL['familySize'] = X_FULL['SibSp'] + X_FULL['Parch'] + 1 # Get total members of person family
X_FULL['Alone'] = 0                                          # is the person traveling alone
X_FULL.loc[X_FULL['familySize'] == 1, 'Alone'] = 1           # if the person traveling alone mark 1 else 0
X_FULL['Title'] = X_FULL['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False) # Fetch Titles from the name to understand Mr, Miss, Mrs
X_FULL['Deck'] = X_FULL['Cabin'].str[0]                                       # Fetch the Deck from Cabin, decks closer to safety boat has high chances of survival

# Normalize titles
X_FULL['Title'] = X_FULL['Title'].replace([
    'Lady', 'Countess','Capt', 'Col', 'Don', 'Dr',
    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'
], 'Rare')

X_FULL['Title'] = X_FULL['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

X_FULL['Deck'] = X_FULL['Deck'].fillna('U')  # Fill missing as Unknown
X_FULL['Deck'] = X_FULL['Deck'].replace(['T', 'G'], 'Rare')

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'familySize', 'Alone', 'Title', 'Deck']
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

my_model_1 = XGBClassifier(eval_metric='logloss', random_state=0)
my_model_2 = XGBClassifier(eval_metric='logloss', n_estimators=500, learning_rate=0.01, random_state=0)
my_model_3 = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=0)

def acc_cross_validation():

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', my_model_3)
    ])

    clf.fit(x_train, y_train)

    scores = cross_val_score(clf, X_FULL, Y, cv=5, scoring='accuracy')
    print(scores)

    # Classifiers Accuracy
    # [0.82122905 0.8258427  0.86516854 0.79775281 0.84269663]

# acc_cross_validation() # max 86% accuracy



def acc_with_early_stopping():

    # Step 1: Fit and transform preprocessing manually
    X_train_proc = preprocessor.fit_transform(x_train)
    X_valid_proc = preprocessor.transform(x_valid)

    # Step 2: Train model with early stopping
    model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=0, early_stopping_rounds = 20)
    model.fit(
        X_train_proc, y_train,
        eval_set=[(X_valid_proc, y_valid)],
        verbose=False
    )

    # Step 3: Predict and evaluate
    preds = model.predict(X_valid_proc)
    print("Accuracy:", accuracy_score(y_valid, preds))

    return model


# acc_with_early_stopping() # max 86% accuracy 0.8603351955307262

def feature_engg_test_data():

    # Feature Engineering
    X_TEST_FULL['familySize'] = X_TEST_FULL['SibSp'] + X_TEST_FULL['Parch'] + 1  # Get total members of person family
    X_TEST_FULL['Alone'] = 0  # is the person traveling alone
    X_TEST_FULL.loc[X_TEST_FULL['familySize'] == 1, 'Alone'] = 1  # if the person traveling alone mark 1 else 0
    X_TEST_FULL['Title'] = X_TEST_FULL['Name'].str.extract(r' ([A-Za-z]+)\.',
                                                 expand=False)  # Fetch Titles from the name to understand Mr, Miss, Mrs
    X_TEST_FULL['Deck'] = X_TEST_FULL['Cabin'].str[0]
    # Normalize titles
    X_TEST_FULL['Title'] = X_TEST_FULL['Title'].replace([
        'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
        'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'
    ], 'Rare')

    X_TEST_FULL['Title'] = X_TEST_FULL['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    X_TEST_FULL['Deck'] = X_TEST_FULL['Deck'].fillna('U')  # Fill missing as Unknown
    X_TEST_FULL['Deck'] = X_TEST_FULL['Deck'].replace(['T', 'G'], 'Rare')

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'familySize', 'Alone', 'Title', 'Deck']
    return X_TEST_FULL[features]


def write_to_file(test_preds):
    submission = pd.DataFrame({
        'PassengerId': X_TEST_FULL['PassengerId'],
        'Survived': test_preds
    })

    submission.to_csv('submission.csv', index=False)
    print("submission.csv created ✅")


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

def run_on_real_world():
    model = acc_with_early_stopping()
    x_test = feature_engg_test_data()
    x_test_proc = preprocessor.transform(x_test)
    test_preds = model.predict(x_test_proc)
    compare_accuracy_with_gender_submission(test_preds)
    write_to_file(test_preds)

run_on_real_world()


# Synopsis:
# Accuracy: 0.8603351955307262
# Agreement with gender_submission.csv: 0.8708
# submission.csv created ✅
# On Kaggle Accuracy: 70%
# Score: 0.77033
# Rank: 10923