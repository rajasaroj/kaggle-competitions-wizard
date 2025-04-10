from LogisticRegressionWithGridCV import log_best_model
from RandomForestClassifierWithGridCV import rf_best_model
from XGBWithGridSearchCV import xgb_best_model
from feature_engineering.AdvanceFeatureEngineering import FULL_DF
from sklearn.metrics import accuracy_score
import pandas as pd

X_TEST_FULL = FULL_DF[FULL_DF['set'] == 'test'].drop(columns=['set'])
baseline = pd.read_csv(r"../resources/data/titanic/gender_submission.csv")

def compare_accuracy_with_gender_submission(test_preds):
    # Your predictions DataFrame
    submission = pd.DataFrame({
        'PassengerId': X_TEST_FULL['PassengerId'],
        'Survived': test_preds
    })

    # Merge your prediction with the baseline
    comparison = submission.merge(baseline, on='PassengerId', suffixes=('_model', '_baseline'))

    accuracy_vs_baseline = accuracy_score(comparison['Survived_baseline'], comparison['Survived_model'])
    print(f"Ensemble Agreement with gender_submission.csv: {accuracy_vs_baseline:.4f}")


# SOFT VOTING ENSEMBLE
prob_rf = rf_best_model.predict(X_TEST_FULL)
prob_log = log_best_model.predict(X_TEST_FULL)
prob_xgb = xgb_best_model.predict(X_TEST_FULL)

# Average Probablities
final_probs = (prob_rf + prob_log + prob_xgb) / 3
final_preds = (final_probs >= 0.5).astype(int)

compare_accuracy_with_gender_submission(final_preds)

# Agreement with gender_submission.csv: 0.8062