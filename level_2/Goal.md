🎯 Goal: Tune Hyperparameters of XGBClassifier
Instead of guessing values like n_estimators=10000 or max_depth=20, we’ll use GridSearchCV to:
Try many parameter combinations
Use cross-validation to test them properly
Find the best config that gives the highest validation accuracy


let’s take all those new feature boosts you've built:

✅ TicketGroupSize
✅ FarePerPerson
✅ (and previously) Title, Deck, Alone, FamilySize

And now tune your model to squeeze out the maximum accuracy possible.