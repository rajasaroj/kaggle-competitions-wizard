import matplotlib.pyplot as plt
import pandas as pd

X_FULL = pd.read_csv(r"/resources/data/titanic/train.csv")
X_TEST_FULL = pd.read_csv(r"/resources/data/titanic/test.csv")

print(X_FULL.columns)

gender_survivor_stats = X_FULL.where(X_FULL['Survived'] == 1).groupby('Sex').count()
gender_non_survivor_stats = X_FULL.where(X_FULL['Survived'] == 0).groupby('Sex').count()

print(gender_survivor_stats['Survived'])
print(gender_non_survivor_stats['Survived'])



def histo(survived):

    d = X_FULL.where(X_FULL['Survived'] == survived)[['Sex', 'Age']].dropna(axis=0)
    survied_female_age = d[d['Sex'] == 'female']['Age']
    survied_male_age = d[d['Sex'] != 'female']['Age']

    print(len(survied_male_age))
    print(len(survied_female_age))

    plt.figure(figsize=(10, 5))
    plt.hist(survied_male_age, bins=10, edgecolor='black', alpha=0.5, label='Male', color='blue')
    plt.hist(survied_female_age, bins=10, edgecolor='black', alpha=0.5, label='Female', color='red')


    # Labels and Title
    plt.xlabel('Age Groups')
    plt.ylabel('Count of People')
    plt.title('Age Distribution Histogram')
    plt.xticks(range(0, 101, 10))  # Set x-axis ticks for better readability

    # Show Plot
    plt.show()

# histo(1)
# histo(0)

# From the histogram it show Gender, Age play huge role in success of survival chances
Pclass_survivor_stats = X_FULL.where(X_FULL['Survived'] == 1).groupby('Pclass').count()
Pclass_non_survivor_stats = X_FULL.where(X_FULL['Survived'] == 0).groupby('Pclass').count()

print(Pclass_survivor_stats['Survived'])
print(Pclass_non_survivor_stats['Survived'])

# survived / (survived + non survived) * 100 gives the chances of survival per class 1, 2, 3
