import pandas as pd

FULL_DF = pd.read_csv(r"../resources/data/titanic/train.csv")
X_TEST_FULL = pd.read_csv(r"../resources/data/titanic/test.csv")

FULL_DF['set'] = 'train'
X_TEST_FULL['set'] = 'test'

FULL_DF = pd.concat([FULL_DF, X_TEST_FULL], sort=False)
ticket_counts = FULL_DF['Ticket'].value_counts()

# STEP: 1
# 🧠 Feature Engineering Booster #1: 🎫 TicketGroupSize
# Objective to group the tickets into large, small and alone group the survial rate of small group is higher that any other group
FULL_DF['TicketGroupSize'] = FULL_DF['Ticket'].map(ticket_counts)


def simplify_group_size(size):

    if size == 1:
        return 'Solo'
    elif size <= 4:
        return 'Small'
    else:
        return 'Large'


FULL_DF['TicketGroupType'] = FULL_DF['TicketGroupSize'].apply(simplify_group_size)


# STEP: 2
# 🪙 Feature Engineering Booster #2: FarePerPerson
# On Titanic, many passengers shared tickets (especially in 3rd class). The Fare column shows the total fare paid, not per person.
# So someone who paid 100 for 4 people really paid 25 per person — a more accurate signal of wealth/class, the wealthy one has higher survival rate

# Avoid division by zero
FULL_DF['TicketGroupSize'] = FULL_DF['TicketGroupSize'].replace(0, 1)

# Create FarePerPerson
FULL_DF['FarePerPerson'] = FULL_DF['Fare'] / FULL_DF['TicketGroupSize']

# ✅ Optional: Create Fare Bins
FULL_DF['FarePerPersonBin'] = pd.qcut(FULL_DF['FarePerPerson'], 4, labels=False)


# STEP: 3
# 📛 Feature Engineering Booster #3: SurnameGroupSurvivalRate (or FamilySurvivalScore)
# ✅ Why This Feature?
# On Titanic, people often traveled with family (same last name). In many cases:
# If some family members survived, others did too
# If a surname group had all deaths, it could signal poor survival odds for others in that group
# This is a way to leverage label correlations inside the train set, without peeking at test labels.

# ✅ Step-by-Step Integration
# 🔹 1. Extract Surname from Name
FULL_DF['Surname'] = FULL_DF['Name'].apply(lambda x: x.split(',')[0].strip())

# 🔹 2. Compute SurnameSurvivalRate from train data only, Compute average survival rate per surname (only from train set)
surname_survival = FULL_DF[FULL_DF['set'] == 'train'].groupby('Surname')['Survived'].mean()

# 🔹 3. Map it to full dataset (train + test), Map to all passengers (some test surnames won't be in train)
FULL_DF['SurnameSurvivalRate'] = FULL_DF['Surname'].map(surname_survival)

# 🔹 4. Handle Missing Values
global_survival_rate = FULL_DF[FULL_DF['set'] == 'train']['Survived'].mean()
FULL_DF['SurnameSurvivalRate'] = FULL_DF['SurnameSurvivalRate'].fillna(global_survival_rate)


# 🔹 5. Basic Feature Engineering
FULL_DF['familySize'] = FULL_DF['SibSp'] + FULL_DF['Parch'] + 1                    # Get total members of person family
FULL_DF['Alone'] = 0                                                             # is the person traveling alone
FULL_DF.loc[FULL_DF['familySize'] == 1, 'Alone'] = 1                              # if the person traveling alone mark 1 else 0
FULL_DF['Title'] = FULL_DF['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)   # Fetch Titles from the name to understand Mr, Miss, Mrs
FULL_DF['Deck'] = FULL_DF['Cabin'].str[0]                                         # Fetch the Deck from Cabin, decks closer to safety boat has high chances of survival

# Normalize titles
FULL_DF['Title'] = FULL_DF['Title'].replace([
    'Lady', 'Countess','Capt', 'Col', 'Don', 'Dr',
    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'
], 'Rare')

FULL_DF['Title'] = FULL_DF['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# Normalize Deck
FULL_DF['Deck'] = FULL_DF['Deck'].fillna('U')  # Fill missing as Unknown
FULL_DF['Deck'] = FULL_DF['Deck'].replace(['T', 'G'], 'Rare')

FULL_DF.drop(columns=['Surname', 'Name'], inplace=True)