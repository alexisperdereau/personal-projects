# Import modules
import pandas as pd
import seaborn as sns

# Figures inline and set visualization style
sns.set_theme()

# Import data
data = pd.read_csv("https://www.stat4decision.com/telecom.csv")


# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])


data.info()

# Impute missing numerical variables
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

# Check out info of data
data.info()

print(survived_train)

