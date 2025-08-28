from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('ML/Supervisé/us_salary_dataset.txt', sep=',')
data = data.dropna()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print("----------HEAD----------")
print(data.head())
print("----------COLUMNS----------")
print(data.columns)
print("----------DESCRIBE----------")
print(data.describe())

x = data.Age.values
y = data.Salary.values
print("x : ", x)
print("y : ", y)


x = x.reshape(len(data), 1)
y = y.reshape(len(data), 1)
print("x reshaped : ", x)
print("y reshaped : ", y)

regr = linear_model.LinearRegression()
regr.fit(x, y)

# plot it as in the example at http://scikit-learn.org/
plt.scatter(x, y,  color='orange')
plt.plot(x, regr.predict(x), color='red', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
