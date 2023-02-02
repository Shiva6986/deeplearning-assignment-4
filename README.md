# deeplearning-assignment-4
video demo link:https://drive.google.com/drive/folders/1nBV7KGnWgAqn9hqWSxq5qHFfEm1Jyj5S?usp=share_link

DESCRIPTION:
PROGRAM1: We have done the data manipulation and executed the program perfectly
all the null values,rows, calaories, pulse, df modified and maxplus scatter plot all are been executed.

import pandas as pd
df=pd.read_csv("data.csv")
dfmean_value=df['Calories'].mean()
df.isnull().sum()
df['Calories'].fillna(value=mean_value,inplace=True)
df
df.Duration.describe()
df
df.Pulse.describe()
df
df[(df['Calories']>500) & (df['Calories']<1000)]
df
df_modified=df.drop("Maxpulse",axis=1)
df
df=df.drop("Maxpulse",axis=1)
df
df["Calories"] = df["Calories"].astype(float).astype(int)
df
df.plot.scatter(x = 'Duration', y = 'Calories')
final output will be arrived perfecltly.

PROGRAM 2:
It is linear regression found the Salary data.csv
splited the data in train_test partitions,such that 1/3 of the data is reserved as test subset.
trained and predicted the model 
calculated the mean_squared error
Then we have visualized both train and test using scatter plot.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from google.colab import files
#uploaded = files.upload()
datasets=pd.read_csv('Salary_Data.csv')
datasets+
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test,Y_pred)
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),)
plt.title('Salary vs Experience(Train set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs experience(Test set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()
perfect output has been arrived and executed without errors.
