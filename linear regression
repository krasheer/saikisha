#Linear Regression on winequality-red

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('winequality-red.csv')
missing_entries=data.isnull().sum().sum()
if missing_entries!=0:
        data=x.dropna()

x=data.drop('quality',axis=1).values
y=data['quality'].values
train_size=int(0.8* len(x))

x_train=x[:train_size]
x_test=x[train_size:]
y_train=y[:train_size]
y_test=y[train_size:]

print("shape of x_train=", x_train.shape)
print("shape of x_test=", x_test.shape)
print("shape of y_train=", y_train.shape)
print("shape of y_test=", y_test.shape)

x_bias = np.c_[np.ones((len(x_train),1)),x_train]
x_test_bias =np.c_[np.ones((len(x_test),1)),x_test]
w= np.linalg.inv(x_bias.T.dot(x_bias)).dot(x_bias.T).dot(y_train)
print(w)
y_predic = x_test_bias.dot(w)

mean_squared_error = ((y_predic - y_test)**2).mean()
print(mean_squared_error)

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_predic, alpha=0.5)
plt.xlabel('actual quality')
plt.ylabel('predicted quality')
plt.title('actual quality vs predicted quality')
plt.show()
