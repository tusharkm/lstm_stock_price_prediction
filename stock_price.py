# 2012 to 2016
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Part 1- Data Preprocessing

#importing training set
training_set=pd.read_csv('Google_Stock_Price_Train.csv')

#extract open value from the trainng data
training_set=training_set.iloc[:,1:2].values 

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set=sc.fit_transform(training_set)

#Getting the input and output
X_train= training_set[0:1257]                   
Y_train=training_set[1:1258]               


#Reshaping
X_train=np.reshape(X_train,(1257,1,1))



#Part-2 Building RNN
#importing keras library and packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Initalizing RNN

regressor=Sequential()

regressor.add(LSTM(units=4,activation='sigmoid', input_shape=(None,1)))

#Adding output layer (default argument)
regressor.add(Dense(units=1))

#Compile LSTM
regressor.compile(optimizer='adam',loss='mean_squared_error')


#Fitting the RNN on training set
regressor.fit(X_train,Y_train,batch_size=32,epochs=200)


#Part 3-Making Prediction and Visualizing Results

#Getting real Stock price for 2017
test_set=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=test_set.iloc[:,1:2].values


#Getting predicted Stock price for 2017
inputs=real_stock_price
inputs=sc.transform(inputs)
inputs=np.reshape(inputs,(20,1,1))  #scaling the values

predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) #scaling to input values

#Visualize the results
plt.plot(real_stock_price,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price,color='green',label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


#Part 4- Evaluating the RNN
# since it is linear regression problem we will evaluate RMSE

import math
from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

#expressing RMSE in percentage
rmse=rmse/800        # 800 becasue it is average value