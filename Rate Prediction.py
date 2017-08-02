
# coding: utf-8

# In[2]:


import tensorflow 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.wrappers import TimeDistributed
from sklearn.model_selection import StratifiedKFold

dataset = pd.read_csv("C:\R.csv")

plt.plot(dataset.rates)
plt.show()
int(len(dataset)) 
dataset['lagrates'] = dataset['rates'].shift(1)
dataset['lagrates1'] = dataset['rates'].shift(2)
dataset['lagrates2'] = dataset['rates'].shift(3)
dataset['lagrates3'] = dataset['rates'].shift(4)
dataset= dataset.fillna(dataset.mean())
train = dataset.iloc[0:69]
test = dataset.iloc[70:88]
#train_X =  train.loc[: , "rates"]
#train_Y = train.loc[:,"lagrates":"lagrates3" ]
#test_X =  test.loc[: , "rates"]
#test_Y = test.loc[:,"lagrates":"lagrates3" ]

# reshape input to be [samples, time steps, features]
#train_Y = np.reshape(np.array(train_X), np.array(train_X.shape[0]), 3)
#test_X = np.reshape(np.array(test_X), np.array(test_X.shape[0]), 3)


train_Y =  train.loc[: , "rates"]
train_X = train.loc[:,"lagrates":"lagrates3" ]
test_Y =  test.loc[: , "rates"]
test_X = test.loc[:,"lagrates":"lagrates3" ]


# reshape input to be [samples, time steps, features]
np.array(train_Y.shape[0])
np.array(train_X.shape[1])

train_X = np.reshape(np.array(train_X), (np.array(train_X.shape[0]), 1, np.array(train_X.shape[1])))
test_X = np.reshape(np.array(test_X), (np.array(test_X.shape[0]), 1, np.array(test_X.shape[1])))



# In[17]:


seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(train_X,train_Y):
    #model Building
    model = Sequential()
    model.add(LSTM(20, input_shape=(1,4)))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(train_X, train_Y, epochs= 50, batch_size=1, verbose=2)

    trainPredict = model.predict(train_X)
    testPredict = model.predict(test_X)

    trainScore = math.sqrt(mean_squared_error(train_Y,trainPredict))
    print('Train Set RMSE: %.2f ' % (trainScore))
    testScore = math.sqrt(mean_squared_error(test_Y,testPredict))
    print('Test Set RMSE: %.2f ' % (testScore))
    trainPredict= pd.DataFrame(trainPredict)
    testPredict =pd.DataFrame(testPredict)
    prediction= pd.concat([trainPredict, testPredict], ignore_index=True)


# In[21]:



plt.plot(dataset.rates)
plt.plot(prediction, color='red')
plt.show()


# In[ ]:





