import numpy as np
import pandas as pd
import os

from datetime import datetime



time1=datetime.now()
PATH = '/home/sparkuser/jupyter/Bin/NYC_Taxi_Fare/input'
os.listdir(PATH)
train_df = pd.read_csv(f'{PATH}/train.csv')
#data size: 5.4GB 
time2=datetime.now()
data_load_time=time2-time1
print("Data Load Consuming Time:")
print(data_load_time)
print("finished")
train_df.dtypes


time3=datetime.now()
# Check NaNs in the dataset
print(train_df.isnull().sum())


print('Old size %d'% len(train_df))
train_df = train_df.dropna(how='any',axis='rows')
print('New size %d' % len(train_df))

def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
add_travel_vector_features(train_df)



train_df = train_df[(train_df.abs_diff_longitude<5) & (train_df.abs_diff_latitude<5)]
print(len(train_df))

time4=datetime.now()
data_processing_time=time4-time3
print("Data Processing Consuming Time:")
print(data_processing_time)
data_prepare_time=data_load_time+data_processing_time
print("Data prepare Consuming Time:")
print(data_prepare_time)


import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

time5=datetime.now()



model = nn.Sequential(nn.Linear(2, 10),
                     nn.Linear(10, 5),
                      nn.Linear(5, 1))

criterion = torch.nn.MSELoss()

# bigger learning rate 
optimizer1 = torch.optim.SGD(model.parameters(), lr=0.01)



X = np.stack((train_df.abs_diff_latitude.values,train_df.abs_diff_longitude.values)).T
X = torch.from_numpy(X)
X = X.type(torch.FloatTensor)

y = torch.from_numpy(train_df.fare_amount.values.T)
y = y.type(torch.FloatTensor)
y.unsqueeze_(-1)
X_train, X_evalutation, y_train, y_evalutation = train_test_split(X, y, test_size = 0.3, random_state = 0)
print("Train Size:")
print(len(X_train))

for epoch in range(200):
    # Forward Propagation
    y_pred = model(X_train)
    # Compute and print loss
    loss = criterion(y_pred, y_train)
    print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    optimizer1.zero_grad()
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    optimizer1.step()

    # smaller LR 
optimizer2 = torch.optim.SGD(model.parameters(), lr=0.001)



for epoch in range(700):
    # Forward Propagation
    y_pred = model(X_train)
    # Compute and print loss
    loss = criterion(y_pred, y_train)
    print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    
    
    optimizer2.zero_grad()
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    optimizer2.step()

time6=datetime.now()
model_train_time=time6-time5
print("Model Train Consuming Time:")
print(model_train_time)



time7=datetime.now()

def RMSE(x,y):
    criterion = nn.MSELoss()
    loss = torch.sqrt(criterion(x, y))
    return loss
print(X_evalutation)
y_evalutation_result=model(X_evalutation)
print (y_evalutation_result)

rmse=RMSE(y_evalutation_result,y_evalutation)

print("RMSE Value:")
print(rmse)

time8=datetime.now()
evalutation_time=time8-time7
print("Evalutation Consuming Time:")
print(evalutation_time)

print("Data Load Consuming Time:")
print(data_load_time)
print("Data prepare Consuming Time:")
print(data_prepare_time)
print("Model Train Consuming Time:")
print(model_train_time)
print("Evalutation Consuming Time:")
print(evalutation_time)
total_time=data_load_time+data_prepare_time+model_train_time+evalutation_time
print("Total Consuming Time:")
print(total_time)
