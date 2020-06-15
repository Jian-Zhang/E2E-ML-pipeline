import numpy as np
import pandas as pd
import os
# Add horovod with torch import
import horovod.torch as hvd
from datetime import datetime
#%matplotlib inline

hvd.init()


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
# travel vectors between start and end points for the taxi ride, in both longitude and latitude coordinates 
# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.

# remove the bizzare travelling distance
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
add_travel_vector_features(train_df)

#plot a subset of travel vector to see its distribution 
#plot = train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')

#there are some further data processing skiped below

#We expect most of these values to be very small (likely between 0 and 1) since it should all 
#be differences between GPS coordinates within one city. For reference, one degree of latitude is about 69 miles. 
#However, we can see the dataset has extreme values which do not make sense. 
#Let's remove those values from our training set. Based on the scatterplot, 
#it looks like we can safely exclude values above 5 (though remember the scatterplot is only showing the first 2000 rows...)

#print('Old size: %d' % len(train_df))
#train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]
#print('New size: %d' % len(train_df))


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

#A sequential container. Modules will be added to it in the order they are passed in the constructor. 
#Alternatively, an ordered dict of modules can also be passed in.

#一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
#同时以神经网络模块为元素的有序字典也可以作为传入参数。

#a three layer NN model 

model = nn.Sequential(nn.Linear(2, 10),
                     nn.Linear(10, 5),
                      nn.Linear(5, 1))

criterion = torch.nn.MSELoss()

# bigger learning rate 
optimizer1 = torch.optim.SGD(model.parameters(), lr=0.01)

#Add Horovod Distributed Optimizer
optimizer_horovod1 = hvd.DistributedOptimizer(optimizer1, named_parameters=model.named_parameters())
# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

X = np.stack((train_df.abs_diff_latitude.values,train_df.abs_diff_longitude.values)).T
X = torch.from_numpy(X)
X = X.type(torch.FloatTensor)

y = torch.from_numpy(train_df.fare_amount.values.T)
y = y.type(torch.FloatTensor)
y.unsqueeze_(-1)
X_train, X_evalutation, y_train, y_evalutation = train_test_split(X, y, test_size = 0.3, random_state = 0)
print("Train Size:")
print(len(X_train))

for epoch in range(3):
    # Forward Propagation
    y_pred = model(X_train)
    # Compute and print loss
    loss = criterion(y_pred, y_train)
    print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    optimizer_horovod1.zero_grad()
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    optimizer_horovod1.step()

    # smaller LR 
optimizer2 = torch.optim.SGD(model.parameters(), lr=0.001)

# Met some issue when using horovord, so we switched to single nodes 
# Next: Figure out the reason 
optimizer_horovod2 = hvd.DistributedOptimizer(optimizer2, named_parameters=model.named_parameters())
# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

for epoch in range(3):
    # Forward Propagation
    y_pred = model(X_train)
    # Compute and print loss
    loss = criterion(y_pred, y_train)
    print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    
    
    optimizer_horovod2.zero_grad()
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    optimizer_horovod2.step()

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
