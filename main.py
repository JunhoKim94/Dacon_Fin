import numpy as np
import pandas as pd
from torch import nn, tensor, from_numpy
import matplotlib.pyplot as plt
import torch
from utils.preprocessing import processing
from models.model import Regression_Model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import time


print("\n ==============================> Training Start <=============================")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(torch.cuda.is_available())
if torch.cuda.device_count() > 1:
    print("\n ====> Training Start")


train_path = "./data/train.csv"
test_path = "./data/test.csv"

input_var = ['in_out','6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride', "bus_route_id", "station_code",
       '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff', '8~9_takeoff',
       '9~10_takeoff', '10~11_takeoff', '11~12_takeoff','weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
       'weekday_5', 'weekday_6', 'dis_jejusi', 'dis_seoquipo']


data_x, data_y = processing(train_path, input_var = input_var)

'''
data_x = np.array(data_x, dtype = np.float32)
data_y = np.array(data_y, dtype = np.float32)
'''

# data 섞기
total_size = data_x.shape[0]
perm = np.random.permutation(total_size)
x = data_x.iloc[perm,:]
y = data_y.iloc[perm]

# train, validation split
train_val_ratio = 0.8 # 0.8
tmp = int(total_size*train_val_ratio)

train_x = x.iloc[:tmp,:]
val_x = x.iloc[tmp:,:]
train_y = y.iloc[:tmp]
val_y = y.iloc[tmp:]

del x, y, perm

test_x= processing(test_path, training = False, input_var = input_var)

iteration = 20
batch_size = 3
learning_rate= 0.001

model = Regression_Model(inputs = len(train_x.columns), hidden_layer = 512, drop_out = 0.5)
criterion = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-8)

model.to(device)

model.train()
loss_stack = []
# Training loop
ts = time.time()

for epoch in range(iteration):
    # 1) Forward pass: Compute predicted y by passing x to the model
    for i in range(len(train_x) // batch_size):
        seed = np.random.choice(total_size, batch_size)
        x_data = tensor(np.array(train_x.iloc[seed,:], dtype = np.float32)).to(device)
        y_data = tensor(np.array(train_y.iloc[seed], dtype = np.float32)).to(device)
        
        y_pred = model(x_data)

        # 2) Compute and print loss
        loss = criterion(y_pred, y_data)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del x_data, y_data

    print(f'Epoch: {epoch} | Loss: {loss.item()} ')
    loss_stack.append(loss.item())

print("time spend %d"%(time.time() - ts))

test_x = tensor(np.array(test_x, dtype = np.float32)).to(device)
model.eval()
y_pred = model(test_x)
print(y_pred)

epochs = np.linspace(1, 100, 100)
plt.figure(figsize = (10, 8))
plt.plot(epochs, loss_stack)
plt.show()

'''
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

fit = svr_rbf.fit(train_x, train_y)
y_pred = fit.predict(test_x)
print(y_pred)

#Random Forest
rf = RandomForestRegressor(random_state=1217)
rf.fit(train_x, train_y)
y_pred = rf.predict(test_x)
print(y_pred)

'''