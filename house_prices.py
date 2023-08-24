import torch 
import numpy as np
import csv
import matplotlib.pyplot as plt

INPUTDIMENSION = 1
OUTPUTDIMENSION = 1
LEARNINGRATE = -0.0001
EPOCHS = 100

xs = []
smn = []
ys = []

with open('Housing.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        try:
            temp = []
            smn.append(int(row[1]))
            for item in row[1:]:
                if (item == 'yes' or item == "furnished"):
                    temp.append(1)
                elif (item == "no" or item == 'unfurnished'):
                    temp.append(0)
                elif (item == 'semi-furnished'):
                    temp.append(0.5)
                else:
                    temp.append(int(item))
                    
            xs.append(temp)
            ys.append(int(row[0]))
        except (ValueError):
            continue            
        
xs = torch.tensor(xs, dtype=torch.float32, requires_grad=True)#.reshape(-1,1)
ys = torch.tensor(ys, dtype=torch.float32, requires_grad=True).reshape(-1,1)

# print(xs.shape)

model = torch.nn.Linear(12,1)
mse = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=0.0000000315)

# print(xs[:1].detach().numpy()[0][0])

# ypred = model(xs)
# print(ypred)

for epoch in range(1_000_000):
    # forward pass
    ypred = model(xs)
    loss = mse(ypred, ys)
    
    # reset grads
    optim.zero_grad()
    
    loss.backward()
    
    # optimization
    optim.step()
        
    print(epoch, loss.item())
    
    
print("---------------------")    
print("Tesing")    
print("---------------------")    
test = [
    [7800, 3, 2, 2, 1, 0, 0, 0, 0, 0, 1, 0.5],
    [6000, 4, 1, 2, 1, 0, 1, 0, 0, 2, 0, 0.5],
    [6600, 4, 2, 2, 1, 1, 1, 0, 1, 1, 1, 0],
    [8500, 3, 2, 4, 1, 0, 0, 0, 1, 2, 0, 1],
    [4600, 3, 2, 2, 1, 1, 0, 0, 1, 2, 0, 1],
]
test = torch.tensor(test, dtype=torch.float32)    
output = model(test)
print(output)    

# smn = torch.tensor(smn, dtype=torch.float32).reshape(-1,1)
# plt.clf()
# plt.plot(smn , ys.detach(), 'go', label='True data', alpha=0.5)
# plt.plot(smn, ypred.detach(), '--', label='Predictions', alpha=0.5)
# plt.legend(loc='best')
# plt.show()  