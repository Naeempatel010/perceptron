
import numpy as np 
from sklearn.utils import shuffle
import random as r
lr = 0.1
file_name = "/User/naeempatel/data.txt"

#Check if the perceptron has converged
def did_converge(w, X, Y):
    counter = 0
    c = True
    for i,x in enumerate(X):
        new_x = np.insert(x, 0, 1).reshape(-1,1)
        y_hat = step(np.dot(new_x.T, w))
        if Y[i] * y_hat <= 0:
            c = False
        else:
            counter+=1
    print("Examples correctly classified : " + str(counter))
    return c

#Step function for activations
def step(z):
    if z > 0:
        return 1.0
    else:
        return -1.0

def read_data_set(filename): 
    with open("data.txt") as file:
        data = file.read().splitlines()
        new_data = [x.split("\t") for x in data]
        X = []
        Y = []
        for data_point in new_data:
            X.append([float(x) for x in data_point[:len(data_point)-1]])
            Y.append(float(data_point[-1]))
    return X,Y
def convert_to_numpy(X):
    X = np.array(X)
    return X


X,Y = read_data_set(file_name)
X = convert_to_numpy(X)
Y = convert_to_numpy(Y)    
X,Y = shuffle(X,Y)
row,col = X.shape
w = np.zeros((col+1, 1))
converged = False
iteration = 1
while not converged:
    print("****Iteration : {} ****".format(iteration))
    random_i = r.randint(0, len(Y)-1)
    x_i_random = np.insert(X[random_i], 0, 1).reshape(-1,1)
    y_hat = step(np.dot(x_i_random.T, w))
    if Y[random_i] * y_hat <=0:
        w += lr * x_i_random * Y[random_i]
    converged = did_converge(w, X, Y)
    iteration+=1

print("Perceptron has converged After Iterations {}".format(iteration))

import matplotlib.pyplot as plt 

fig = plt.figure(figsize=(10,8))
plt.plot(X[:, 0][Y == -1], X[:, 1][Y == -1], 'r+')
plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], 'bo')
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title('Random Classification Data with 2 classes')

plt.show()

def plot_decision_boundary(X, w):
    
    # X --> Inputs
    # theta --> parameters
    
    # The Line is y=mx+c
    # So, Equate mx+c = theta0.X0 + theta1.X1 + theta2.X2
    # Solving we find m and c
    x1 = [min(X[:,0]), max(X[:,0])]
    m = -w[1]/w[2]
    c = -w[0]/w[2]
    x2 = m*x1 + c
    
    # Plotting
    fig = plt.figure(figsize=(10,8))
    plt.plot(X[:, 0][Y==-1], X[:, 1][Y==-1], "r^")
    plt.plot(X[:, 0][Y==1], X[:, 1][Y==1], "bs")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title("Perceptron Algorithm")
    plt.plot(x1, x2, 'y-')
    plt.show()


plot_decision_boundary(X, w)
