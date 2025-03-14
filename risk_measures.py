import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

dax_returns = pd.read_csv(r"C:\Users\jose3\Downloads\dax_returns.txt", delim_whitespace=True,names=["index", "dax_return"], skiprows=1)

stock_price = 100

losses = np.exp(dax_returns["dax_return"].to_numpy()-1)*stock_price

weights = np.abs(losses)/np.sum(np.abs(losses))

#Computes value at risk
def VAR(alpha,loss):
    sorted_loss = np.sort(loss)
    size = sorted_loss.size
    i = int(size*alpha)
    return sorted_loss[i-1]

#Computes expected shortfall
def ES(alpha,loss):
    sorted_loss = np.sort(loss)
    size = sorted_loss.size
    return np.mean(sorted_loss[int(size*alpha)-1:])




#Make 95% CI for expected shortfall and value at risk with weighted bootstrap
def CI(loss,n,alpha,beta):
    var = np.zeros(n)
    es = np.zeros(n)
    for i in range(n):
        data = np.random.choice(loss, size=1000, replace = True, p=weights)
        var[i] += VAR(alpha,data)
        es[i] += ES(alpha,data)
    var = np.sort(var)
    es = np.sort(es)
    right = int(beta*n)
    left = n-right
    return [var[left],var[right]], [es[left],es[right]]

alpha = 0.99
beta = 0.95
sim = 1000

print(f"value at risk for alpha={alpha}: {VAR(alpha,losses)}")
print(f"expected shortfall for alpha={alpha}: {ES(alpha,losses)}")

print(f"95% confidence interval for value at risk for alpha={alpha}: {CI(losses,sim,alpha,beta)[0]}")
print(f"95% confidence interval for expected shortfall for alpha={alpha}: {CI(losses,sim,alpha,beta)[1]}")

