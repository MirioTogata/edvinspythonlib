
'''
Functions and classes for couse Algorithms and Datastructures 1+2
'''
import numpy as np

class Stack:
    def __init__(self, arr=[]):
        self.stack = arr
    
    def insert(self,x):
        self.stack.append(x)
    
    def extract(self):
        save = self.stack[len(self.stack)-1]
        self.stack = self.stack[:len(self.stack)-1]
        return save
    
    def __repr__(self):
        return f'{self.stack}'

class Queue:
    def __init__(self,arr=[]):
        self.queue = arr
    
    def insert(self,x):
        self.queue.append(x)
    
    def extract(self):
        save = self.queue[0]
        self.queue = self.queue[1:len(self.queue)]
        return save
    
    def __repr__(self):
        return f'{self.queue}'

def SA(X,Y,d,A,order="ACGT"):
    M = np.zeros((len(X)+1,len(Y)+1))
    for i in range(1,len(X)+1):
        M[i,0] = i*d
    
    for j in range(1,len(Y)+1):
        M[0,j] = j*d

    for i in range(1,len(X)+1):
        for j in range(1,len(Y)+1):
            M[i,j] = min(M[i-1,j]+d,M[i,j-1]+d,A[order.find(X[i-1]),order.find(Y[j-1])]+M[i-1,j-1])

    return M.T

