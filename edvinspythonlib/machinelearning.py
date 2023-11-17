
'''
Functions and classes for course Machine Learning and Data Mining
'''
import numpy as np
import matplotlib.pyplot as plt
import random

#data science
def standardizedata(variable):
        '''Standardizes the variable'''
        return (variable-np.mean(variable,axis=0))/np.std(variable,axis=0)

def PCA(X, components="all", explainedvar = 1):
    '''
    Performs principal component analysis on the data X
    Returns the components and eigenvalues in order of most to least important
    Returns as many components as specified or until it's hit the explained variance
    '''
    #standardize data
    X = standardizedata(X)

#regression
class linearRegression:
    '''
    Linear regression class
    Uses methods of:
    fit(Xtrain,ytrain,reg=0,standardize=False)
    predict(Xtest,standardize=False)
    residualPlot(Xtest,ytest,yrange=10)
    '''
    def __init__(self):
        self.w = np.array([])
        self.standardize = False
    
    def fit(self,Xtrain,ytrain,reg=0, standardize=False):
        '''
        Fits the model to the given data
        fitting does not stack and WILL overwrite previous fits
        '''
        #transform data
        self.standardize = standardize
        if standardize:
            Xtrain = standardizedata(Xtrain)
            ytrain = standardizedata(ytrain)
        #append column of ones to X
        Xtrain = np.append(np.ones((Xtrain.shape[0],1)),Xtrain,axis=1)
        #calculate weights
        Xty = Xtrain.T @ ytrain
        XtX = Xtrain.T @ Xtrain
        regI = reg*np.eye(XtX.shape[0])
        regI[0,0] = 0
        self.w = np.linalg.solve(XtX+regI,Xty).squeeze()


    def predict(self,Xtest):
        '''Predicts y based on sum of weights and X'''
        if self.standardize:
            Xtest = standardizedata(Xtest)
        #we append column of ones to X
        Xtest = np.append(np.ones((Xtest.shape[0],1)),Xtest,axis=1)
        return np.dot(Xtest,self.w)
    
    def residualPlot(self,Xtest,ytest,yrange = 10):
        '''Plots residuals of the model'''
        ypred = self.predict(Xtest)
        for i in range(len(ypred)):
            ypred[i] = ypred[i]+random.uniform(-0.3,0.3)
        plt.scatter(ypred,ytest-ypred)
        plt.ylim(-yrange,yrange)
        plt.plot([min(ypred),max(ypred)],[0,0],color='black',linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residual')
        plt.show()

#classification WIP
'''
class logisticRegression:

class KNN

class NaiveBayes

class DecisionTree

class RandomForest

class SVM

class NeuralNetwork

class Kmeans
'''