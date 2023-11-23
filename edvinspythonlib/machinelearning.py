
'''
Functions and classes for course Machine Learning and Data Mining
'''
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn import model_selection

#data science
def standardizedata(variable):
        '''Standardizes the variable'''
        return (variable-np.mean(variable,axis=0))/np.std(variable,axis=0)

def PCAe(X, components=2):
    '''
    Performs principal component analysis on the data X
    Returns the components and explained varience ratio
    '''
    #standardize data
    X = standardizedata(X)
    
    #do pca on X
    pca = PCA(n_components=components)
    X = pca.fit_transform(X)

    #return components and explanied vaiance ratio
    return X, pca.explained_variance_ratio_

#k-fold cross validation
def kFoldCV(Xtrain,ytrain,k,modelClass, params):
    CV = model_selection.KFold(n_splits=k,shuffle=True)
    err = []
    for train_index,test_index in CV.split(Xtrain,ytrain):
        X_train = Xtrain[train_index]
        y_train = ytrain[train_index]
        X_test = Xtrain[test_index]
        y_test = ytrain[test_index]

        model = modelClass()
        model.fit(X_train,y_train,params)
        MSE = sum(np.power(y_test-model.predict(X_test),2))
        err.append(MSE)
    
    return np.mean(err)

#regression
class baselineRegression:
    '''
    Baseline regression class
    '''
    def __init__(self):
        self.w = None

    def fit(self,Xtrain,ytrain,param):
        self.w = np.mean(ytrain)
    
    def predict(self,Xtest):
        return self.w*np.ones(np.shape(Xtest,axis=0))
    
    def residualPlot(self,Xtest,ytest,yrange=10):
        ypred = self.predict(Xtest)
        for i in range(len(ypred)):
            ypred[i] = ypred[i]+random.uniform(-0.3,0.3)
        plt.scatter(ypred,ytest-ypred)
        plt.ylim(-yrange,yrange)
        plt.plot([min(ypred),max(ypred)],[0,0],color='black',linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residual')
        plt.show()

class linearRegression:
    '''
    Linear regression class
    Uses methods of:
    fit(Xtrain,ytrain,reg=0,standardize=False)
    predict(Xtest)
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

class ANN:
    def __init__(self):
        print("WIP!!")

#Fit fourier series to signal WIP

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