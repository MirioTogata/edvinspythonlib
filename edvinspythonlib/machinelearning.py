
'''
Functions and classes for course Machine Learning and Data Mining
'''
import numpy as np
import matplotlib.pyplot as plt

class linearRegression:
    '''Linear regression class
    Uses methods of:
    fit(Xtrain,ytrain,reg=0,normX=False,normy=False)
    predict(Xtest,standardize=False)
    '''
    def __init__(self):
        self.w = np.array([])
    
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
    
    def residualPlot(self,Xtest,ytest):
        '''Plots residuals of the model'''
        if self.standardize:
            Xtest = standardizedata(Xtest)
            ytest = standardizedata(ytest)
        ypred = self.predict(Xtest)
        plt.scatter(ypred,ytest-ypred)
        plt.xlabel('Predicted')
        plt.ylabel('Residual')
        plt.show()

def standardizedata(variable):
        '''Standardizes the variable'''
        return (variable-np.mean(variable,axis=0))/np.std(variable,axis=0)



