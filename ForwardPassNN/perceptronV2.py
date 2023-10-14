import numpy as np
import random as rand

def sigmoid(x):
    s=(1/(1+np.e**(-x)))
    return s

def sigmoid_step(x):
    ret= np.where(x>0.5,1,0)
    return ret

class Perceptron:

    def __init__(self, learning_rates=0.01, n_iters=1000):
        self.lr=learning_rates
        self.iters=n_iters
        self.activation_fun=sigmoid
        self.weights=None
        self.bias=None
        self.errors=[]
        self.error=0
        return
    def fit(self,X,y):

        n_samples, n_features= X.shape
        self.weights= np.random.randint(0,100,n_features)
        self.bias=rand.random(0,100)
        
        
        for i in range(self.iters):
            ###testing needed
            size=(n_samples*0.10)
            randStart= rand.randint(0,n_samples-size) 
            miniBatch= X[randStart:randStart+size]
            self.error=0
            ###testing needed
            for idx, x_i in enumerate(miniBatch):
                linear_out=np.dot(x_i,self.weights)+ self.bias
                y_predict=self.activation_fun(linear_out)
                #self.error+=logLoss(y_predict, y[idx])
                update= self.lr*(y[idx]-y_predict)
                for f in range(len(self.weights)):
                    self.weights[f]+=update*x_i[f]
                self.bias+= update
            self.errors.append(self.error)
            
        return self.bias, self.weights, self.error
    
    def predict(self,X):
        linear_out=np.dot(X,self.weights)+ self.bias
        y_predict=self.activation_fun(linear_out)
        return y_predict