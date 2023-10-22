import numpy as np
#import perceptronV2 as p
import random as rand
import matplotlib.pyplot as Matplot
import os

def sigmoid(x):
    s=(1/(1+np.e**(-x)))
    return s

def Dsigmoid(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def reluF(x):
    ret= np.where(x>0,x,0)
    return ret

def logLoss(pred, label):
    epsilon = 1e-15
    pred = np.clip(pred, epsilon, 1 - epsilon)
    i=0
    i=(-label*np.log(pred) - (1-label)*np.log(1-pred))
    return i


class ForwardPassNN:
    def __init__(self, hlayers,distribution,learning_rates=0.01,epochs=10000,func=sigmoid):
        self.act=func
        self.loaded=False
        self.it=epochs
        self.ln=learning_rates
        self.hidden=hlayers
        self.neurons=[]
        self.outPuts=[]
        self.NperL=distribution
        self.connectionWeights=[]
        self.biases=[]
        self.error=[]
        return
    def initParam(self,size):
        self.connectionWeights.append(np.random.randn(size, self.NperL[0]) * 0.01)
        self.biases.append(np.random.randn(self.NperL[elem[0]]) * 0.01)
        for elem in range(self.hidden):
            for i in range(self.NperL[elem]):
                if elem!= (self.hidden-1):
                    weights=np.random.randn(self.NperL[elem], self.NperL[elem+1]) * 0.01
                    bias=np.random.randn(self.NperL[elem+1]) * 0.01    
            #bias=p.biasPerceptron(1)
            self.biases.append(bias)
            self.connectionWeights.append(weights)
        #self.connectionWeights.append(np.random.randn(self.NperL[-1], 1) * 0.01)
        return
    def forwardPass(self,Data):
        results=[]
        vanillaOut=[]
        #first step
        out=np.dot(Data, self.connectionWeights[0]) + self.biases[0]
        vanillaOut.append(out)
        actOut=self.act(out)
        for ind in range(self.hidden):
            out=np.dot(actOut,self.connectionWeights[ind+1])+self.biases[ind+1]
            vanillaOut.append(out)
            actOut=self.act(out)
            results.append(actOut)
        return results, vanillaOut
    def backP(self,pred,vanilla,labels):
        error=0
        mg=0
        gradients=[]
        dloss_dweights = [None] * len(self.connectionWeights)
        dloss_dbiases = [None] * len(self.biases)

        for ind,elem in enumerate(pred):
            error+= logLoss(elem-labels[ind])
            g=(labels[ind]-elem)*2
            mg+=g
            gradients.append(g)
        mg=mg/len(pred)
        for i in reversed(range(len(self.connectionWeights))):
             # Gradient of the loss w.r.t. weights and biases
            dloss_dweights[i] = np.dot(vanilla[i].T, gradients)
            dloss_dbiases[i] = np.sum(gradients, axis=0)  # Sum across the mini-batch

            gradients = np.dot(gradients, self.connectionWeights[i].T)
            #if i != 0:
            gradients *= Dsigmoid(vanilla[i])
        self.error.append(error)
        return dloss_dweights, dloss_dbiases
        
    def errorFun():
        return
    def train(self,data):
        #parse and init data
        points=data[0]
        labels=data[1]
        samples, features= data[0].shape
        if not self.loaded:
            self.initParam(features)
        size=(samples*0.05)
        #minibatches
        indices=np.arange(len(data[0]))
        np.random.shuffle(indices)
        pointsShuffled=[points[i] for i in indices]
        labelsShuffled=[labels[j] for j in indices]
        stInd=0
        #training loop
        for iter in range(self.it):
            ind=stInd+int(size)
            miniBatch= pointsShuffled[stInd:ind]
            lminiBatch=labelsShuffled[stInd:ind]
            if stInd==len(pointsShuffled)-size:
                stInd=0
            else:
                stInd+=int(size)
            #training
            pred=self.forwardPass(miniBatch)
            D=self.backP(pred[0][-1],pred[1][-1],lminiBatch)

            for i in range(len(self.connectionWeights)):
                self.connectionWeights[i] -= self.ln * D[0][i]
                self.biases[i] -= self.ln * D[1][i]

        return self.connectionWeights, self.biases 
    
    def predict(self,data):

        return
    def save(self):
        return
    def load(self):
        return


















'''class ForwardPassNNDEP:
    
    def __init__(self, hlayers,distribution,learning_rates=0.01,epochs=10000):
        self.it=epochs
        self.ln=learning_rates
        self.hidden=hlayers
        self.neurons=[]
        self.outPuts=[]
        #self.out=p.Perceptron(learning_rates=self.ln,n_iters=1)
        self.NperL=distribution
        self.connectionWeights=[]
        self.initP()
        return
    
    def initP(self):
        for elem in range(self.hidden):
            layer=[]
            weights=[]
            for i in range(self.NperL[elem]):
                if elem!= (self.hidden-1):
                    weights=np.random.randn(self.NperL[elem], self.NperL[elem+1]) * 0.01
                pn=p.Perceptron(learning_rates=self.ln,n_iters=1)
                layer.append(pn)
            bias=p.biasPerceptron(1)
            layer.append(bias)
            self.neurons.append(layer)
            self.connectionWeights.append(weights)
        return
    
    def passForward(self,Data):
        passForwardData=[]
        for indL,l in enumerate(self.neurons):
            for indP,p in enumerate(l):
                p.fit(Data[0],Data[1])







        return
    def backPropagation(self):
        return
    def train(self, data):
        for it in range(self.it):

            self.outputs=self.passForward(data)
            self.backPropagation()
            pass
        return
    def predict(self):
        return
    


    '''