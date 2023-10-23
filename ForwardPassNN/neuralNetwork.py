import numpy as np
import random as rand
import matplotlib.pyplot as Matplot
import os

def sigmoid(x):
    x = np.clip(x, -709, 709)  # clip to avoid overflow with exp
    s = 1 / (1 + np.exp(-x))
    #s=(1/(1+np.e**(-x)))
    #res=[]
    #for el in x :
     #   print(el)
    
      #  if el >= 0:
           # s= 1 / (1 + np.exp(-el))
       # else:
        #    z = np.exp(el)
         #   s = z / (1 + z)
        #res.append(s)
    #return np.array(res)
    return s
    
    

def Dsigmoid(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def sigmoid_step(x):
    ret= np.where(x>0.5,1,0)
    return ret


def reluF(x):
    ret= np.where(x>0,x,0)
    return ret

def DReLU(x):
    return (x > 0).astype(float)

def logLoss(pred, label):
    epsilon = 1e-15
    pred = np.clip(pred, epsilon, 1 - epsilon)
    i=0
    i=(-label*np.log(pred) - (1-label)*np.log(1-pred))
    return i


class FeedForwardNN:
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
        self.threshold=10
        return
    def initParam(self,size):
        self.connectionWeights.append(np.random.randn(size, self.NperL[0]) * 0.01)
        self.biases.append(np.random.randn(self.NperL[0]) * 0.01)
        for elem in range(self.hidden-1):
            for i in range(self.NperL[elem]):
                if elem!= (self.hidden-1):
                    weights=np.random.randn(self.NperL[elem], self.NperL[elem+1]) * 0.01
                    bias=np.random.randn(self.NperL[elem+1]) * 0.01    
            #bias=p.biasPerceptron(1)
            self.biases.append(bias)
            self.connectionWeights.append(weights)
        #print(self.connectionWeights)
        #print("\n")
        #print(self.biases)
        #self.connectionWeights.append(np.random.randn(self.NperL[-1], 1) * 0.01)
        return
    def forwardPass(self,Data):
        results=[]
        vanillaOut=[]
        #first step
        out=np.dot(Data, self.connectionWeights[0]) + self.biases[0]
        vanillaOut.append(out)
        actOut=self.act(out)
        for ind in range(self.hidden-1):
            out=np.dot(actOut,self.connectionWeights[ind+1])+self.biases[ind+1]
            
            vanillaOut.append(out)
            actOut=self.act(out)
            #print(out)
            #print(actOut)
            results.append(actOut)
        results=np.array(results)
        return results, vanillaOut
    def backP(self,pred,vanilla,labels):
        error=0
        mg=0
        gradients=[]
        dloss_dweights = [None] * len(self.connectionWeights)
        dloss_dbiases = [None] * len(self.biases)

        for ind,elem in enumerate(pred):
            error+= logLoss(elem,labels[ind])
            g=(labels[ind]-elem)*2
            mg+=g
            gradients.append(g)
        gradients=np.array(gradients)
        #print(vanilla)
        mg=mg/len(pred)
        for i in reversed(range(len(self.connectionWeights))):
             # Gradient of the loss w.r.t. weights and biases
            #print(vanilla)
            dloss_dweights[i] = np.dot(vanilla[i].T, gradients)
            dloss_dbiases[i] = np.sum(gradients, axis=0)  # Sum across the mini-batch

            gradients = np.dot(gradients, self.connectionWeights[i].T)
            if np.linalg.norm(gradients) > self.threshold:
                 gradients *= self.threshold / np.linalg.norm(gradients)
           # print(i)
            #print(vanilla[i])
            
            #print(vanilla[i][0])
            #if i != 0:
            gradients *= Dsigmoid(vanilla[i])
        self.error.append(error)
        return dloss_dweights, dloss_dbiases
        
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
            print(iter)
            ind=stInd+int(size)
            miniBatch= pointsShuffled[stInd:ind]
            lminiBatch=labelsShuffled[stInd:ind]
            if stInd==len(pointsShuffled)-size:
                stInd=0
            else:
                stInd+=int(size)
            #training
            pred=self.forwardPass(miniBatch)
            D=self.backP(pred[0][-1],pred[1],lminiBatch)
            print(self.connectionWeights)

            for i in range(len(self.connectionWeights)):
                #print(D[0][i])
                self.connectionWeights[i] -= self.ln * D[0][i]
                self.biases[i] -= self.ln * D[1][i]

        return self.connectionWeights, self.biases 
    
    def predict(self,data):
        pred=self.forwardPass(data)
        return pred
    def save(self,name):
        os.mkdir(name)
        np.savez(os.path.join(name,f"{name}weights.npz"), self.connectionWeights)
        np.savez(os.path.join(name,f"{name}biases.npz"), self.biases)
        return True
    def load(self,path):
        l=os.listdir(path)
        w=np.load(l[0])
        b=np.load(l[1])
        return w,b


