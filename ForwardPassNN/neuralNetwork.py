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
        self.N=0
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
        self.threshold=1
        return
    def initParam(self,size):
        #xavier init
        n_in=size
        n_out=self.NperL[0]
        limit = np.sqrt(6 / (n_in + n_out))
        self.connectionWeights.append(np.random.uniform(-limit, limit, (size, self.NperL[0])))
        #self.connectionWeights.append(np.random.randn(size, self.NperL[0]) * 0.01)
        self.biases.append(np.random.randn(self.NperL[0]) * 0.01)
        for elem in range(self.hidden-1):
            for i in range(self.NperL[elem]):
                if elem!= (self.hidden-1):
                    n_in= self.NperL[elem]
                    n_out=self.NperL[elem+1]
                    limit = np.sqrt(6 / (n_in + n_out))
                    weights=np.random.uniform(-limit, limit, (self.NperL[elem], self.NperL[elem+1]))
                    #weights=np.random.randn(self.NperL[elem], self.NperL[elem+1]) * 0.01
                    bias=np.random.randn(self.NperL[elem+1]) * 0.01    
           
            self.biases.append(bias)
            self.connectionWeights.append(weights)
        
        #self.connectionWeights.append(np.random.randn(self.NperL[-1], 1) * 0.01)
        return
    def clipGradients(self,gradients, max_value):
        total_norm = np.linalg.norm([np.linalg.norm(grad) for grad in gradients])
        scale_factor = max_value / (total_norm + 1e-6)  # Adding a small value to avoid division by zero
    
        if total_norm > max_value:
            clipped_gradients = [grad * scale_factor for grad in gradients]
            return clipped_gradients
        return gradients

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
            
            print(self.N)
            print("out")
            print(out[0])
            self.N+=1
            
            results.append(actOut)
        results=np.array(results)
        return results, vanillaOut
    def backP(self,pred,vanilla,labels):
        #vanilla = pre activation values
        error=0
        mg=0
        gradients=[]
        dlossWeights = [None] * len(self.connectionWeights)
        dlossBiases = [None] * len(self.biases)

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
            dlossWeights[i] = np.dot(vanilla[i].T, gradients)
            dlossBiases[i] = np.sum(gradients, axis=0)  # Sum across the mini-batch

            gradients = np.dot(gradients, self.connectionWeights[i].T)
            gradients =self.clipGradients(gradients,self.threshold)

            print(self.N)
            print(i)
            print("gradient")
            print(gradients[0])
            self.N+=1
            
            gradients *= Dsigmoid(vanilla[i])
        self.error.append(error)
        return dlossWeights, dlossBiases
        
    def train(self,data):
        #parse and init data
        points=data[0]
        labels=data[1]
        samples, features= data[0].shape
        if not self.loaded:
            self.initParam(features)
        size=(samples*0.05)
        #minibatches init
        indices=np.arange(len(data[0]))
        np.random.shuffle(indices)
        pointsShuffled=[points[i] for i in indices]
        labelsShuffled=[labels[j] for j in indices]
        stInd=0
        #training loop
        for iter in range(self.it):
            #minibatch
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
            #print(self.connectionWeights)

            for i in range(len(self.connectionWeights)):
                #"learning"
                self.connectionWeights[i] -= self.ln * D[0][i]

                print(self.N)
                print("weights")
                print(self.connectionWeights[i])
                self.N+=1

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


