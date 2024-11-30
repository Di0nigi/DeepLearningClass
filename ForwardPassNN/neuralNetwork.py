import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.power(np.e,(-x)))
    return s

def countWeights(d,l):
    out=[]
    out.append(d[0]*l[0])
    s=0
    for ind in range(len(l)-1):
        s=l[ind]*l[ind+1]
        out.append(s)
    out.append(l[-1]*d[1])
    return out

class FeedForwardNN:
    def __init__(hlayers,dims,self,learningRate=0.001,epochs=10000,func=sigmoid, initWeights = "zero"):
        self.layers = hlayers
        self.dims = dims
        self.lr = learningRate
        self.ep = epochs
        self.activation = func
        self.weights=[]
        self.biases =[]
        self.numLayers=len(hlayers)+2
        self.wMatShape=countWeights(self.dims,self.layers)
        self.biases = np.array([[0 for b in elem] for elem in self.dims[0]+self.layers+self.dims[1]],dtype=object)
        if initWeights =="zero":
            wMat = [np.array([0 for _ in range(l)]) for l in self.wMatShape]         
            self.weights= np.array(wMat,dtype=object)
        return 

    def forward(self,data):
        return
    def backward(self):
        return
    def train(self,data):
        for it in range(self.ep):


            pass


        return
    def save(self,path):
        return
    def load(self,path):
        return
    




def main():
    return

print(main())