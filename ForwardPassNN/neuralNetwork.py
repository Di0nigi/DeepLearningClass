import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.power(np.e,(-x)))
    return s
def logLoss(p,y):
    l= -(y*np.log(p)+(1-y)*np.log(1-p))
    return l

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
    def __init__(self,hlayers,dims,learningRate=0.001,epochs=10000,func=sigmoid, initWeights = "zero"):
        self.layers = hlayers
        self.dims = dims
        self.lr = learningRate
        self.ep = epochs
        self.activation = func
        self.weights=[]
        self.biases =[]
        self.netArch = [self.dims[0]]+self.layers+[self.dims[1]]
        self.numLayers=len(hlayers)+2
        self.wMatShape=countWeights(self.dims,self.layers)
        self.biases = np.array([[0.1 for b in range(self.netArch[elem+1])] for elem in range(len(self.netArch)-1)],dtype=object)
        if initWeights == "zero":
            wMat = [np.array([0.1 for _ in range(l)]) for l in self.wMatShape]         
            self.weights= np.array(wMat,dtype=object)
        return 

    def forward(self,data):

        results = []
        nonActivatedRes = []

        for dataElem in data:
            val = dataElem
            for indLayer in range(self.numLayers-1):
                nextLayer=[]
                c=0
                for nelems in range(self.netArch[indLayer+1]):
                    v=0
                    for celems in range(self.netArch[indLayer]):
                        v += val[celems]*self.weights[indLayer][c+celems]
                    c+=self.netArch[indLayer]
                    nextLayer.append(self.activation(v+self.biases[indLayer][nelems]))
                val=nextLayer
            results.append(val)
            
        return np.array(results)
    
    def backward(self):
        return
    def train(self,data):
        dataPoints = data[0]
        dataLabels = data[1]
        for it in range(self.ep):
            predictions = self.forward(dataPoints)
            loss=0
            #print(type(predictions))
            #predictions = list(predictions)
            for i,p in enumerate (predictions): 
                #p = predictions[i]
                loss+=logLoss(p,dataLabels[i])
            loss = loss//len(dataPoints)
            


        return predictions,loss
    def save(self,path):
        return
    def load(self,path):
        return
    




def main():
    N = FeedForwardNN([3,3,2],[2,1],learningRate=0.01,func=sigmoid,initWeights="zero")
    N.train([])
    return

#print(main())