import numpy as np
import random as rand
import matplotlib.pyplot as Matplot
import os



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
        indices=np.arange(len(X))
        np.random.shuffle(indices)
        Xshuffled = [X[i] for i in indices]
        Yshuffled = [y[i] for i in indices]
        size=(n_samples*0.05)
        stInd=0
        Yshuffled


        if self.weights==None:
            self.weights= np.random.random(n_features)
        print(self.weights)
        if self.bias==None:
            self.bias=rand.random()
        print(self.bias)
        
        
        for i in range(self.iters):
            ###testing needed
            ind=stInd+int(size)
            #print(type(stInd))
            #print(type(ind))
            miniBatch= Xshuffled[stInd:ind]
            yminiBatch=Yshuffled[stInd:ind]
            if stInd==len(Xshuffled)-size:
                stInd=0
            else:
                stInd+=int(size)
            self.error=0
            ###testing needed
            for idx, x_i in enumerate(miniBatch):
                linear_out=np.dot(x_i,self.weights)+ self.bias
                y_predict=self.activation_fun(linear_out)
                self.error+=logLoss(y_predict, yminiBatch[idx])
                update= self.lr*(yminiBatch[idx]-y_predict)
                for f in range(len(self.weights)):
                    self.weights[f]+=update*x_i[f]
                self.bias+= update
            self.errors.append(self.error)
            
        return self.bias, self.weights, self.error
    
    def predict(self,X):
        linear_out=np.dot(X,self.weights)+ self.bias
        y_predict=self.activation_fun(linear_out)
        return y_predict
    def uploadWeights(self,l,b):
        self.weights=l
        self.bias=b
        return
    


#######################################testing stuff
def testAccuracy(y_pred, y_actual):
    accuracy =np.sum(y_actual==y_pred)/len(y_actual)
    return accuracy

def logLoss(pred, label):
    epsilon = 1e-15
    pred = np.clip(pred, epsilon, 1 - epsilon)
    i=0
    i=(-label*np.log(pred) - (1-label)*np.log(1-pred))
    return i

def plotErrors(l):
    Matplot.plot(l, [x for x in range(len(l))])
    Matplot.xlabel('iterations')
    Matplot.ylabel('logloss')
    Matplot.title('error')

    Matplot.show()

    return

def visualData(mat,ylist, perceptron):
    
    fig = Matplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    Matplot.scatter(mat[:, 0], mat[:, 1], marker="o", c=ylist)

    x0_1 = np.amin(mat[:, 0])
    x0_2 = np.amax(mat[:, 0])

    x1_1 = (-perceptron.weights[0] * x0_1 - perceptron.bias) / perceptron.weights[1]
    x1_2 = (-perceptron.weights[0] * x0_2 - perceptron.bias) / perceptron.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(mat[:, 1])
    ymax = np.amax(mat[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    Matplot.xlabel('meep')
    Matplot.ylabel('morp')
    Matplot.title('Data')

    Matplot.show()


    return    

def makePoints(l):
        ret=[]
        for x in l:
            x_c=0
            y_c=0
            s=x.split(" ")
            for i in s:
                if i=="meep":
                    x_c+=1
                elif i=="morp":
                    y_c+=1
            ret.append([x_c,y_c])
        ret= np.array(ret)
        
        return ret


def dataParser(dataFile):
    dataList=[]
    with open(dataFile,"r", encoding="utf-8") as f:
        for line in f:
            l=line.split(" 00")
            l[1]=l[1].strip("\n")
            dataList.append(l)
    return dataList

def unlabelData(l):
    unLabeled= [x[0] for x in l]
    normalizedLabel=np.array([0 if x[1]== "sad" else 1 for x in l])
    return unLabeled, normalizedLabel



def main():
    trainDataList, trainLabelList = unlabelData(dataParser(os.path.join(os.getcwd(),"ForwardPassNN\data\dataBatch1.txt")))

    testDataList, testLabelList= unlabelData(dataParser(os.path.join(os.getcwd(),"ForwardPassNN\data\dataBatch.txt")))

    trainData= makePoints(trainDataList)

    testData= makePoints(testDataList)

    Lp = Perceptron(learning_rates=0.001, n_iters=10000)

    tup=Lp.fit(trainData,trainLabelList)

    predictions= Lp.predict(testData)

    accuracy=testAccuracy(sigmoid_step(predictions),testLabelList)
    
    print(tup)
    print(accuracy)
    #print(predictions[-10:-1])
    

    visualData(trainData, trainLabelList, Lp)
    plotErrors(Lp.errors)
    
    return "done"

print(main())