import numpy as np
import matplotlib.pyplot as Matplot
import os

def unit_step(x):
    ret= np.where(x>0,1,0)
    return ret

def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

class perceptron:

    def __init__(self, learning_rates=0.01, n_iters=1000):
        self.lr=learning_rates
        self.iters=n_iters
        self.activation_fun=unit_step
        self.weights=None
        self.bias=None
        return
    def fit(self,X,y):

        n_samples, n_features= X.shape
        self.weights= np.zeros(n_features)
        self.bias=0
        y_ = np.where(y > 0 , 1, 0)
        
        for i in range(self.iters):
            for idx, x_i in enumerate(X):
                linear_out=np.dot(x_i,self.weights)+ self.bias
                y_predict=self.activation_fun(linear_out)
                update= self.lr*(y_[idx]-y_predict)
                self.weights+= update*x_i
                self.bias+= update
        return
    
    def predict(self,X):
        linear_out=np.dot(X,self.weights)+ self.bias
        y_predict=self.activation_fun(linear_out)
        return y_predict


def testAccuracy(y_pred, y_actual):
    accuracy =np.sum(y_actual==y_pred)/len(y_actual)
    return accuracy

def visualData(mat,ylist, perceptron):
    # xPoints= [x[0] for x in mat]
    # yPoints= [y[1] for y in mat]
    # Matplot.scatter(xPoints,yPoints)
    # Matplot.xlabel('meep')
    # Matplot.ylabel('morp')
    # Matplot.title('Data')
    # Matplot.show()
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

#visualData(makePoints(unlabelData(dataParser(os.path.join(os.getcwd(),"PerceptronAlgo\data\dataBatch.txt")))[0]))

def main():
    trainDataList, trainLabelList = unlabelData(dataParser(os.path.join(os.getcwd(),"PerceptronAlgo\data\dataBatch1.txt")))

    testDataList, testLabelList= unlabelData(dataParser(os.path.join(os.getcwd(),"PerceptronAlgo\data\dataBatch.txt")))

    trainData= makePoints(trainDataList)

    testData= makePoints(testDataList)

    P = perceptron(learning_rates=0.001, n_iters=1000)

    P.fit(trainData,trainLabelList)

    predictions= P.predict(testData)

    accuracy=testAccuracy(predictions,testLabelList)

    print(accuracy)

    visualData(trainData, trainLabelList, P)
    
    return "done"

print(main())