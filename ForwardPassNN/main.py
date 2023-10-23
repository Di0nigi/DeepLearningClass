import matplotlib.pyplot as Matplot
import numpy as np
import os
import neuralNetwork as nn



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

def showData(data,labels):
    fig = Matplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    data=np.array(data)
    Matplot.scatter(data[:, 0], data[:, 1], marker="o", c=labels)
    Matplot.xlabel('meep')
    Matplot.ylabel('morp')
    Matplot.title('Data')

    Matplot.show()


    return



def main():
    trainDataList, trainLabelList = unlabelData(dataParser(os.path.join(os.getcwd(),"ForwardPassNN\data\dataBatch1.txt")))

    testDataList, testLabelList= unlabelData(dataParser(os.path.join(os.getcwd(),"ForwardPassNN\data\dataBatch.txt")))

    trainData= makePoints(trainDataList)

    testData= makePoints(testDataList)

    Fn= nn.FeedForwardNN(2,[2,1])

    #showData(trainData,trainLabelList)

    tup=Fn.train([trainData,trainLabelList])

    predictions= Fn.predict(testData)[0]

    accuracy=testAccuracy(nn.sigmoid_step(predictions),testLabelList)
    
    print(tup)
    print(accuracy)
    print(predictions[-10:-1])
    

    #visualData(trainData, trainLabelList, Lp)
    plotErrors(Fn.error)
    
    return "done"

print(main())