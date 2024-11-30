import numpy as np
import matplotlib.pyplot as plt
import random as rand

def manhattanD(p1,p2):
    d=np.abs(p1[0]-p2[0])+np.abs(p1[1]-p2[1])

    return d
def euclideanD(p1,p2):
    d=(((p2[0]-p1[0])**2)-(p2[1]-p1[1])**2)**0.5
    return d 
def maxOfValueD(p1,p2):
    d=()
    return d
def cosineD(p1, p2):
    dot_product = np.dot(p1, p2)
    norm1 = np.linalg.norm(p1)
    norm2 = np.linalg.norm(p2)

    if norm1 == 0 or norm2== 0:
 
        return np.nan

    cosine_similarity = dot_product / (norm1 * norm2)
    d = 1 - cosine_similarity
    return d






class kNN:
    def __init__(self,data):   
        self.dataSet=data  
        self.k=0   
        self.distance=euclideanD
        return

    def predict(self,p):
        
        self.data=list(sorted(self.dataSet, key= lambda k: self.distance(k[0],p)))
        nearestNeighbours=self.dataSet[:self.k]
        label=max(set(self.dataSet), key=lambda x: self.dataSet.count(x[1]))

        

        return label
    


def main():

    l=[[(0,1),"c"],[(0,4),"b"],[(1,3),"c"],[(3,1),"b"],[(5,1),"b"],[(8,1),"c"],[(2,6),"c"]]
    cl=kNN(l)
    cl.k=2
    r=cl.predict((1,2))
    print(r)









    return

main()