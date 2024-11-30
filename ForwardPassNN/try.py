import numpy as np
l=[2,4,2,3]
d=[3,1]



def countWeights(d,l):
    out=[]
    out.append(d[0]*l[0])
    s=0
    for ind in range(len(l)-1):
        s=l[ind]*l[ind+1]
        out.append(s)
    out.append(l[-1]*d[1])
    return out

wMat = [np.array([0 for _ in range(l)]) for l in countWeights(d,l)]   
wMat = np.array(wMat,dtype=object)
print(wMat)
